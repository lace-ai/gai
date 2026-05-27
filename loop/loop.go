package loop

import (
	"context"
	"fmt"
	"sync"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

const (
	defaultMaxLoopIterations = 8
	defaultRetryCount        = 3
)

type ToolResPreProcessor interface {
	Process(req ai.ToolCall, res *ToolResponse) error
}

type Loop struct {
	Iterations        []Iteration
	Model             ai.Model
	Tools             []Tool
	MaxLoopIterations int
	MaxTokens         int
	RetryCount        int
	PromptBuilder     gaictx.PromptBuilder
	PreProcessToolRes ToolResPreProcessor
}

func (a *Loop) Validate() error {
	if a == nil {
		return ErrNilAgent
	}
	if a.MaxLoopIterations <= 0 {
		a.MaxLoopIterations = defaultMaxLoopIterations
	}
	if a.Model == nil {
		return ErrModelNotConfigured
	}
	if a.PromptBuilder == nil {
		return ErrPromptNotConfigured
	}
	return nil
}

func New(model ai.Model, tools []Tool, promptBuilder gaictx.PromptBuilder, toolResPreProcessor ToolResPreProcessor) *Loop {
	agent := &Loop{
		Model:             model,
		Tools:             tools,
		MaxLoopIterations: defaultMaxLoopIterations,
		RetryCount:        defaultRetryCount,
		PromptBuilder:     promptBuilder,
		PreProcessToolRes: toolResPreProcessor,
	}
	return agent
}

type iterationToolCall struct {
	id       int
	toolCall ai.ToolCall
}

func (a *Loop) Loop(ctx context.Context) (<-chan ai.Token, <-chan IterationInformation, <-chan error) {
	errCh := make(chan error, 1)
	tokenCh := make(chan ai.Token, 16)
	statusCh := make(chan IterationInformation, 16)

	if err := a.Validate(); err != nil {
		errCh <- err
		close(errCh)
		close(tokenCh)
		close(statusCh)
		return tokenCh, statusCh, errCh
	}

	go func() {
		defer close(errCh)
		defer close(tokenCh)
		defer close(statusCh)

		retryCount := 0
		completedToolCalls := map[string]struct{}{}
		var promptSession gaictx.PromptSession
		var currentPrompt ai.Prompt
		incrementalPrompt := false

		if builder, ok := a.PromptBuilder.(gaictx.IncrementalPromptBuilder); ok {
			session, err := builder.StartPrompt(ctx)
			if err != nil {
				errCh <- fmt.Errorf("%w: %w", ErrBuildPrompt, err)
				return
			}
			promptSession = session
			currentPrompt = session.Prompt()
			incrementalPrompt = true
		}

		var iteration Iteration
		for i := range a.MaxLoopIterations {
			iteration = Iteration{Count: i + 1}
			var toolCalls []iterationToolCall

			iterCtx, cancel := context.WithCancel(ctx)

			prompt := currentPrompt
			if !incrementalPrompt {
				var err error
				prompt, err = a.PromptBuilder.BuildPrompt(iterCtx, a)
				if err != nil {
					errCh <- fmt.Errorf("%w: %w", ErrBuildPrompt, err)
					cancel()
					return
				}
			}

			request := ai.AIRequest{
				Prompt:    prompt,
				MaxTokens: a.MaxTokens,
			}
			iteration.Request = &request

			tokens := a.Model.GenerateStream(iterCtx, request)

			retrying := false
			for t := range tokens {
				if t.Err != nil {
					if retryCount < a.RetryCount {
						retrying = true
						break
					} else {
						errCh <- fmt.Errorf("%w limit:%v error: %v", ErrMaxRetries, a.RetryCount, t.Err)
						cancel()
						return
					}
				}

				if t.Type == ai.TokenTypeToolCall && t.ToolCall != nil {
					signature := toolCallSignature(*t.ToolCall)
					if _, ok := completedToolCalls[signature]; ok {
						continue
					}

					iteration.AppendToken(t)
					tokenCh <- t

					toolReq := t.ToolCall
					partIdx := len(iteration.Parts) - 1

					toolCalls = append(toolCalls, iterationToolCall{
						id:       partIdx,
						toolCall: *toolReq,
					})
				} else {
					iteration.AppendToken(t)
					tokenCh <- t
				}
			}

			if retrying {
				retryCount++
				statusCh <- IterationInformation{
					IterationCount: iteration.Count,
					PartCount:      len(iteration.Parts),
					RetryCount:     retryCount,
				}
				cancel()
				continue
			}

			wg := sync.WaitGroup{}
			for _, tc := range toolCalls {
				wg.Add(1)
				go func(tc iterationToolCall) {
					defer wg.Done()

					toolRes := CallTool(&tc.toolCall, a.Tools)
					if a.PreProcessToolRes != nil {
						if err := a.PreProcessToolRes.Process(tc.toolCall, toolRes); err != nil {
							errCh <- fmt.Errorf("pre-processing tool response failed: %w", err)
							return
						}
					}

					iteration.Parts[tc.id].ToolResp = toolRes
				}(tc)
			}
			wg.Wait()

			for _, tc := range toolCalls {
				part := iteration.Parts[tc.id]
				if part.ToolResp != nil {
					completedToolCalls[toolCallSignature(tc.toolCall)] = struct{}{}
				}
			}

			a.Iterations = append(a.Iterations, iteration)
			if retrying {
				statusCh <- IterationInformation{
					IterationCount: iteration.Count,
					PartCount:      len(iteration.Parts),
					RetryCount:     retryCount,
				}
				cancel()
				continue
			}
			retryCount = 0

			statusCh <- IterationInformation{
				Iteration:      iteration,
				IterationCount: iteration.Count,
				PartCount:      len(iteration.Parts),
				RetryCount:     retryCount,
			}
			if len(toolCalls) == 0 && !retrying {
				cancel()
				return
			}
			if incrementalPrompt {
				var err error
				currentPrompt, err = promptSession.AppendMessages(iterCtx, iteration.DeltaMessages())
				if err != nil {
					errCh <- fmt.Errorf("%w: %w", ErrBuildPrompt, err)
					cancel()
					return
				}
			}
			cancel()
		}

		errCh <- fmt.Errorf("%w: limit=%d", ErrMaxIterations, a.MaxLoopIterations)
	}()

	return tokenCh, statusCh, errCh
}

func (a *Loop) Messages() []gaictx.Message {
	var msgs []gaictx.Message

	for _, i := range a.Iterations {
		msgs = append(msgs, i.Messages()...)
	}

	return msgs
}
