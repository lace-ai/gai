package loop

import (
	"context"
	"fmt"
	"sync"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"go.opentelemetry.io/otel/attribute"
)

const (
	defaultMaxLoopIterations = 8
	defaultRetryCount        = 3
)

const loopTracerName = "github.com/lace-ai/gai/loop"

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
		ctx, loopSpan := gai.StartOperationSpan(ctx, loopTracerName, "loop", "loop.operation", "run",
			attribute.Int("loop.max_iterations", a.MaxLoopIterations),
			attribute.Int("loop.retry_limit", a.RetryCount),
			attribute.Int("loop.max_tokens", a.MaxTokens),
			attribute.Int("loop.tool_count", len(a.Tools)),
			attribute.String("ai.model", a.Model.Name()),
		)
		var loopErr error
		iterationCount := 0
		totalTokenCount := 0
		totalToolCallCount := 0
		incrementalPrompt := false
		defer func() {
			loopSpan.SetAttributes(
				attribute.Int("loop.iteration_count", iterationCount),
				attribute.Int("loop.token_count", totalTokenCount),
				attribute.Int("loop.tool_call_count", totalToolCallCount),
				attribute.Bool("loop.incremental_prompt", incrementalPrompt),
			)
			gai.EndSpan(loopSpan, loopErr)
		}()
		defer close(errCh)
		defer close(tokenCh)
		defer close(statusCh)

		retryCount := 0
		completedToolCalls := map[string]struct{}{}

		a.PromptBuilder.BuildContext(ctx)

		var iteration Iteration
		for i := range a.MaxLoopIterations {
			iteration = Iteration{Count: i + 1}
			iterationCount = iteration.Count
			var toolCalls []iterationToolCall

			iterCtx, iterationSpan := gai.StartOperationSpan(ctx, loopTracerName, "loop", "loop.operation", "iteration",
				attribute.Int("loop.iteration", iteration.Count),
				attribute.Bool("loop.incremental_prompt", incrementalPrompt),
			)
			iterCtx, cancel := context.WithCancel(iterCtx)
			var iterationErr error

			prompt, err := a.PromptBuilder.BuildPrompt(iterCtx, a)
			if err != nil {
				iterationErr = fmt.Errorf("%w: %w", ErrBuildPrompt, err)
				loopErr = iterationErr
				errCh <- iterationErr
				cancel()
				gai.EndSpan(iterationSpan, iterationErr)
				return
			}

			request := ai.AIRequest{
				Prompt:    prompt,
				MaxTokens: a.MaxTokens,
			}
			iteration.Request = a.PromptBuilder.GetUserPrompt()

			tokens := a.Model.GenerateStream(iterCtx, request)

			retrying := false
			for t := range tokens {
				if t.Err != nil {
					if retryCount < a.RetryCount {
						retrying = true
						break
					} else {
						iterationErr = fmt.Errorf("%w limit:%v error: %v", ErrMaxRetries, a.RetryCount, t.Err)
						loopErr = iterationErr
						errCh <- iterationErr
						cancel()
						gai.EndSpan(iterationSpan, iterationErr)
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
					totalTokenCount++

					toolReq := t.ToolCall
					partIdx := len(iteration.Parts) - 1

					toolCalls = append(toolCalls, iterationToolCall{
						id:       partIdx,
						toolCall: *toolReq,
					})
					totalToolCallCount++
				} else {
					iteration.AppendToken(t)
					tokenCh <- t
					totalTokenCount++
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
				iterationSpan.SetAttributes(
					attribute.Bool("loop.retrying", true),
					attribute.Int("loop.part_count", len(iteration.Parts)),
					attribute.Int("loop.retry_count", retryCount),
					attribute.Int("loop.tool_call_count", len(toolCalls)),
				)
				gai.EndSpan(iterationSpan, nil)
				continue
			}

			wg := sync.WaitGroup{}
			var preProcessErr error
			var preProcessErrMu sync.Mutex
			for _, tc := range toolCalls {
				wg.Add(1)
				go func(tc iterationToolCall) {
					defer wg.Done()

					toolRes := CallTool(iterCtx, &tc.toolCall, a.Tools)
					if a.PreProcessToolRes != nil {
						if err := a.PreProcessToolRes.Process(tc.toolCall, toolRes); err != nil {
							preProcessErrMu.Lock()
							if preProcessErr == nil {
								preProcessErr = fmt.Errorf("pre-processing tool response failed: %w", err)
							}
							preProcessErrMu.Unlock()
							return
						}
					}

					iteration.Parts[tc.id].ToolResp = toolRes
				}(tc)
			}
			wg.Wait()
			if preProcessErr != nil {
				iterationErr = preProcessErr
				loopErr = iterationErr
				errCh <- iterationErr
				cancel()
				gai.EndSpan(iterationSpan, iterationErr)
				return
			}

			for _, tc := range toolCalls {
				part := iteration.Parts[tc.id]
				if part.ToolResp != nil {
					completedToolCalls[toolCallSignature(tc.toolCall)] = struct{}{}
				}
			}
			toolErrorCount := 0
			for _, part := range iteration.Parts {
				if part.ToolResp != nil && part.ToolResp.Err != nil {
					toolErrorCount++
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
				iterationSpan.SetAttributes(
					attribute.Bool("loop.retrying", true),
					attribute.Int("loop.part_count", len(iteration.Parts)),
					attribute.Int("loop.retry_count", retryCount),
					attribute.Int("loop.tool_call_count", len(toolCalls)),
					attribute.Int("loop.tool_error_count", toolErrorCount),
				)
				gai.EndSpan(iterationSpan, iterationErr)
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
				iterationSpan.SetAttributes(
					attribute.Bool("loop.final_iteration", true),
					attribute.Int("loop.part_count", len(iteration.Parts)),
					attribute.Int("loop.retry_count", retryCount),
					attribute.Int("loop.tool_call_count", len(toolCalls)),
					attribute.Int("loop.tool_error_count", toolErrorCount),
				)
				gai.EndSpan(iterationSpan, nil)
				return
			}
			cancel()
			iterationSpan.SetAttributes(
				attribute.Int("loop.part_count", len(iteration.Parts)),
				attribute.Int("loop.retry_count", retryCount),
				attribute.Int("loop.tool_call_count", len(toolCalls)),
				attribute.Int("loop.tool_error_count", toolErrorCount),
			)
			gai.EndSpan(iterationSpan, iterationErr)
		}

		loopErr = fmt.Errorf("%w: limit=%d", ErrMaxIterations, a.MaxLoopIterations)
		errCh <- loopErr
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
