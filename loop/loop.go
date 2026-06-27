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

// ToolResPreProcessor can inspect or modify a tool response before the loop
// records it and builds the next prompt.
type ToolResPreProcessor interface {
	// Process handles the response produced for req.
	Process(req ai.ToolCall, res *ToolResponse) error
}

// Loop coordinates prompt construction, model generation, and tool execution.
//
// Use New for initialized defaults. A Loop stores run state in Iterations and
// must not be run concurrently or reused without explicitly clearing that
// state.
type Loop struct {
	// Iterations contains completed model/tool interaction rounds.
	Iterations []Iteration
	// Model generates tokens for each iteration.
	Model ai.Model
	// Tools contains the functions available to the model.
	Tools []Tool
	// MaxLoopIterations limits model/tool interaction rounds.
	MaxLoopIterations int
	// MaxTokens limits model output for each generation request.
	MaxTokens int
	// RetryCount is the number of model stream failures retried before stopping.
	RetryCount int
	// PromptBuilder constructs the prompt for each iteration.
	PromptBuilder gaictx.PromptBuilder
	// PreProcessToolRes optionally processes tool responses before they are recorded.
	PreProcessToolRes ToolResPreProcessor
}

// Validate applies default limits and checks required loop dependencies.
func (a *Loop) Validate() error {
	if a == nil {
		return ErrNilAgent
	}
	if a.MaxLoopIterations <= 0 {
		a.MaxLoopIterations = defaultMaxLoopIterations
	}
	if a.RetryCount < 0 {
		a.RetryCount = defaultRetryCount
	}
	if a.Model == nil {
		return ErrModelNotConfigured
	}
	if a.PromptBuilder == nil {
		return ErrPromptNotConfigured
	}
	return nil
}

// New constructs a Loop with default iteration and retry limits.
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

// Run starts asynchronous model and tool execution.
//
// The returned channels carry generated tokens, iteration snapshots, and
// terminal errors respectively. All three channels are closed when execution
// ends. Callers should consume every channel concurrently so execution cannot
// block when one stream's buffer fills.
func (a *Loop) Run(ctx context.Context) (<-chan ai.Token, <-chan IterationInformation, <-chan error) {
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
		ctx, runState := newLoopRunState(ctx, a)
		defer runState.finish()
		defer close(errCh)
		defer close(tokenCh)
		defer close(statusCh)

		toolDefinitions, err := ToolDefinitions(a.Tools)
		if err != nil {
			runState.fail(err)
			errCh <- err
			return
		}

		_, err = a.PromptBuilder.BuildContext(ctx)
		if err != nil {
			errCh <- runState.fail(fmt.Errorf("%w: %w", ErrBuildPrompt, err))
			return
		}

		var iteration Iteration
		for i := range a.MaxLoopIterations {
			iteration = Iteration{Count: i + 1}
			var toolCalls []iterationToolCall

			iterCtx, iterState := runState.startIteration(ctx, iteration.Count)
			iterCtx, cancel := context.WithCancel(iterCtx)
			var iterationErr error

			prompt, err := a.PromptBuilder.BuildPrompt(iterCtx, a)
			if err != nil {
				iterationErr = fmt.Errorf("%w: %w", ErrBuildPrompt, err)
				errCh <- runState.fail(iterationErr)
				cancel()
				iterState.finish(iterationErr)
				return
			}

			request := ai.AIRequest{
				Prompt:    prompt,
				MaxTokens: a.MaxTokens,
				Tools:     toolDefinitions,
			}
			input := a.PromptBuilder.Input()
			if input.User != nil {
				iteration.UserMessage = &gaictx.Message{Role: gaictx.RoleUser, Content: input.User}
			}

			tokens := a.Model.GenerateStream(iterCtx, request)

			retrying := false
			for t := range tokens {
				if t.Err != nil {
					if runState.canRetry(a.RetryCount) {
						retrying = true
						break
					} else {
						iterationErr = fmt.Errorf("%w limit:%v error: %v", ErrMaxRetries, a.RetryCount, t.Err)
						errCh <- runState.fail(iterationErr)
						cancel()
						iterState.finish(iterationErr)
						return
					}
				}

				if t.Type == ai.TokenTypeToolCall && t.ToolCall != nil {
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
				runState.recordToken(t)
				iterState.recordToken(t)
			}

			if retrying {
				runState.retry()
				iterState.markRetrying(runState.retryCount)
				statusCh <- runState.retryStatus(iteration)
				cancel()
				iterState.finish(nil)
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
				errCh <- runState.fail(iterationErr)
				cancel()
				iterState.finish(iterationErr)
				return
			}

			iterState.recordToolResponses(iteration)

			a.Iterations = append(a.Iterations, iteration)
			runState.resetRetries()

			statusCh <- IterationInformation{
				Iteration:      iteration,
				IterationCount: iteration.Count,
				PartCount:      len(iteration.Parts),
				RetryCount:     runState.retryCount,
			}
			if len(toolCalls) == 0 {
				cancel()
				iterState.markFinal()
				iterState.finish(nil)
				return
			}
			cancel()
			iterState.finish(iterationErr)
		}

		errCh <- runState.fail(fmt.Errorf("%w: limit=%d", ErrMaxIterations, a.MaxLoopIterations))
	}()

	return tokenCh, statusCh, errCh
}

// Messages returns the completed iterations as ordered conversation messages.
func (a *Loop) Messages() []gaictx.Message {
	var msgs []gaictx.Message

	for _, i := range a.Iterations {
		msgs = append(msgs, i.Messages()...)
	}

	return msgs
}
