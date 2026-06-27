package loop

import (
	"context"
	"errors"
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
// The returned channel carries every token, retry, iteration, and terminal
// event in the exact order it occurred. Callers must consume the channel until
// it closes or cancel ctx.
func (a *Loop) Run(ctx context.Context) <-chan Event {
	events := make(chan Event, 32)

	if err := a.Validate(); err != nil {
		events <- ErrorEvent(err)
		close(events)
		return events
	}

	go func() {
		ctx, runState := newLoopRunState(ctx, a)
		defer runState.finish()
		defer close(events)

		toolDefinitions, err := ToolDefinitions(a.Tools)
		if err != nil {
			sendLoopError(ctx, events, runState, err)
			return
		}

		_, err = a.PromptBuilder.BuildContext(ctx)
		if err != nil {
			sendLoopError(ctx, events, runState, fmt.Errorf("%w: %w", ErrBuildPrompt, err))
			return
		}

		for i := range a.MaxLoopIterations {
			iteration := Iteration{Count: i + 1}
			var userMessage *gaictx.Message
			if i == 0 {
				input := a.PromptBuilder.Input()
				if input.User != nil {
					userMessage = &gaictx.Message{Role: gaictx.RoleUser, Content: input.User}
				}
			}
			var toolCalls []iterationToolCall
			var iterState *loopIterationState
			var iterationErr error
			var iterCtx context.Context
			var cancel context.CancelFunc

			for attempt := 1; ; attempt++ {
				attemptIteration := Iteration{Count: iteration.Count}
				toolCalls = nil

				iterCtx, iterState = runState.startIteration(ctx, iteration.Count, attempt)
				iterCtx, cancel = context.WithCancel(iterCtx)
				attemptID := iterState.attemptID()
				if err := sendEvent(ctx, events, AttemptStartEvent(iteration.Count, attemptID, runState.retryCount)); err != nil {
					sendLoopError(ctx, events, runState, err)
					cancel()
					iterState.finish(err)
					return
				}

				prompt, err := a.PromptBuilder.BuildPrompt(iterCtx, a)
				if err != nil {
					iterationErr = fmt.Errorf("%w: %w", ErrBuildPrompt, err)
					sendLoopError(ctx, events, runState, iterationErr)
					cancel()
					iterState.finish(iterationErr)
					return
				}

				request := ai.AIRequest{
					Prompt:    prompt,
					MaxTokens: a.MaxTokens,
					Tools:     toolDefinitions,
				}

				tokens := a.Model.GenerateStream(iterCtx, request)

				retrying := false
				for t := range tokens {
					if t.Err != nil {
						if errors.Is(t.Err, context.Canceled) || errors.Is(t.Err, context.DeadlineExceeded) {
							iterationErr = t.Err
							sendLoopError(ctx, events, runState, iterationErr)
							cancel()
							iterState.finish(iterationErr)
							return
						}

						if runState.canRetry(a.RetryCount) {
							retrying = true
							break
						}

						iterationErr = fmt.Errorf("%w: limit=%d: %w", ErrMaxRetries, a.RetryCount, t.Err)
						sendLoopError(ctx, events, runState, iterationErr)
						cancel()
						iterState.finish(iterationErr)
						return
					}

					if t.Type == ai.TokenTypeToolCall && t.ToolCall != nil {
						attemptIteration.AppendToken(t)

						toolReq := t.ToolCall
						partIdx := len(attemptIteration.Parts) - 1

						toolCalls = append(toolCalls, iterationToolCall{
							id:       partIdx,
							toolCall: *toolReq,
						})
					} else {
						attemptIteration.AppendToken(t)
					}
					runState.recordToken(t)
					iterState.recordToken(t)
					if err := sendEvent(ctx, events, TokenEvent(iteration.Count, attemptID, runState.retryCount, t)); err != nil {
						sendLoopError(ctx, events, runState, err)
						cancel()
						iterState.finish(err)
						return
					}
				}

				if !retrying {
					attemptIteration.UserMessage = userMessage
					iteration = attemptIteration
					break
				}

				runState.retry()
				iterState.markRetrying(runState.retryCount)
				if err := sendEvent(ctx, events, RetryEvent(iteration.Count, attemptID, runState.retryCount, attemptIteration)); err != nil {
					sendLoopError(ctx, events, runState, err)
					cancel()
					iterState.finish(err)
					return
				}
				cancel()
				iterState.finish(nil)
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
				sendLoopError(ctx, events, runState, iterationErr)
				cancel()
				iterState.finish(iterationErr)
				return
			}

			iterState.recordToolResponses(iteration)

			a.Iterations = append(a.Iterations, iteration)
			runState.resetRetries()

			if err := sendEvent(ctx, events, IterationDoneEvent(iteration, iterState.attemptID(), runState.retryCount)); err != nil {
				sendLoopError(ctx, events, runState, err)
				cancel()
				iterState.finish(err)
				return
			}
			if len(toolCalls) == 0 {
				cancel()
				iterState.markFinal()
				iterState.finish(nil)
				_ = sendEvent(ctx, events, DoneEvent())
				return
			}
			cancel()
			iterState.finish(iterationErr)
		}

		sendLoopError(ctx, events, runState, fmt.Errorf("%w: limit=%d", ErrMaxIterations, a.MaxLoopIterations))
	}()

	return events
}

func sendLoopError(ctx context.Context, events chan<- Event, state *loopRunState, err error) {
	if state != nil {
		state.fail(err)
	}
	_ = sendEvent(ctx, events, ErrorEvent(err))
}

// Messages returns the completed iterations as ordered conversation messages.
func (a *Loop) Messages() []gaictx.Message {
	var msgs []gaictx.Message

	for _, i := range a.Iterations {
		msgs = append(msgs, i.Messages()...)
	}

	return msgs
}
