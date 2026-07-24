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

// ToolResponseProcessor can inspect or modify a tool response before the loop
// records it and builds the next prompt. Implementations must be safe for
// concurrent use.
type ToolResponseProcessor interface {
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
	// ToolResponseProcessor optionally processes tool responses before they are recorded.
	ToolResponseProcessor ToolResponseProcessor
}

// Validate applies default iteration limits and checks required loop dependencies.
func (l *Loop) Validate() error {
	if l == nil {
		return ErrNilLoop
	}
	if l.MaxLoopIterations <= 0 {
		l.MaxLoopIterations = defaultMaxLoopIterations
	}
	if l.Model == nil {
		return ErrModelNotConfigured
	}
	if l.PromptBuilder == nil {
		return ErrPromptNotConfigured
	}
	return nil
}

// New constructs a Loop with default iteration and retry limits.
func New(model ai.Model, tools []Tool, promptBuilder gaictx.PromptBuilder, toolResponseProcessor ToolResponseProcessor) *Loop {
	l := &Loop{
		Model:                 model,
		Tools:                 tools,
		MaxLoopIterations:     defaultMaxLoopIterations,
		RetryCount:            defaultRetryCount,
		PromptBuilder:         promptBuilder,
		ToolResponseProcessor: toolResponseProcessor,
	}
	return l
}

type pendingToolCall struct {
	partIndex int
	call      ai.ToolCall
}

// renderedPromptRequest creates the compatibility request used by the loop.
// Conversation state remains in Prompt until a provider-native message path is
// introduced deliberately; this boundary keeps that future change separate
// from the current rendered-prompt behavior.
func renderedPromptRequest(prompt string, maxTokens int, tools []ai.ToolDefinition) ai.AIRequest {
	return ai.AIRequest{
		Prompt:    prompt,
		MaxTokens: maxTokens,
		Tools:     tools,
	}
}

// Run starts asynchronous model and tool execution.
//
// The returned channel carries every token, retry, iteration, and terminal
// event in the exact order it occurred. Tokens are forwarded in real time;
// when an attempt is retried, consumers that keep visible token state must
// discard that attempt's tokens on its RetryEvent. Callers must consume the
// channel until it closes or cancel ctx.
func (l *Loop) Run(ctx context.Context) <-chan Event {
	events := make(chan Event, 32)

	if err := l.Validate(); err != nil {
		events <- ErrorEvent(err)
		close(events)
		return events
	}

	go func() {
		ctx, runState := newLoopRunState(ctx, l)
		defer runState.finish()
		defer close(events)
		if err := ctx.Err(); err != nil {
			sendLoopCanceled(ctx, events, runState, err)
			return
		}

		toolDefinitions, err := ToolDefinitions(l.Tools)
		if err != nil {
			if cancelErr := cancellationError(ctx, err); cancelErr != nil {
				sendLoopCanceled(ctx, events, runState, cancelErr)
				return
			}
			sendLoopError(ctx, events, runState, err)
			return
		}

		_, err = l.PromptBuilder.BuildContext(ctx)
		if err != nil {
			if cancelErr := cancellationError(ctx, err); cancelErr != nil {
				sendLoopCanceled(ctx, events, runState, cancelErr)
				return
			}
			sendLoopError(ctx, events, runState, fmt.Errorf("%w: %w", ErrBuildPrompt, err))
			return
		}
		if err := ctx.Err(); err != nil {
			sendLoopCanceled(ctx, events, runState, err)
			return
		}

		for i := range l.MaxLoopIterations {
			iteration := Iteration{Count: i + 1}
			userMessage := userMessageForIteration(l.PromptBuilder, i)
			var toolCalls []pendingToolCall
			var iterState *loopIterationState
			var iterationErr error
			var iterCtx context.Context
			var cancel context.CancelFunc

			for attempt := 1; ; attempt++ {
				attemptIteration := Iteration{Count: iteration.Count, UserMessage: userMessage}
				toolCalls = nil

				iterCtx, iterState = runState.startIteration(ctx, iteration.Count, attempt)
				iterCtx, cancel = context.WithCancel(iterCtx)
				attemptID := iterState.attemptID()
				if err := iterCtx.Err(); err != nil {
					sendAttemptCanceled(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, err)
					cancel()
					iterState.markCanceled(err)
					iterState.finish(nil)
					return
				}
				if err := sendEvent(ctx, events, AttemptStartEvent(iteration.Count, attemptID, runState.retryCount)); err != nil {
					if cancelErr := cancellationError(iterCtx, err); cancelErr != nil {
						sendAttemptCanceled(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, cancelErr)
						iterState.markCanceled(cancelErr)
					} else {
						sendAttemptError(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, err)
					}
					cancel()
					iterState.finish(nil)
					return
				}

				prompt, err := l.PromptBuilder.BuildPrompt(iterCtx, l)
				if err != nil {
					if cancelErr := cancellationError(iterCtx, err); cancelErr != nil {
						sendAttemptCanceled(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, cancelErr)
						cancel()
						iterState.markCanceled(cancelErr)
						iterState.finish(nil)
						return
					}
					iterationErr = fmt.Errorf("%w: %w", ErrBuildPrompt, err)
					sendAttemptError(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, iterationErr)
					cancel()
					iterState.finish(iterationErr)
					return
				}

				request := renderedPromptRequest(prompt, l.MaxTokens, toolDefinitions)

				tokens := l.Model.GenerateStream(iterCtx, request)

				retrying := false
				for t := range tokens {
					if t.Err != nil {
						if cancelErr := cancellationError(iterCtx, t.Err); cancelErr != nil {
							sendAttemptCanceled(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, cancelErr)
							cancel()
							iterState.markCanceled(cancelErr)
							iterState.finish(nil)
							return
						}

						if runState.canRetry(l.RetryCount) {
							retrying = true
							break
						}

						iterationErr = fmt.Errorf("%w: limit=%d: %w", ErrMaxRetries, l.RetryCount, t.Err)
						sendAttemptError(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, iterationErr)
						cancel()
						iterState.finish(iterationErr)
						return
					}

					if t.Type == ai.TokenTypeToolCall && t.ToolCall != nil {
						attemptIteration.AppendToken(t)

						toolReq := t.ToolCall
						partIdx := len(attemptIteration.Parts) - 1

						toolCalls = append(toolCalls, pendingToolCall{
							partIndex: partIdx,
							call:      *toolReq,
						})
					} else {
						attemptIteration.AppendToken(t)
					}
					runState.recordToken(t)
					iterState.recordToken(t)
					if err := sendEvent(ctx, events, TokenEvent(iteration.Count, attemptID, runState.retryCount, t)); err != nil {
						if cancelErr := cancellationError(iterCtx, err); cancelErr != nil {
							sendAttemptCanceled(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, cancelErr)
							iterState.markCanceled(cancelErr)
						} else {
							sendAttemptError(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, err)
						}
						cancel()
						iterState.finish(nil)
						return
					}
				}

				if err := iterCtx.Err(); err != nil {
					sendAttemptCanceled(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, err)
					cancel()
					iterState.markCanceled(err)
					iterState.finish(nil)
					return
				}
				if !retrying {
					iteration = attemptIteration
					break
				}

				runState.retry()
				iterState.recordIteration(attemptIteration)
				iterState.markRetrying(runState.retryCount)
				if err := sendEvent(ctx, events, RetryEvent(iteration.Count, attemptID, runState.retryCount, attemptIteration)); err != nil {
					if cancelErr := cancellationError(iterCtx, err); cancelErr != nil {
						sendAttemptCanceled(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, cancelErr)
						iterState.markCanceled(cancelErr)
					} else {
						sendAttemptError(ctx, events, runState, iteration.Count, attemptID, runState.retryCount, &attemptIteration, err)
					}
					cancel()
					iterState.finish(nil)
					return
				}
				cancel()
				iterState.finish(nil)
			}

			if err := l.executeToolCalls(iterCtx, &iteration, toolCalls); err != nil {
				if cancelErr := cancellationError(iterCtx, err); cancelErr != nil {
					sendAttemptCanceled(ctx, events, runState, iteration.Count, iterState.attemptID(), runState.retryCount, &iteration, cancelErr)
					cancel()
					iterState.markCanceled(cancelErr)
					iterState.finish(nil)
					return
				}
				iterationErr = err
				sendAttemptError(ctx, events, runState, iteration.Count, iterState.attemptID(), runState.retryCount, &iteration, iterationErr)
				cancel()
				iterState.finish(iterationErr)
				return
			}
			if err := iterCtx.Err(); err != nil {
				sendAttemptCanceled(ctx, events, runState, iteration.Count, iterState.attemptID(), runState.retryCount, &iteration, err)
				cancel()
				iterState.markCanceled(err)
				iterState.finish(nil)
				return
			}

			iterState.recordToolResponses(iteration)

			attemptID := iterState.attemptID()
			retryCount := runState.retryCount

			if err := sendEvent(ctx, events, IterationDoneEvent(iteration, attemptID, retryCount)); err != nil {
				if cancelErr := cancellationError(iterCtx, err); cancelErr != nil {
					sendAttemptCanceled(ctx, events, runState, iteration.Count, attemptID, retryCount, &iteration, cancelErr)
					iterState.markCanceled(cancelErr)
				} else {
					sendAttemptError(ctx, events, runState, iteration.Count, attemptID, retryCount, &iteration, err)
				}
				cancel()
				iterState.finish(nil)
				return
			}
			l.Iterations = append(l.Iterations, iteration)
			runState.resetRetries()
			if len(toolCalls) == 0 {
				cancel()
				iterState.markFinal()
				iterState.finish(nil)
				if err := sendEvent(ctx, events, DoneEvent()); err != nil {
					if cancelErr := cancellationError(ctx, err); cancelErr != nil {
						sendLoopCanceled(ctx, events, runState, cancelErr)
					}
				}
				return
			}
			cancel()
			iterState.finish(iterationErr)
		}

		if err := ctx.Err(); err != nil {
			sendLoopCanceled(ctx, events, runState, err)
			return
		}
		sendLoopError(ctx, events, runState, fmt.Errorf("%w: limit=%d", ErrMaxIterations, l.MaxLoopIterations))
	}()

	return events
}

func userMessageForIteration(promptBuilder gaictx.PromptBuilder, index int) *gaictx.Message {
	if index != 0 || promptBuilder == nil {
		return nil
	}
	input := promptBuilder.Input()
	if input.User == nil {
		return nil
	}
	return &gaictx.Message{Role: gaictx.RoleUser, Content: input.User}
}

// executeToolCalls records tool responses on iteration. Tool execution
// failures are stored in ToolResponse.Err and are not returned. Only framework
// or tool-response processing failures are returned.
func (l *Loop) executeToolCalls(ctx context.Context, iteration *Iteration, toolCalls []pendingToolCall) error {
	var wg sync.WaitGroup
	var toolErr error
	var toolErrMu sync.Mutex

	for _, tc := range toolCalls {
		wg.Add(1)
		go func(tc pendingToolCall) {
			defer wg.Done()

			toolRes := CallTool(ctx, &tc.call, l.Tools)
			iteration.Parts[tc.partIndex].ToolResp = toolRes
			if l.ToolResponseProcessor != nil {
				if err := l.ToolResponseProcessor.Process(tc.call, toolRes); err != nil {
					toolErrMu.Lock()
					if toolErr == nil {
						toolErr = fmt.Errorf("%w: %w", ErrToolResponseProcess, err)
					}
					toolErrMu.Unlock()
					return
				}
			}
		}(tc)
	}
	wg.Wait()

	return toolErr
}

func sendLoopError(ctx context.Context, events chan<- Event, state *loopRunState, err error) {
	if state != nil {
		state.fail(err)
	}
	_ = sendEvent(ctx, events, ErrorEvent(err))
}

func sendAttemptError(ctx context.Context, events chan<- Event, state *loopRunState, iterationCount, attemptID, retryCount int, attemptIteration *Iteration, err error) {
	if state != nil {
		state.fail(err)
	}
	_ = sendEvent(ctx, events, AttemptErrorEvent(iterationCount, attemptID, retryCount, attemptIteration, err))
}

func sendLoopCanceled(ctx context.Context, events chan<- Event, state *loopRunState, err error) {
	if state != nil {
		state.cancel(err)
	}
	sendTerminalEvent(ctx, events, CanceledEvent(err))
}

func sendAttemptCanceled(ctx context.Context, events chan<- Event, state *loopRunState, iterationCount, attemptID, retryCount int, attemptIteration *Iteration, err error) {
	if state != nil {
		state.cancel(err)
	}
	sendTerminalEvent(ctx, events, AttemptCanceledEvent(iterationCount, attemptID, retryCount, attemptIteration, err))
}

func sendTerminalEvent(_ context.Context, events chan<- Event, event Event) {
	select {
	case events <- event:
	default:
	}
}

func cancellationError(ctx context.Context, err error) error {
	if ctx != nil && ctx.Err() != nil {
		return ctx.Err()
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}
	return nil
}

// Messages returns the completed iterations as ordered conversation messages.
func (l *Loop) Messages() []gaictx.Message {
	var msgs []gaictx.Message

	for _, i := range l.Iterations {
		msgs = append(msgs, i.Messages()...)
	}

	return msgs
}
