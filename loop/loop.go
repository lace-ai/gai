package loop

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

const (
	defaultMaxLoopIterations = 8
	defaultRetryCount        = 3
)

// ToolTransportMode controls whether a loop sends tool definitions through the
// provider-native AIRequest.Tools field. It does not affect Loop.Tools, which
// always remains available to resolve and execute tool calls.
type ToolTransportMode uint8

const (
	// ToolTransportNative sends Loop.Tools as AIRequest.Tools. This is the
	// default to preserve direct loop.New compatibility.
	ToolTransportNative ToolTransportMode = iota
	// ToolTransportText omits AIRequest.Tools for prompt-rendered tool protocols.
	ToolTransportText
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
	// ToolChoice controls whether and how the model may call Tools.
	ToolChoice ai.ToolChoice
	// ToolTransport controls whether Tools are serialized into AIRequest.Tools.
	// The default is ToolTransportNative; Tools remain executable in either mode.
	ToolTransport ToolTransportMode
	// MaxLoopIterations limits model/tool interaction rounds.
	MaxLoopIterations int
	// MaxTokens limits model output for each generation request.
	MaxTokens int
	// ResponseFormat requests the output shape for each model generation.
	ResponseFormat ai.ResponseFormat
	// Reasoning configures model reasoning/thinking behavior for each model generation.
	Reasoning ai.ReasoningConfig
	// RetryCount is the number of model stream failures retried before stopping.
	RetryCount int
	// PromptBuilder constructs the prompt for each iteration.
	PromptBuilder gaictx.PromptBuilder
	// ToolResponseProcessor optionally processes tool responses after they are recorded on the iteration and before it is persisted.
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
	if err := l.ResponseFormat.Validate(); err != nil {
		return err
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
func renderedPromptRequest(prompt string, maxTokens int, tools []ai.ToolDefinition, toolChoice ai.ToolChoice, responseFormat ai.ResponseFormat, reasoning ai.ReasoningConfig) ai.AIRequest {
	return ai.AIRequest{
		Prompt:         prompt,
		MaxTokens:      maxTokens,
		Tools:          tools,
		ToolChoice:     toolChoice,
		ResponseFormat: responseFormat,
		Reasoning:      reasoning,
	}
}

// NativeMessages returns the provider-neutral history for completed iterations.
// Prompt builders combine it with their own base user message.
func (l *Loop) NativeMessages() []ai.RequestMessage {
	var messages []ai.RequestMessage
	if l == nil {
		return nil
	}
	iterations := l.Iterations
	for _, iteration := range iterations {
		var text string
		var toolCalls []ai.RequestToolCall
		var toolResults []ai.RequestMessage
		for _, part := range iteration.Parts {
			switch part.Type {
			case IterationTypeResponse:
				if part.Response != nil && part.Response.Text != "" {
					text += part.Response.Text
				}
			case IterationTypeToolCall, IterationTypeToolError:
				if part.ToolReq == nil {
					continue
				}
				toolCalls = append(toolCalls, ai.RequestToolCall{
					ID:               part.ToolReq.ID,
					Name:             part.ToolReq.Name,
					Arguments:        append([]byte(nil), part.ToolReq.Args...),
					ThoughtSignature: append([]byte(nil), part.ToolReq.ThoughtSignature...),
				})
				if part.ToolResp != nil {
					result := ai.RequestToolResult{ToolCallID: part.ToolReq.ID, Name: part.ToolReq.Name, Content: part.ToolResp.TextValue()}
					if err := part.ToolResp.ErrorValue(); err != nil {
						result.Content = err.Error()
						result.IsError = true
					}
					toolResults = append(toolResults, ai.RequestMessage{Role: ai.RequestMessageRoleTool, ToolResult: &result})
				}
			}
		}
		if text != "" || len(toolCalls) > 0 {
			messages = append(messages, ai.RequestMessage{Role: ai.RequestMessageRoleAssistant, Text: text, ToolCalls: toolCalls})
			messages = append(messages, toolResults...)
		}
	}
	return messages
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

		var (
			toolDefinitions []ai.ToolDefinition
			err             error
		)
		if l.ToolTransport == ToolTransportNative {
			toolDefinitions, err = ToolDefinitions(l.Tools)
			if err != nil {
				if cancelErr := cancellationError(ctx, err); cancelErr != nil {
					sendLoopCanceled(ctx, events, runState, cancelErr)
					return
				}
				sendLoopError(ctx, events, runState, err)
				return
			}
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

		requiredToolCallSatisfied := l.ToolChoice.Mode != ai.ToolChoiceRequired
		for i := range l.MaxLoopIterations {
			iteration := Iteration{Count: i + 1}
			userMessage := userMessageForIteration(l.PromptBuilder, i)
			var toolCalls []pendingToolCall
			var deferredTokens []ai.Token
			var iterState *loopIterationState
			var iterationErr error
			var iterCtx context.Context
			var cancel context.CancelFunc
			deferTokens := l.ToolTransport == ToolTransportText && !requiredToolCallSatisfied

			for attempt := 1; ; attempt++ {
				attemptIteration := Iteration{Count: iteration.Count, UserMessage: userMessage}
				toolCalls = nil
				deferredTokens = nil

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

				var prompt string
				var nativeMessages []ai.RequestMessage
				if builder, ok := l.PromptBuilder.(gaictx.NativeMessageBuilder); ok {
					prompt, nativeMessages, err = builder.BuildRequest(iterCtx, l)
				} else {
					prompt, err = l.PromptBuilder.BuildPrompt(iterCtx, l)
				}
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

				toolChoice := l.ToolChoice
				if l.ToolChoice.Mode == ai.ToolChoiceRequired && requiredToolCallSatisfied {
					toolChoice = ai.ToolChoice{Mode: ai.ToolChoiceAuto}
				}
				if l.ToolTransport == ToolTransportText {
					// Text transport exposes tools through the rendered prompt, not
					// AIRequest.Tools. Provider-native tool choice is therefore invalid.
					toolChoice = ai.ToolChoice{}
				}
				request := renderedPromptRequest(prompt, l.MaxTokens, toolDefinitions, toolChoice, l.ResponseFormat, l.Reasoning)
				request.Messages = nativeMessages

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
					if deferTokens {
						deferredTokens = append(deferredTokens, t)
						continue
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

			if deferTokens && len(toolCalls) == 0 {
				// A text-transport response that does not satisfy a required tool
				// call is not part of the conversation and must not be observable.
				cancel()
				iterState.finish(iterationErr)
				continue
			}
			for _, token := range deferredTokens {
				runState.recordToken(token)
				iterState.recordToken(token)
				if err := sendEvent(ctx, events, TokenEvent(iteration.Count, iterState.attemptID(), runState.retryCount, token)); err != nil {
					if cancelErr := cancellationError(iterCtx, err); cancelErr != nil {
						sendAttemptCanceled(ctx, events, runState, iteration.Count, iterState.attemptID(), runState.retryCount, &iteration, cancelErr)
						iterState.markCanceled(cancelErr)
					} else {
						sendAttemptError(ctx, events, runState, iteration.Count, iterState.attemptID(), runState.retryCount, &iteration, err)
					}
					cancel()
					iterState.finish(nil)
					return
				}
			}

			if err := l.executeToolCalls(iterCtx, &iteration, toolCalls, events, iteration.Count, iterState.attemptID(), runState.retryCount); err != nil {
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
			if hasPermittedToolCall(toolCalls, l.Tools) {
				requiredToolCallSatisfied = true
			}
			runState.resetRetries()
			if l.ToolTransport == ToolTransportText && len(toolCalls) == 0 && !requiredToolCallSatisfied {
				// Text transport cannot rely on provider enforcement. Keep the
				// run active until the required tool call has been observed.
				cancel()
				iterState.finish(iterationErr)
				continue
			}
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

// hasPermittedToolCall reports whether the model requested a valid configured
// tool. Rejected calls, including calls for unavailable tools, do not satisfy a
// required tool choice.
func hasPermittedToolCall(toolCalls []pendingToolCall, tools []Tool) bool {
	for _, toolCall := range toolCalls {
		if err := toolCall.call.Validate(); err != nil {
			continue
		}
		for _, tool := range tools {
			if tool != nil && tool.Name() == toolCall.call.Name {
				return true
			}
		}
	}
	return false
}

// executeToolCalls records tool responses on iteration. Tool execution
// failures are stored in ToolResponse.Err and are not returned. Only framework
// or tool-response processing failures are returned.
func (l *Loop) executeToolCalls(ctx context.Context, iteration *Iteration, toolCalls []pendingToolCall, events chan<- Event, iterationCount, attemptID, retryCount int) error {
	var wg sync.WaitGroup
	var toolErr error
	var toolErrMu sync.Mutex

	for _, tc := range toolCalls {
		if events != nil {
			if err := sendEvent(ctx, events, ToolStartEvent(iterationCount, attemptID, retryCount, tc.call)); err != nil {
				return err
			}
		}
		wg.Add(1)
		go func(tc pendingToolCall) {
			defer wg.Done()

			started := time.Now()
			toolRes := callObservedTool(ctx, tc.call, l.Tools)
			duration := time.Since(started)
			iteration.Parts[tc.partIndex].ToolResp = toolRes
			if l.ToolResponseProcessor != nil {
				if err := l.ToolResponseProcessor.Process(tc.call, toolRes); err != nil {
					processErr := fmt.Errorf("%w: %w", ErrToolResponseProcess, err)
					toolErrMu.Lock()
					if toolErr == nil {
						toolErr = processErr
					}
					toolErrMu.Unlock()
					if events != nil {
						if err := sendEvent(ctx, events, ToolErrorEvent(iterationCount, attemptID, retryCount, tc.call, toolRes, duration, processErr)); err != nil {
							toolErrMu.Lock()
							if toolErr == nil {
								toolErr = err
							}
							toolErrMu.Unlock()
						}
					}
					return
				}
			}
			if events != nil {
				if err := sendEvent(ctx, events, ToolResultEvent(iterationCount, attemptID, retryCount, tc.call, toolRes, duration)); err != nil {
					toolErrMu.Lock()
					if toolErr == nil {
						toolErr = err
					}
					toolErrMu.Unlock()
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
