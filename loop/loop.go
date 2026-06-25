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

// Loop starts asynchronous model and tool execution.
//
// The returned channels carry generated tokens, iteration snapshots, and
// terminal errors respectively. All three channels are closed when execution
// ends. Callers should consume every channel concurrently so execution cannot
// block when one stream's buffer fills.
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
		toolDefinitions, err := ToolDefinitions(a.Tools)
		if err != nil {
			loopErr = err
			errCh <- err
			return
		}

		_, err = a.PromptBuilder.BuildContext(ctx)
		if err != nil {
			loopErr = fmt.Errorf("%w: %w", ErrBuildPrompt, err)
			errCh <- loopErr
			return
		}

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

// Messages returns the completed iterations as ordered conversation messages.
func (a *Loop) Messages() []gaictx.Message {
	var msgs []gaictx.Message

	for _, i := range a.Iterations {
		msgs = append(msgs, i.Messages()...)
	}

	return msgs
}
