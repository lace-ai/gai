package agent

import (
	"context"
	"errors"
	"sync"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
)

var (
	ErrWorkflowAlreadyRun      = errors.New("workflow has already been run")
	ErrWorkflowNotConfigured   = errors.New("workflow is not configured")
	ErrMiddlewareNotConfigured = errors.New("middleware is not configured")
)

// Stream is the streaming output transformed by middleware.
type Stream struct {
	Tokens   <-chan ai.Token
	Statuses <-chan loop.IterationInformation
	Errors   <-chan error
}

// AgentResult is the captured result of one agent execution.
type AgentResult struct {
	Tokens     []ai.Token
	Text       string
	Messages   []gaictx.Message
	Iterations []loop.Iteration
	Errors     []error
}

// StageResult is the named result produced by agent middleware.
type StageResult struct {
	Name   string
	Output OutputPolicy
	Result AgentResult
}

// WorkflowResult is a snapshot of the complete workflow state.
type WorkflowResult struct {
	Input    RunInput
	Primary  AgentResult
	Tokens   []ai.Token
	Text     string
	Stages   []StageResult
	Errors   []error
	Complete bool
}

// MiddlewareContext gives middleware access to the accumulated workflow result.
type MiddlewareContext struct {
	workflow *Workflow
}

// Result returns a safe snapshot of the current workflow state.
func (c *MiddlewareContext) Result() WorkflowResult {
	if c == nil || c.workflow == nil {
		return WorkflowResult{}
	}
	return c.workflow.Result()
}

// Middleware transforms one workflow stream into another.
type Middleware interface {
	Process(ctx context.Context, run *MiddlewareContext, upstream Stream) Stream
}

// MiddlewareFunc adapts a function into Middleware.
type MiddlewareFunc func(ctx context.Context, run *MiddlewareContext, upstream Stream) Stream

func (f MiddlewareFunc) Process(ctx context.Context, run *MiddlewareContext, upstream Stream) Stream {
	return f(ctx, run, upstream)
}

type middlewareValidator interface {
	validate() error
}

// Workflow runs a configured agent loop and its stream middleware.
type Workflow struct {
	Loop       *loop.Loop
	middleware []Middleware

	mu      sync.RWMutex
	started bool
	result  WorkflowResult
}

func newWorkflow(input RunInput, l *loop.Loop, middleware []Middleware) *Workflow {
	input = cloneRunInput(input, false)
	return &Workflow{
		Loop:       l,
		middleware: append([]Middleware(nil), middleware...),
		result:     WorkflowResult{Input: input},
	}
}

func validateMiddleware(middleware []Middleware) error {
	for _, item := range middleware {
		if item == nil {
			return ErrMiddlewareNotConfigured
		}
		if middlewareFunc, ok := item.(MiddlewareFunc); ok && middlewareFunc == nil {
			return ErrMiddlewareNotConfigured
		}
		if validator, ok := item.(middlewareValidator); ok {
			if err := validator.validate(); err != nil {
				return err
			}
		}
	}
	return nil
}

// Run starts the workflow. It can be called only once.
func (w *Workflow) Run(ctx context.Context) (<-chan ai.Token, <-chan loop.IterationInformation, <-chan error) {
	if w == nil || w.Loop == nil {
		return failedStream(ErrWorkflowNotConfigured)
	}

	w.mu.Lock()
	if w.started {
		w.mu.Unlock()
		return failedStream(ErrWorkflowAlreadyRun)
	}
	w.started = true
	w.mu.Unlock()

	tokens, statuses, errs := w.Loop.Loop(ctx)
	stream := w.capturePrimary(ctx, Stream{Tokens: tokens, Statuses: statuses, Errors: errs})
	run := &MiddlewareContext{workflow: w}
	for _, middleware := range w.middleware {
		stream = middleware.Process(ctx, run, stream)
	}
	stream = w.captureFinal(ctx, stream)
	return stream.Tokens, stream.Statuses, stream.Errors
}

// Result returns a safe snapshot. Complete is true after Run's channels close.
func (w *Workflow) Result() WorkflowResult {
	if w == nil {
		return WorkflowResult{}
	}
	w.mu.RLock()
	defer w.mu.RUnlock()
	return cloneWorkflowResult(w.result)
}

func (w *Workflow) capturePrimary(ctx context.Context, upstream Stream) Stream {
	return captureStream(ctx, upstream, func(tokens []ai.Token, errs []error) {
		result := AgentResult{
			Tokens:     cloneTokens(tokens),
			Text:       tokenText(tokens),
			Messages:   append([]gaictx.Message(nil), w.Loop.Messages()...),
			Iterations: append([]loop.Iteration(nil), w.Loop.Iterations...),
			Errors:     append([]error(nil), errs...),
		}
		w.mu.Lock()
		w.result.Primary = result
		w.result.Tokens = cloneTokens(tokens)
		w.result.Text = result.Text
		w.result.Errors = append([]error(nil), errs...)
		w.mu.Unlock()
	})
}

func (w *Workflow) captureFinal(ctx context.Context, upstream Stream) Stream {
	tokens := make(chan ai.Token, 16)
	statuses := make(chan loop.IterationInformation, 16)
	errs := make(chan error)

	var capturedTokens []ai.Token
	var capturedErrs []error
	var wg sync.WaitGroup
	wg.Add(3)
	go func() {
		defer wg.Done()
		for token := range upstream.Tokens {
			capturedTokens = append(capturedTokens, token)
			send(ctx, tokens, token)
		}
	}()
	go func() {
		defer wg.Done()
		for status := range upstream.Statuses {
			send(ctx, statuses, status)
		}
	}()
	go func() {
		defer wg.Done()
		for err := range upstream.Errors {
			if err != nil {
				capturedErrs = append(capturedErrs, err)
			}
		}
	}()
	go func() {
		wg.Wait()
		w.mu.Lock()
		w.result.Tokens = cloneTokens(capturedTokens)
		w.result.Text = tokenText(capturedTokens)
		w.result.Errors = append([]error(nil), capturedErrs...)
		w.result.Complete = true
		w.mu.Unlock()
		close(tokens)
		close(statuses)
		for _, err := range capturedErrs {
			errs <- err
		}
		close(errs)
	}()

	return Stream{Tokens: tokens, Statuses: statuses, Errors: errs}
}

func (w *Workflow) addStage(stage StageResult, tokens []ai.Token, errs []error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.result.Stages = append(w.result.Stages, cloneStageResult(stage))
	w.result.Tokens = cloneTokens(tokens)
	w.result.Text = tokenText(tokens)
	w.result.Errors = append([]error(nil), errs...)
}

func captureStream(ctx context.Context, upstream Stream, completed func([]ai.Token, []error)) Stream {
	tokens := make(chan ai.Token, 16)
	statuses := make(chan loop.IterationInformation, 16)
	errs := make(chan error, 1)

	var capturedTokens []ai.Token
	var capturedErrs []error
	var wg sync.WaitGroup
	wg.Add(3)

	go func() {
		defer wg.Done()
		for token := range upstream.Tokens {
			capturedTokens = append(capturedTokens, token)
			send(ctx, tokens, token)
		}
	}()
	go func() {
		defer wg.Done()
		for status := range upstream.Statuses {
			send(ctx, statuses, status)
		}
	}()
	go func() {
		defer wg.Done()
		for err := range upstream.Errors {
			if err != nil {
				capturedErrs = append(capturedErrs, err)
			}
			send(ctx, errs, err)
		}
	}()
	go func() {
		wg.Wait()
		completed(capturedTokens, capturedErrs)
		close(tokens)
		close(statuses)
		close(errs)
	}()

	return Stream{Tokens: tokens, Statuses: statuses, Errors: errs}
}

func failedStream(err error) (<-chan ai.Token, <-chan loop.IterationInformation, <-chan error) {
	tokens := make(chan ai.Token)
	statuses := make(chan loop.IterationInformation)
	errs := make(chan error, 1)
	close(tokens)
	close(statuses)
	errs <- err
	close(errs)
	return tokens, statuses, errs
}

func send[T any](ctx context.Context, ch chan<- T, value T) {
	select {
	case ch <- value:
	case <-ctx.Done():
	}
}

func tokenText(tokens []ai.Token) string {
	var text []byte
	for _, token := range tokens {
		if token.Type != ai.TokenTypeText {
			continue
		}
		if token.Text != "" {
			text = append(text, token.Text...)
		} else {
			text = append(text, token.Data...)
		}
	}
	return string(text)
}

func cloneRunInput(input RunInput, includeResult bool) RunInput {
	cloned := input
	if input.Meta != nil {
		cloned.Meta = make(map[string]any, len(input.Meta))
		for key, value := range input.Meta {
			cloned.Meta[key] = value
		}
	}
	if !includeResult {
		cloned.Result = nil
	}
	return cloned
}

func cloneTokens(tokens []ai.Token) []ai.Token {
	return append([]ai.Token(nil), tokens...)
}

func cloneAgentResult(result AgentResult) AgentResult {
	result.Tokens = cloneTokens(result.Tokens)
	result.Messages = append([]gaictx.Message(nil), result.Messages...)
	result.Iterations = append([]loop.Iteration(nil), result.Iterations...)
	result.Errors = append([]error(nil), result.Errors...)
	return result
}

func cloneStageResult(stage StageResult) StageResult {
	stage.Result = cloneAgentResult(stage.Result)
	return stage
}

func cloneWorkflowResult(result WorkflowResult) WorkflowResult {
	result.Input = cloneRunInput(result.Input, false)
	result.Primary = cloneAgentResult(result.Primary)
	result.Tokens = cloneTokens(result.Tokens)
	result.Errors = append([]error(nil), result.Errors...)
	result.Stages = append([]StageResult(nil), result.Stages...)
	for i := range result.Stages {
		result.Stages[i] = cloneStageResult(result.Stages[i])
	}
	return result
}
