package agent

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
)

var (
	ErrMiddlewareAgentNotConfigured = errors.New("middleware agent is not configured")
	ErrMiddlewareNameMissing        = errors.New("middleware name is missing")
	ErrMiddlewareOutputInvalid      = errors.New("middleware output policy is invalid")
	ErrMiddlewareFailureInvalid     = errors.New("middleware failure policy is invalid")
)

// OutputPolicy controls how an agent middleware transforms upstream tokens.
type OutputPolicy uint8

const (
	// PreserveOutput forwards upstream tokens and records the middleware agent's
	// result without exposing its tokens to the caller.
	PreserveOutput OutputPolicy = iota
	// AppendOutput forwards upstream tokens followed by middleware-agent tokens.
	AppendOutput
	// ReplaceOutput buffers upstream tokens and, when the stage runs, emits
	// middleware-agent tokens instead.
	ReplaceOutput
)

// FailurePolicy controls whether an agent-middleware failure fails the workflow.
type FailurePolicy uint8

const (
	// PropagateFailure sends middleware failures through the workflow error stream
	// and includes them in WorkflowResult.Errors.
	PropagateFailure FailurePolicy = iota
	// RecordFailure records middleware failures in StageResult without failing the
	// surrounding workflow.
	RecordFailure
)

// MiddlewareInputMapper projects the typed upstream workflow state into the
// ordinary RunInput accepted by a nested agent.
type MiddlewareInputMapper func(ctx context.Context, result WorkflowResult) (RunInput, error)

// AgentMiddlewareConfig configures an agent as stream middleware.
type AgentMiddlewareConfig struct {
	// Name identifies the stage in WorkflowResult.Stages. The agent name is used
	// when Name is empty.
	Name string
	// Output controls how the nested agent changes visible workflow tokens.
	Output OutputPolicy
	// MapInput controls exactly what the nested agent receives. When nil, the
	// current visible text, original run ID, and metadata are forwarded.
	MapInput MiddlewareInputMapper
	// Failure controls whether nested-agent and input-mapping failures propagate
	// to the surrounding workflow. The zero value is PropagateFailure.
	Failure FailurePolicy
	// ShouldRun overrides the default success-only policy. It receives the full
	// upstream result and may enable stages such as failure auditing.
	ShouldRun func(result WorkflowResult) bool
}

// AgentMiddleware runs an agent after its upstream stream completes.
type AgentMiddleware struct {
	agent  *Agent
	config AgentMiddlewareConfig
}

// NewAgentMiddleware adapts an ordinary agent into workflow middleware. Use
// AgentMiddlewareConfig.MapInput when the nested agent needs a deliberate
// projection of the upstream workflow state.
func NewAgentMiddleware(agent *Agent, config AgentMiddlewareConfig) *AgentMiddleware {
	return &AgentMiddleware{agent: agent, config: config}
}

func (m *AgentMiddleware) validate() error {
	if m == nil || m.agent == nil {
		return ErrMiddlewareAgentNotConfigured
	}
	if m.name() == "" {
		return ErrMiddlewareNameMissing
	}
	if m.config.Output > ReplaceOutput {
		return fmt.Errorf("%w: %d", ErrMiddlewareOutputInvalid, m.config.Output)
	}
	if m.config.Failure > RecordFailure {
		return fmt.Errorf("%w: %d", ErrMiddlewareFailureInvalid, m.config.Failure)
	}
	return nil
}

// Process implements Middleware.
func (m *AgentMiddleware) Process(ctx context.Context, run *MiddlewareContext, upstream Stream) Stream {
	tokens := make(chan ai.Token, 16)
	statuses := make(chan loop.IterationInformation, 16)
	errs := make(chan error, 1)

	go func() {
		defer close(tokens)
		defer close(statuses)
		defer close(errs)

		upstreamTokens, upstreamErrs := consumeUpstream(ctx, upstream, tokens, statuses, errs, m.config.Output != ReplaceOutput)
		result := run.Result()
		result.Tokens = cloneTokens(upstreamTokens)
		result.Text = tokenText(upstreamTokens)
		result.Errors = append([]error(nil), upstreamErrs...)

		shouldRun := len(upstreamErrs) == 0
		if m.config.ShouldRun != nil {
			shouldRun = m.config.ShouldRun(result)
		}
		if !shouldRun {
			if m.config.Output == ReplaceOutput {
				for _, token := range upstreamTokens {
					send(ctx, tokens, token)
				}
			}
			return
		}

		postInput := RunInput{
			ID:   result.Input.ID,
			Text: result.Text,
			Meta: cloneRunInput(result.Input).Meta,
		}
		var err error
		if m.config.MapInput != nil {
			postInput, err = m.config.MapInput(ctx, result)
		}
		if err != nil {
			m.finishSetupFailure(ctx, run, tokens, errs, upstreamTokens, upstreamErrs, err)
			return
		}
		postWorkflow, err := m.agent.NewRun(ctx, postInput)
		if err != nil {
			m.finishSetupFailure(ctx, run, tokens, errs, upstreamTokens, upstreamErrs, err)
			return
		}

		postTokens, postStatuses, postErrs := postWorkflow.Run(ctx)
		stageTokens, stageErrs := consumePostAgent(ctx, postTokens, postStatuses, postErrs, tokens, errs, m.config.Output != PreserveOutput, m.config.Failure == PropagateFailure)
		postResult := postWorkflow.Result()
		stage := StageResult{
			Name:   m.name(),
			Output: m.config.Output,
			Result: AgentResult{
				Tokens:     cloneTokens(stageTokens),
				Text:       tokenText(stageTokens),
				Messages:   append([]gaictx.Message(nil), postResult.Primary.Messages...),
				Iterations: append([]loop.Iteration(nil), postResult.Primary.Iterations...),
				Errors:     append([]error(nil), stageErrs...),
			},
		}
		visible := visibleTokens(m.config.Output, upstreamTokens, stageTokens)
		workflowErrs := append([]error(nil), upstreamErrs...)
		if m.config.Failure == PropagateFailure {
			workflowErrs = append(workflowErrs, stageErrs...)
		}
		run.workflow.addStage(stage, visible, workflowErrs)
	}()

	return Stream{Tokens: tokens, Statuses: statuses, Errors: errs}
}

func (m *AgentMiddleware) finishSetupFailure(
	ctx context.Context,
	run *MiddlewareContext,
	tokens chan<- ai.Token,
	errs chan<- error,
	upstreamTokens []ai.Token,
	upstreamErrs []error,
	err error,
) {
	if m.config.Failure == PropagateFailure {
		send(ctx, errs, err)
	}
	if m.config.Output == ReplaceOutput {
		for _, token := range upstreamTokens {
			send(ctx, tokens, token)
		}
	}
	stage := StageResult{
		Name:   m.name(),
		Output: m.config.Output,
		Result: AgentResult{Errors: []error{err}},
	}
	workflowErrs := append([]error(nil), upstreamErrs...)
	if m.config.Failure == PropagateFailure {
		workflowErrs = append(workflowErrs, err)
	}
	run.workflow.addStage(stage, cloneTokens(upstreamTokens), workflowErrs)
}

func (m *AgentMiddleware) name() string {
	if m == nil {
		return ""
	}
	if m.config.Name != "" {
		return m.config.Name
	}
	if m.agent != nil {
		return m.agent.def.Name
	}
	return ""
}

func consumeUpstream(
	ctx context.Context,
	stream Stream,
	tokens chan<- ai.Token,
	statuses chan<- loop.IterationInformation,
	errs chan<- error,
	forwardTokens bool,
) ([]ai.Token, []error) {
	var capturedTokens []ai.Token
	var capturedErrs []error
	var wg sync.WaitGroup
	wg.Add(3)
	go func() {
		defer wg.Done()
		for token := range stream.Tokens {
			capturedTokens = append(capturedTokens, token)
			if forwardTokens {
				send(ctx, tokens, token)
			}
		}
	}()
	go func() {
		defer wg.Done()
		for status := range stream.Statuses {
			send(ctx, statuses, status)
		}
	}()
	go func() {
		defer wg.Done()
		for err := range stream.Errors {
			if err != nil {
				capturedErrs = append(capturedErrs, err)
			}
			send(ctx, errs, err)
		}
	}()
	wg.Wait()
	return capturedTokens, capturedErrs
}

func consumePostAgent(
	ctx context.Context,
	tokens <-chan ai.Token,
	statuses <-chan loop.IterationInformation,
	errs <-chan error,
	outputTokens chan<- ai.Token,
	outputErrs chan<- error,
	forwardTokens bool,
	forwardErrors bool,
) ([]ai.Token, []error) {
	var capturedTokens []ai.Token
	var capturedErrs []error
	var wg sync.WaitGroup
	wg.Add(3)
	go func() {
		defer wg.Done()
		for token := range tokens {
			capturedTokens = append(capturedTokens, token)
			if forwardTokens {
				send(ctx, outputTokens, token)
			}
		}
	}()
	go func() {
		defer wg.Done()
		for range statuses {
		}
	}()
	go func() {
		defer wg.Done()
		for err := range errs {
			if err != nil {
				capturedErrs = append(capturedErrs, err)
			}
			if forwardErrors {
				send(ctx, outputErrs, err)
			}
		}
	}()
	wg.Wait()
	return capturedTokens, capturedErrs
}

func visibleTokens(policy OutputPolicy, upstream, stage []ai.Token) []ai.Token {
	switch policy {
	case AppendOutput:
		visible := cloneTokens(upstream)
		return append(visible, stage...)
	case ReplaceOutput:
		return cloneTokens(stage)
	default:
		return cloneTokens(upstream)
	}
}
