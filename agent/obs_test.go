package agent_test

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/testutil/mocks"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestAgentWorkflowEmitsLifecycleEventsAndSpans(t *testing.T) {
	previousProvider := otel.GetTracerProvider()
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	otel.SetTracerProvider(provider)
	defer func() {
		_ = provider.Shutdown(context.Background())
		otel.SetTracerProvider(previousProvider)
	}()

	sink := &agentDebugSink{}
	post := agent.New(agent.Definition{
		Name:  "post",
		Model: &mocks.MockModel{Responses: []mocks.MockModelResponse{{Res: ai.AIResponse{Text: "post"}}}},
		Prompt: func(_ context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
	})
	primary := agent.New(agent.Definition{
		Name:      "primary",
		Model:     &mocks.MockModel{Responses: []mocks.MockModelResponse{{Res: ai.AIResponse{Text: "primary"}}}},
		DebugSink: sink,
		Prompt: func(_ context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
		Middleware: []agent.Middleware{agent.NewAgentMiddleware(post, agent.AgentMiddlewareConfig{
			Output: agent.PreserveOutput,
		})},
	})

	workflow, err := primary.NewRun(t.Context(), agent.RunInput{
		ID:     "run-1",
		Prompt: gaictx.PromptInput{User: gaictx.NewTextContent("question")},
		Meta:   map[string]any{"session_id": "session-1"},
	})
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	consumed := consumeWorkflow(t, workflow)
	if len(consumed.errs) != 0 {
		t.Fatalf("workflow errors: %v", consumed.errs)
	}

	wantEvents := []string{
		"agent_run_created",
		"agent_workflow_started",
		"agent_primary_finished",
		"agent_middleware_started",
		"agent_middleware_finished",
		"agent_workflow_finished",
	}
	for _, name := range wantEvents {
		if !sink.hasEvent(name) {
			t.Errorf("missing debug event %q; got %v", name, sink.names())
		}
	}
	created, ok := sink.event("agent_run_created")
	if !ok {
		t.Fatal("missing run-created event")
	}
	if _, leaked := created.Fields["user_input"]; leaked {
		t.Fatalf("non-sensitive sink received input text: %+v", created.Fields)
	}

	spanNames := map[string]bool{}
	for _, span := range recorder.Ended() {
		spanNames[span.Name()] = true
	}
	for _, name := range []string{"agent.run.create", "agent.workflow.run", "agent.middleware.run"} {
		if !spanNames[name] {
			t.Errorf("missing span %q; got %v", name, spanNames)
		}
	}
}

func TestAgentObservabilityReportsCreationAndMiddlewareFailures(t *testing.T) {
	sink := &agentDebugSink{}
	_, err := agent.New(agent.Definition{Name: "broken", DebugSink: sink}).NewRun(t.Context(), textRunInput("question"))
	if err == nil {
		t.Fatal("expected run creation failure")
	}
	if event, ok := sink.event("agent_run_creation_failed"); !ok || event.Err == nil {
		t.Fatalf("missing creation failure event: %+v", event)
	}

	mapErr := errors.New("map input")
	post := agent.New(agent.Definition{
		Name:  "post",
		Model: &mocks.MockModel{},
		Prompt: func(_ context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
	})
	primary := agent.New(agent.Definition{
		Name:      "primary",
		Model:     &mocks.MockModel{Responses: []mocks.MockModelResponse{{Res: ai.AIResponse{Text: "primary"}}}},
		DebugSink: sink,
		Prompt: func(_ context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{}, nil
		},
		Middleware: []agent.Middleware{agent.NewAgentMiddleware(post, agent.AgentMiddlewareConfig{
			ErrorPolicy: agent.RecordError,
			MapInput: func(context.Context, agent.WorkflowResult) (agent.RunInput, error) {
				return agent.RunInput{}, mapErr
			},
		})},
	})
	workflow, err := primary.NewRun(t.Context(), textRunInput("question"))
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	consumed := consumeWorkflow(t, workflow)
	if len(consumed.errs) != 0 {
		t.Fatalf("record-only middleware failure propagated: %v", consumed.errs)
	}
	event, ok := sink.event("agent_middleware_failed")
	if !ok || !errors.Is(event.Err, mapErr) {
		t.Fatalf("missing middleware failure event: %+v", event)
	}
}

type agentDebugSink struct {
	mu     sync.Mutex
	events []gai.DebugEvent
}

func (s *agentDebugSink) Emit(_ context.Context, event gai.DebugEvent) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, event)
}

func (*agentDebugSink) IncludeSensitiveData() bool { return false }

func (s *agentDebugSink) hasEvent(name string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, event := range s.events {
		if event.Name == name {
			return true
		}
	}
	return false
}

func (s *agentDebugSink) event(name string) (gai.DebugEvent, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, event := range s.events {
		if event.Name == name {
			return event, true
		}
	}
	return gai.DebugEvent{}, false
}

func (s *agentDebugSink) names() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	names := make([]string, len(s.events))
	for i, event := range s.events {
		names[i] = event.Name
	}
	return names
}
