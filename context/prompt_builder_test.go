package context

import (
	"context"
	"strings"
	"testing"

	"github.com/lace-ai/gai"
)

type testContextSource struct {
	name   string
	budget int
	text   string
}

func (s *testContextSource) Name() string {
	return s.name
}

func (s *testContextSource) Function(ctx context.Context, tokenBudget int) (Part, error) {
	s.budget = tokenBudget
	return NewTextPart(s.text), nil
}

type emptyConversation struct{}

func (emptyConversation) Messages() []Message {
	return nil
}

type debugTestTokenizer struct{}

func (debugTestTokenizer) ID() string {
	return "debug.test"
}

func (debugTestTokenizer) Tokenize(ctx context.Context, text string) ([]string, error) {
	return strings.Fields(text), nil
}

func (debugTestTokenizer) CountTokens(ctx context.Context, text string) (int, error) {
	return len(strings.Fields(text)), nil
}

func TestNewPromptBuilderFromDefinition(t *testing.T) {
	t.Parallel()

	source := &testContextSource{name: "source", text: "context"}
	builder := New(Definition{
		SystemInstructions: []Part{NewTextPart("system")},
		ContextSources:     []ContextSource{source},
		UserPrompt:         "user",
		TokenBudget:        12,
	})

	if builder.Renderer == nil {
		t.Fatal("expected default renderer")
	}
	if got := builder.GetUserPrompt(); got != "user" {
		t.Fatalf("expected user prompt %q, got %q", "user", got)
	}

	_, err := builder.BuildContext(context.Background())
	if err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	if source.budget != 12 {
		t.Fatalf("expected source token budget 12, got %d", source.budget)
	}

	prompt, err := builder.BuildPrompt(context.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}

	systemIndex := strings.Index(prompt, "system")
	contextIndex := strings.Index(prompt, "context")
	userIndex := strings.Index(prompt, "user")
	if systemIndex < 0 || contextIndex < 0 || userIndex < 0 {
		t.Fatalf("expected prompt to contain system, context, and user prompt: %q", prompt)
	}
	if !(systemIndex < contextIndex && contextIndex < userIndex) {
		t.Fatalf("expected system, context, user prompt order: %q", prompt)
	}
}

func TestPromptBuilderDebugEvents(t *testing.T) {
	t.Parallel()

	var events []gai.DebugEvent
	source := &testContextSource{name: "source", text: "context"}
	builder := New(Definition{
		SystemInstructions: []Part{NewTextPart("system")},
		ContextSources:     []ContextSource{source},
		UserPrompt:         "user",
		TokenBudget:        12,
		Tokenizer:          debugTestTokenizer{},
		DebugSink: func(ctx context.Context, e gai.DebugEvent) {
			events = append(events, e)
		},
	})

	if _, err := builder.BuildContext(context.Background()); err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	if _, err := builder.BuildPrompt(context.Background(), emptyConversation{}); err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}

	want := []string{
		"prompt_builder_context_build_started",
		"prompt_builder_source_included",
		"prompt_builder_context_build_finished",
		"prompt_builder_render_finished",
	}
	for _, name := range want {
		if !hasDebugEvent(events, name) {
			t.Fatalf("expected debug event %q in %+v", name, events)
		}
	}
	for _, event := range events {
		if _, ok := event.Fields["prompt"]; ok {
			t.Fatalf("non-sensitive debug sink should not receive prompt text: %+v", event)
		}
	}
}

func hasDebugEvent(events []gai.DebugEvent, name string) bool {
	for _, event := range events {
		if event.Name == name {
			return true
		}
	}
	return false
}
