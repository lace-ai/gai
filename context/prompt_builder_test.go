package context

import (
	"context"
	"strings"
	"testing"
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
