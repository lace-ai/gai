package context_test

import (
	stdcontext "context"
	"errors"
	"strings"
	"testing"

	aicontext "github.com/lace-ai/gai/context"
)

type emptyConversation struct{}

func (emptyConversation) Messages() []aicontext.Message {
	return nil
}

func TestPromptBuilderBuildsStructuredPrompt(t *testing.T) {
	t.Parallel()

	sourceCalled := false
	builder := aicontext.NewPromptBuilder().
		System(
			aicontext.StaticPart("base", "base system").RequiredPart().WithTokens(12),
			aicontext.StaticPart("dynamic", "dynamic system").WithTokens(4),
		).
		ContextSource("memory", aicontext.SourceFunc(func(ctx stdcontext.Context, conv aicontext.Conversation) ([]aicontext.Part, error) {
			sourceCalled = true
			return []aicontext.Part{
				aicontext.StaticPart("history", "stored history"),
				aicontext.StaticPart("current-loop", "current messages"),
			}, nil
		}), true).
		User(aicontext.StaticPart("request", "answer this").RequiredPart())

	prompt, err := builder.BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}

	if !sourceCalled {
		t.Fatal("expected context source to be called")
	}

	assertContainsAll(t, prompt.System, "system prompt", "base system", "dynamic system")
	assertOrdered(t, prompt.System, "base system", "dynamic system")
	assertContainsNone(t, prompt.System, "system prompt", "stored history", "current messages", "answer this")

	assertContainsAll(t, prompt.Context, "context prompt", "stored history", "current messages")
	assertOrdered(t, prompt.Context, "stored history", "current messages")
	assertContainsNone(t, prompt.Context, "context prompt", "base system", "dynamic system", "answer this")

	assertContainsAll(t, prompt.Prompt, "user prompt", "answer this")
	assertContainsNone(t, prompt.Prompt, "user prompt", "base system", "dynamic system", "stored history", "current messages")
}

func TestPromptBuilderEscapesPartNames(t *testing.T) {
	t.Parallel()

	prompt, err := aicontext.NewPromptBuilder().
		User(aicontext.StaticPart(`request "<tag>"`, "answer this")).
		BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}

	assertContainsAll(t, prompt.Prompt, "user prompt", `request &#34;&lt;tag&gt;&#34;`, "answer this")
	assertContainsNone(t, prompt.Prompt, "user prompt", `request "<tag>"`)
}

func assertContainsAll(t *testing.T, text, name string, values ...string) {
	t.Helper()

	for _, value := range values {
		if !strings.Contains(text, value) {
			t.Fatalf("expected %s to contain %q: %q", name, value, text)
		}
	}
}

func assertContainsNone(t *testing.T, text, name string, values ...string) {
	t.Helper()

	for _, value := range values {
		if strings.Contains(text, value) {
			t.Fatalf("expected %s not to contain %q: %q", name, value, text)
		}
	}
}

func assertOrdered(t *testing.T, text string, values ...string) {
	t.Helper()

	previous := -1
	for _, value := range values {
		index := strings.Index(text, value)
		if index == -1 {
			t.Fatalf("expected %q to contain %q", text, value)
		}
		if index < previous {
			t.Fatalf("expected values to be ordered as %q in %q", values, text)
		}
		previous = index
	}
}

func TestPromptBuilderSourceFailurePolicy(t *testing.T) {
	t.Parallel()

	sourceErr := errors.New("source unavailable")
	failingSource := aicontext.SourceFunc(func(ctx stdcontext.Context, conv aicontext.Conversation) ([]aicontext.Part, error) {
		return nil, sourceErr
	})

	prompt, err := aicontext.NewPromptBuilder().
		Context(aicontext.StaticPart("kept", "kept context")).
		ContextSource("optional-rag", failingSource, false).
		User(aicontext.StaticPart("request", "answer this").RequiredPart()).
		BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("optional source should be skipped, got error: %v", err)
	}
	if strings.Contains(prompt.Context, "optional-rag") {
		t.Fatalf("optional failing source should not render: %q", prompt.Context)
	}
	if !strings.Contains(prompt.Context, "kept context") {
		t.Fatalf("expected static context to remain: %q", prompt.Context)
	}

	_, err = aicontext.NewPromptBuilder().
		ContextSource("required-rag", failingSource, true).
		User(aicontext.StaticPart("request", "answer this").RequiredPart()).
		BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err == nil {
		t.Fatal("expected required source error")
	}
	if !errors.Is(err, aicontext.ErrPromptSource) {
		t.Fatalf("expected ErrPromptSource, got %v", err)
	}
}

func TestPromptBuilderUsesCustomRenderer(t *testing.T) {
	t.Parallel()

	renderer := sectionNameRenderer{}
	prompt, err := aicontext.NewPromptBuilder().
		Renderer(renderer).
		System(aicontext.StaticPart("base", "system")).
		Context(aicontext.StaticPart("ctx", "context")).
		User(aicontext.StaticPart("request", "user")).
		BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}

	if prompt.System != "system:base" || prompt.Context != "context:ctx" || prompt.Prompt != "user:request" {
		t.Fatalf("custom renderer was not used: %+v", prompt)
	}
}

type sectionNameRenderer struct{}

func (sectionNameRenderer) Render(section aicontext.Section, parts []aicontext.Part) string {
	names := make([]string, 0, len(parts))
	for _, part := range parts {
		names = append(names, part.Name)
	}
	return string(section) + ":" + strings.Join(names, ",")
}
