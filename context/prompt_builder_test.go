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

	builder := aicontext.NewPromptBuilder().
		System(
			aicontext.StaticPart("base", "base system").RequiredPart().WithTokens(12),
			aicontext.StaticPart("dynamic", "dynamic system").WithTokens(4),
		).
		ContextSource("memory", aicontext.SourceFunc(func(ctx stdcontext.Context, conv aicontext.Conversation) ([]aicontext.Part, error) {
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

	wantSystem := `<system>
<part name="base" tokens="12" required="true">
base system
</part>
<part name="dynamic" tokens="4" required="false">
dynamic system
</part>
</system>
`
	if prompt.System != wantSystem {
		t.Fatalf("unexpected system prompt:\nwant: %q\ngot:  %q", wantSystem, prompt.System)
	}
	if !strings.Contains(prompt.Context, `<part name="history" tokens="0" required="true">`) {
		t.Fatalf("expected history part in context prompt: %q", prompt.Context)
	}
	if !strings.Contains(prompt.Context, "stored history") || !strings.Contains(prompt.Context, "current messages") {
		t.Fatalf("expected source parts to render in context prompt: %q", prompt.Context)
	}
	if !strings.Contains(prompt.Prompt, `<user>`) || !strings.Contains(prompt.Prompt, "answer this") {
		t.Fatalf("expected user section to render into Prompt: %q", prompt.Prompt)
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
