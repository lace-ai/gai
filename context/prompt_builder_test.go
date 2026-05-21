package context_test

import (
	stdcontext "context"
	"errors"
	"strings"
	"testing"

	"github.com/lace-ai/gai"
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
		System("base", "base system", aicontext.Required(), aicontext.Tokens(12)).
		System("dynamic", "dynamic system", aicontext.Tokens(4)).
		Source(aicontext.SectionContext, "memory", aicontext.SourceFunc(func(ctx stdcontext.Context, view aicontext.PromptView, budget aicontext.SourceBudget) ([]aicontext.Part, error) {
			sourceCalled = true
			if _, ok := view.Entry("base"); !ok {
				t.Fatal("expected source view to expose full configured plan")
			}
			return []aicontext.Part{
				aicontext.NewPart("history", "stored history"),
				aicontext.NewPart("current-loop", "current messages"),
			}, nil
		}), aicontext.Required()).
		User("request", "answer this", aicontext.Required())

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

	trace := builder.LastTrace()
	if len(trace.Entries) != 4 {
		t.Fatalf("expected four traced entries, got %d", len(trace.Entries))
	}
	if got := len(trace.Parts[aicontext.SectionContext]); got != 2 {
		t.Fatalf("expected two traced context parts, got %d", got)
	}
}

func TestPromptBuilderEscapesPartIDs(t *testing.T) {
	t.Parallel()

	prompt, err := aicontext.NewPromptBuilder().
		User(`request "<tag>"`, "answer this").
		BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}

	assertContainsAll(t, prompt.Prompt, "user prompt", `request &#34;&lt;tag&gt;&#34;`, "answer this")
	assertContainsNone(t, prompt.Prompt, "user prompt", `request "<tag>"`)
}

func TestPromptBuilderRejectsDuplicateIDs(t *testing.T) {
	t.Parallel()

	_, err := aicontext.NewPromptBuilder().
		System("same", "system").
		User("same", "user").
		BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err == nil {
		t.Fatal("expected duplicate ID error")
	}
	if !errors.Is(err, aicontext.ErrPromptEntryID) {
		t.Fatalf("expected ErrPromptEntryID, got %v", err)
	}
}

func TestPromptBuilderRejectsDuplicateEmittedPartIDs(t *testing.T) {
	t.Parallel()

	_, err := aicontext.NewPromptBuilder().
		System("base", "system").
		Source(aicontext.SectionContext, "dup-source", aicontext.SourceFunc(func(ctx stdcontext.Context, view aicontext.PromptView, budget aicontext.SourceBudget) ([]aicontext.Part, error) {
			return []aicontext.Part{aicontext.NewPart("base", "duplicate")}, nil
		}), aicontext.Required()).
		BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err == nil {
		t.Fatal("expected duplicate emitted part ID error")
	}
	if !errors.Is(err, aicontext.ErrPromptEntryID) {
		t.Fatalf("expected ErrPromptEntryID, got %v", err)
	}
}

func TestPromptBuilderSourceFailurePolicy(t *testing.T) {
	t.Parallel()

	sourceErr := errors.New("source unavailable")
	failingSource := aicontext.SourceFunc(func(ctx stdcontext.Context, view aicontext.PromptView, budget aicontext.SourceBudget) ([]aicontext.Part, error) {
		return nil, sourceErr
	})

	builder := aicontext.NewPromptBuilder().
		Context("kept", "kept context").
		Source(aicontext.SectionContext, "optional-rag", failingSource, aicontext.Optional()).
		User("request", "answer this", aicontext.Required())

	prompt, err := builder.BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("optional source should be skipped, got error: %v", err)
	}
	if strings.Contains(prompt.Context, "optional-rag") {
		t.Fatalf("optional failing source should not render: %q", prompt.Context)
	}
	if !strings.Contains(prompt.Context, "kept context") {
		t.Fatalf("expected static context to remain: %q", prompt.Context)
	}
	if got := traceEntryStatus(t, builder.LastTrace(), "optional-rag"); got != "skipped" {
		t.Fatalf("expected optional source to be traced as skipped, got %q", got)
	}

	_, err = aicontext.NewPromptBuilder().
		Source(aicontext.SectionContext, "required-rag", failingSource, aicontext.Required()).
		User("request", "answer this", aicontext.Required()).
		BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err == nil {
		t.Fatal("expected required source error")
	}
	if !errors.Is(err, aicontext.ErrPromptSource) {
		t.Fatalf("expected ErrPromptSource, got %v", err)
	}
}

func TestPromptBuilderSourceCanInspectWholePlan(t *testing.T) {
	t.Parallel()

	prompt, err := aicontext.NewPromptBuilder().
		System("base", "base system", aicontext.Required(), aicontext.Meta("role", "base")).
		Source(aicontext.SectionContext, "conditional", aicontext.SourceFunc(func(ctx stdcontext.Context, view aicontext.PromptView, budget aicontext.SourceBudget) ([]aicontext.Part, error) {
			entry, ok := view.Entry("base")
			if !ok || entry.Meta["role"] != "base" {
				return nil, nil
			}
			if got := len(view.SectionEntries(aicontext.SectionUser)); got != 1 {
				t.Fatalf("expected source to see later user entry, got %d", got)
			}
			return []aicontext.Part{aicontext.NewPart("conditional-context", "visible from source")}, nil
		})).
		User("request", "answer this").
		BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	if !strings.Contains(prompt.Context, "visible from source") {
		t.Fatalf("expected conditional source output: %q", prompt.Context)
	}
}

func TestPromptBuilderUsesCustomRenderer(t *testing.T) {
	t.Parallel()

	renderer := sectionNameRenderer{}
	prompt, err := aicontext.NewPromptBuilder().
		Renderer(renderer).
		System("base", "system").
		Context("ctx", "context").
		User("request", "user").
		BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}

	if prompt.System != "system:base" || prompt.Context != "context:ctx" || prompt.Prompt != "user:request" {
		t.Fatalf("custom renderer was not used: %+v", prompt)
	}
}

func TestPromptBuilderEmitsDebugEvents(t *testing.T) {
	t.Parallel()

	var events []gai.DebugEvent
	debug := gai.DebugSinkFunc(func(ctx stdcontext.Context, event gai.DebugEvent) {
		events = append(events, event)
	})

	_, err := aicontext.NewPromptBuilder().
		Debug(debug).
		System("base", "system").
		User("request", "user").
		BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}

	if len(events) == 0 {
		t.Fatal("expected debug events")
	}
	if events[0].Name != "prompt_build_started" {
		t.Fatalf("expected first debug event to start build, got %q", events[0].Name)
	}
	for _, event := range events {
		if _, ok := event.Fields["emitted_parts"]; ok {
			t.Fatalf("non-sensitive debug sink should not receive emitted part text: %+v", event)
		}
	}
}

func TestPromptBuilderDropsOptionalSourceOverBudget(t *testing.T) {
	t.Parallel()

	builder := aicontext.NewPromptBuilder().
		Budget(aicontext.PromptBudget{
			Tokenizer:           whitespaceTokenizer{},
			ContextWindowTokens: 14,
		}).
		System("system", "system", aicontext.Required()).
		Source(aicontext.SectionContext, "optional", aicontext.SourceFunc(func(ctx stdcontext.Context, view aicontext.PromptView, budget aicontext.SourceBudget) ([]aicontext.Part, error) {
			return []aicontext.Part{aicontext.NewPart("optional-part", "optional content that does not fit")}, nil
		}), aicontext.Optional()).
		User("request", "question", aicontext.Required())

	prompt, err := builder.BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	if strings.Contains(prompt.Context, "optional content") {
		t.Fatalf("expected optional context to be dropped: %q", prompt.Context)
	}
	trace := builder.LastTrace()
	if got := traceEntryStatus(t, trace, "optional"); got != "dropped" {
		t.Fatalf("expected optional source to be dropped, got %q", got)
	}
}

func TestPromptBuilderFailsRequiredOverBudget(t *testing.T) {
	t.Parallel()

	_, err := aicontext.NewPromptBuilder().
		Budget(aicontext.PromptBudget{
			Tokenizer:           whitespaceTokenizer{},
			ContextWindowTokens: 5,
		}).
		System("system", "system prompt", aicontext.Required()).
		User("request", "question", aicontext.Required()).
		BuildPrompt(stdcontext.Background(), emptyConversation{})
	if !errors.Is(err, aicontext.ErrPromptBudget) {
		t.Fatalf("expected ErrPromptBudget, got %v", err)
	}
	if !strings.Contains(err.Error(), `prompt with "`) || !strings.Contains(err.Error(), "would use") {
		t.Fatalf("expected prompt-wide budget error wording, got %v", err)
	}
}

func TestPromptBuilderTraceSplitsEntryAndPromptTokens(t *testing.T) {
	t.Parallel()

	builder := aicontext.NewPromptBuilder().
		Budget(aicontext.PromptBudget{
			Tokenizer:           whitespaceTokenizer{},
			ContextWindowTokens: 100,
		}).
		System("system", "system", aicontext.Required()).
		User("request", "question", aicontext.Required())
	_, err := builder.BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}

	trace := builder.LastTrace()
	request := traceEntry(t, trace, "request")
	if request.EntryTokens == 0 {
		t.Fatalf("expected entry tokens: %+v", request)
	}
	if request.PromptTokens <= request.EntryTokens {
		t.Fatalf("expected prompt tokens to include prior rendered prompt: %+v", request)
	}
	if request.TokenCount != request.EntryTokens {
		t.Fatalf("expected TokenCount compatibility alias to match entry tokens: %+v", request)
	}
}

func TestPromptBuilderPassesSourceCap(t *testing.T) {
	t.Parallel()

	sourceCalled := false
	_, err := aicontext.NewPromptBuilder().
		Budget(aicontext.PromptBudget{
			Tokenizer:           whitespaceTokenizer{},
			ContextWindowTokens: 50,
		}).
		Source(aicontext.SectionContext, "capped", aicontext.SourceFunc(func(ctx stdcontext.Context, view aicontext.PromptView, budget aicontext.SourceBudget) ([]aicontext.Part, error) {
			sourceCalled = true
			if budget.MaxTokens != 3 {
				t.Fatalf("expected source cap of 3, got %d", budget.MaxTokens)
			}
			return []aicontext.Part{aicontext.NewPart("small", "small")}, nil
		}), aicontext.SourceTokenCap(3)).
		User("request", "question", aicontext.Required()).
		BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	if !sourceCalled {
		t.Fatal("expected capped source to be called")
	}
}

func TestPromptBuilderReusesTokenCountForSourceBudget(t *testing.T) {
	t.Parallel()

	tokenizer := &countingTokenizer{}
	_, err := aicontext.NewPromptBuilder().
		Budget(aicontext.PromptBudget{
			Tokenizer:           tokenizer,
			ContextWindowTokens: 100,
		}).
		System("system", "system", aicontext.Required()).
		Source(aicontext.SectionContext, "source", aicontext.SourceFunc(func(ctx stdcontext.Context, view aicontext.PromptView, budget aicontext.SourceBudget) ([]aicontext.Part, error) {
			if tokenizer.CountCalls != 2 {
				t.Fatalf("source budget should reuse current prompt count, got %d token counts before source", tokenizer.CountCalls)
			}
			return []aicontext.Part{aicontext.NewPart("source-part", "source", aicontext.Required())}, nil
		}), aicontext.Required()).
		User("request", "question", aicontext.Required()).
		BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
}

func TestPromptBuilderDropsEarlierOptionalContextForLaterUserPrompt(t *testing.T) {
	t.Parallel()

	builder := aicontext.NewPromptBuilder().
		Budget(aicontext.PromptBudget{
			Tokenizer:           whitespaceTokenizer{},
			ContextWindowTokens: 12,
		}).
		System("system", "system", aicontext.Required()).
		Source(aicontext.SectionContext, "optional", aicontext.SourceFunc(func(ctx stdcontext.Context, view aicontext.PromptView, budget aicontext.SourceBudget) ([]aicontext.Part, error) {
			return []aicontext.Part{aicontext.NewPart("optional-part", "optional")}, nil
		}), aicontext.Optional()).
		User("request", "question", aicontext.Required())
	prompt, err := builder.BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	if strings.Contains(prompt.Context, "optional") {
		t.Fatalf("expected earlier optional context to be dropped for user prompt: %q", prompt.Context)
	}
	if !strings.Contains(prompt.Prompt, "question") {
		t.Fatalf("expected user prompt to remain: %q", prompt.Prompt)
	}
	if got := traceEntryStatus(t, builder.LastTrace(), "optional"); got != "dropped" {
		t.Fatalf("expected earlier optional trace to be dropped, got %q", got)
	}
}

func TestPromptBuilderBudgetsRequiredSourceBeforeEarlierOptionalContext(t *testing.T) {
	t.Parallel()

	requiredSourceCalled := false
	builder := aicontext.NewPromptBuilder().
		Budget(aicontext.PromptBudget{
			Tokenizer:           whitespaceTokenizer{},
			ContextWindowTokens: 19,
		}).
		System("system", "system", aicontext.Required()).
		Source(aicontext.SectionContext, "optional", aicontext.SourceFunc(func(ctx stdcontext.Context, view aicontext.PromptView, budget aicontext.SourceBudget) ([]aicontext.Part, error) {
			return []aicontext.Part{aicontext.NewPart("optional-part", "optional content with many extra words")}, nil
		}), aicontext.Optional()).
		Source(aicontext.SectionContext, "required", aicontext.SourceFunc(func(ctx stdcontext.Context, view aicontext.PromptView, budget aicontext.SourceBudget) ([]aicontext.Part, error) {
			requiredSourceCalled = true
			if budget.MaxTokens == 0 {
				t.Fatal("required source should receive budget before optional context consumes it")
			}
			return []aicontext.Part{aicontext.NewPart("required-part", "required", aicontext.Required())}, nil
		}), aicontext.Required()).
		User("request", "question", aicontext.Required())

	prompt, err := builder.BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	if !requiredSourceCalled {
		t.Fatal("expected required source to be called")
	}
	if !strings.Contains(prompt.Context, "required") {
		t.Fatalf("expected required source in context: %q", prompt.Context)
	}
	if strings.Contains(prompt.Context, "optional content") {
		t.Fatalf("expected optional context to be dropped: %q", prompt.Context)
	}
}

func TestPromptBuilderRendersRequiredPartsBeforeOptionalParts(t *testing.T) {
	t.Parallel()

	prompt, err := aicontext.NewPromptBuilder().
		Context("optional", "optional").
		Context("required", "required", aicontext.Required()).
		User("request", "question", aicontext.Required()).
		BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	assertOrdered(t, prompt.Context, "required", "optional")
}

func TestPromptBuilderDropsOptionalStaticSystemPartOverBudget(t *testing.T) {
	t.Parallel()

	builder := aicontext.NewPromptBuilder().
		Budget(aicontext.PromptBudget{
			Tokenizer:           whitespaceTokenizer{},
			ContextWindowTokens: 8,
		}).
		System("optional-system", "optional system prompt with too many words", aicontext.Optional()).
		User("request", "question", aicontext.Required())

	prompt, err := builder.BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	if strings.Contains(prompt.System, "optional system prompt") {
		t.Fatalf("expected optional system prompt to be dropped: %q", prompt.System)
	}
	if !strings.Contains(prompt.Prompt, "question") {
		t.Fatalf("expected required user prompt to remain: %q", prompt.Prompt)
	}
	if got := traceEntryStatus(t, builder.LastTrace(), "optional-system"); got != "dropped" {
		t.Fatalf("expected optional static system part to be dropped, got %q", got)
	}
}

func TestPromptBuilderSummarizesOptionalStaticUserPartBeforeDropping(t *testing.T) {
	t.Parallel()

	summarizer := fakeSummarizer{summary: "tiny"}
	builder := aicontext.NewPromptBuilder().
		Budget(aicontext.PromptBudget{
			Tokenizer:           whitespaceTokenizer{},
			ContextWindowTokens: 17,
			Summarizer:          summarizer,
		}).
		System("system", "system", aicontext.Required()).
		User("request", "question", aicontext.Required()).
		User("optional-user", "optional user prompt with too many words", aicontext.Optional())

	prompt, err := builder.BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	if strings.Contains(prompt.Prompt, "optional user prompt") {
		t.Fatalf("expected original optional user prompt to be summarized: %q", prompt.Prompt)
	}
	if !strings.Contains(prompt.Prompt, "tiny") {
		t.Fatalf("expected summarized optional user prompt: %q", prompt.Prompt)
	}
	if got := traceEntryStatus(t, builder.LastTrace(), "optional-user"); got != "summarized" {
		t.Fatalf("expected optional static user part to be summarized, got %q", got)
	}
}

func assertContainsAll(t *testing.T, text, name string, values ...string) {
	t.Helper()

	for _, value := range values {
		if !strings.Contains(text, value) {
			t.Fatalf("expected %s to contain %q: %q", name, value, text)
		}
	}
}

func traceEntryStatus(t *testing.T, trace aicontext.BuildTrace, id string) string {
	t.Helper()
	return traceEntry(t, trace, id).Status
}

func traceEntry(t *testing.T, trace aicontext.BuildTrace, id string) aicontext.BuildTraceEntry {
	t.Helper()
	for _, entry := range trace.Entries {
		if entry.ID == id {
			return entry
		}
	}
	t.Fatalf("trace entry %q not found: %+v", id, trace.Entries)
	return aicontext.BuildTraceEntry{}
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

type sectionNameRenderer struct{}

func (sectionNameRenderer) Render(section aicontext.Section, parts []aicontext.Part) string {
	ids := make([]string, 0, len(parts))
	for _, part := range parts {
		ids = append(ids, part.ID)
	}
	return string(section) + ":" + strings.Join(ids, ",")
}

type fakeSummarizer struct {
	summary string
	err     error
}

func (s fakeSummarizer) Summarize(ctx stdcontext.Context, req aicontext.SummaryRequest) (string, error) {
	if s.err != nil {
		return "", s.err
	}
	return s.summary, nil
}

type countingTokenizer struct {
	CountCalls int
}

func (t *countingTokenizer) Tokenize(ctx stdcontext.Context, text string) ([]string, error) {
	return strings.Fields(text), nil
}

func (t *countingTokenizer) CountTokens(ctx stdcontext.Context, text string) (int, error) {
	t.CountCalls++
	tokens, err := t.Tokenize(ctx, text)
	if err != nil {
		return 0, err
	}
	return len(tokens), nil
}
