package context_test

import (
	stdcontext "context"
	"errors"
	"strings"
	"testing"

	"github.com/lace-ai/gai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/testutil/mocks"
)

type emptyConversation struct{}

func (emptyConversation) Messages() []gaictx.Message {
	return nil
}

func TestPromptBuilderBuildsStructuredPrompt(t *testing.T) {
	t.Parallel()

	sourceCalled := false
	builder := gaictx.NewPromptBuilder().
		System("base", "base system", gaictx.Required(), gaictx.Tokens(12)).
		System("dynamic", "dynamic system", gaictx.Tokens(4)).
		Source(gaictx.SectionContext, "memory", gaictx.SourceFunc(func(ctx stdcontext.Context, view gaictx.PromptView, budget gaictx.SourceBudget) ([]gaictx.Part, error) {
			sourceCalled = true
			if _, ok := view.Entry("base"); !ok {
				t.Fatal("expected source view to expose full configured plan")
			}
			return []gaictx.Part{
				gaictx.NewPart("history", "stored history"),
				gaictx.NewPart("current-loop", "current messages"),
			}, nil
		}), gaictx.Required()).
		User("request", "answer this", gaictx.Required())

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
	if got := len(trace.Parts[gaictx.SectionContext]); got != 2 {
		t.Fatalf("expected two traced context parts, got %d", got)
	}
}

func TestPromptBuilderEscapesPartIDs(t *testing.T) {
	t.Parallel()

	prompt, err := gaictx.NewPromptBuilder().
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

	_, err := gaictx.NewPromptBuilder().
		System("same", "system").
		User("same", "user").
		BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err == nil {
		t.Fatal("expected duplicate ID error")
	}
	if !errors.Is(err, gaictx.ErrPromptEntryID) {
		t.Fatalf("expected ErrPromptEntryID, got %v", err)
	}
}

func TestPromptBuilderRejectsDuplicateEmittedPartIDs(t *testing.T) {
	t.Parallel()

	_, err := gaictx.NewPromptBuilder().
		System("base", "system").
		Source(gaictx.SectionContext, "dup-source", gaictx.SourceFunc(func(ctx stdcontext.Context, view gaictx.PromptView, budget gaictx.SourceBudget) ([]gaictx.Part, error) {
			return []gaictx.Part{gaictx.NewPart("base", "duplicate")}, nil
		}), gaictx.Required()).
		BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err == nil {
		t.Fatal("expected duplicate emitted part ID error")
	}
	if !errors.Is(err, gaictx.ErrPromptEntryID) {
		t.Fatalf("expected ErrPromptEntryID, got %v", err)
	}
}

func TestPromptBuilderSourceFailurePolicy(t *testing.T) {
	t.Parallel()

	sourceErr := errors.New("source unavailable")
	failingSource := gaictx.SourceFunc(func(ctx stdcontext.Context, view gaictx.PromptView, budget gaictx.SourceBudget) ([]gaictx.Part, error) {
		return nil, sourceErr
	})

	builder := gaictx.NewPromptBuilder().
		Context("kept", "kept context").
		Source(gaictx.SectionContext, "optional-rag", failingSource, gaictx.Optional()).
		User("request", "answer this", gaictx.Required())

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

	_, err = gaictx.NewPromptBuilder().
		Source(gaictx.SectionContext, "required-rag", failingSource, gaictx.Required()).
		User("request", "answer this", gaictx.Required()).
		BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err == nil {
		t.Fatal("expected required source error")
	}
	if !errors.Is(err, gaictx.ErrPromptSource) {
		t.Fatalf("expected ErrPromptSource, got %v", err)
	}
}

func TestPromptBuilderSourceCanInspectWholePlan(t *testing.T) {
	t.Parallel()

	prompt, err := gaictx.NewPromptBuilder().
		System("base", "base system", gaictx.Required(), gaictx.Meta("role", "base")).
		Source(gaictx.SectionContext, "conditional", gaictx.SourceFunc(func(ctx stdcontext.Context, view gaictx.PromptView, budget gaictx.SourceBudget) ([]gaictx.Part, error) {
			entry, ok := view.Entry("base")
			if !ok || entry.Meta["role"] != "base" {
				return nil, nil
			}
			if got := len(view.SectionEntries(gaictx.SectionUser)); got != 1 {
				t.Fatalf("expected source to see later user entry, got %d", got)
			}
			return []gaictx.Part{gaictx.NewPart("conditional-context", "visible from source")}, nil
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
	prompt, err := gaictx.NewPromptBuilder().
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

	_, err := gaictx.NewPromptBuilder().
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

	builder := gaictx.NewPromptBuilder().
		Budget(gaictx.PromptBudget{
			Tokenizer:           &mocks.MockTokenizer{},
			ContextWindowTokens: 6,
		}).
		System("system", "system", gaictx.Required()).
		Source(gaictx.SectionContext, "optional", gaictx.SourceFunc(func(ctx stdcontext.Context, view gaictx.PromptView, budget gaictx.SourceBudget) ([]gaictx.Part, error) {
			return []gaictx.Part{gaictx.NewPart("optional-part", "optional content that does not fit", gaictx.Tokens(5))}, nil
		}), gaictx.Optional()).
		User("request", "question", gaictx.Required())

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

func TestPromptBuilderCountsSourcePartsWithoutExplicitTokens(t *testing.T) {
	t.Parallel()

	builder := gaictx.NewPromptBuilder().
		Budget(gaictx.PromptBudget{
			Tokenizer:           &mocks.MockTokenizer{},
			ContextWindowTokens: 4,
		}).
		System("system", "system", gaictx.Required()).
		Source(gaictx.SectionContext, "optional", gaictx.SourceFunc(func(ctx stdcontext.Context, view gaictx.PromptView, budget gaictx.SourceBudget) ([]gaictx.Part, error) {
			return []gaictx.Part{
				gaictx.NewPartGroup("group", []gaictx.Part{
					gaictx.NewPart("child", "source child exceeds budget"),
				}),
			}, nil
		}), gaictx.Optional()).
		User("request", "question", gaictx.Required())

	prompt, err := builder.BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	if strings.Contains(prompt.Context, "source child exceeds budget") {
		t.Fatalf("expected un-tokenized source child to be counted and dropped: %q", prompt.Context)
	}
	optional := traceEntry(t, builder.LastTrace(), "optional")
	if optional.Status != "dropped" {
		t.Fatalf("expected optional source to be dropped, got %+v", optional)
	}
	if optional.EntryTokens != 4 {
		t.Fatalf("expected normalized source child token count, got %+v", optional)
	}
}

func TestPromptBuilderFailsRequiredOverBudget(t *testing.T) {
	t.Parallel()

	_, err := gaictx.NewPromptBuilder().
		Budget(gaictx.PromptBudget{
			Tokenizer:           &mocks.MockTokenizer{},
			ContextWindowTokens: 2,
		}).
		System("system", "system prompt", gaictx.Required()).
		User("request", "question", gaictx.Required()).
		BuildPrompt(stdcontext.Background(), emptyConversation{})
	if !errors.Is(err, gaictx.ErrPromptBudget) {
		t.Fatalf("expected ErrPromptBudget, got %v", err)
	}
	if !strings.Contains(err.Error(), `prompt with "`) || !strings.Contains(err.Error(), "would use") {
		t.Fatalf("expected prompt-wide budget error wording, got %v", err)
	}
}

func TestPromptBuilderTraceSplitsEntryAndPromptTokens(t *testing.T) {
	t.Parallel()

	builder := gaictx.NewPromptBuilder().
		Budget(gaictx.PromptBudget{
			Tokenizer:           &mocks.MockTokenizer{},
			ContextWindowTokens: 100,
		}).
		System("system", "system", gaictx.Required()).
		User("request", "question", gaictx.Required())
	_, err := builder.BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}

	trace := builder.LastTrace()
	request := traceEntry(t, trace, "request")
	if request.EntryTokens == 0 {
		t.Fatalf("expected entry tokens: %+v", request)
	}
	if request.PromptTokens < request.EntryTokens {
		t.Fatalf("expected prompt tokens to include at least the current entry tokens: %+v", request)
	}
}

func TestPromptBuilderPassesSourceCap(t *testing.T) {
	t.Parallel()

	sourceCalled := false
	_, err := gaictx.NewPromptBuilder().
		Budget(gaictx.PromptBudget{
			Tokenizer:           &mocks.MockTokenizer{},
			ContextWindowTokens: 50,
		}).
		Source(gaictx.SectionContext, "capped", gaictx.SourceFunc(func(ctx stdcontext.Context, view gaictx.PromptView, budget gaictx.SourceBudget) ([]gaictx.Part, error) {
			sourceCalled = true
			if budget.MaxTokens != 3 {
				t.Fatalf("expected source cap of 3, got %d", budget.MaxTokens)
			}
			return []gaictx.Part{gaictx.NewPart("small", "small")}, nil
		}), gaictx.SourceTokenCap(3)).
		User("request", "question", gaictx.Required()).
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

	tokenizer := &mocks.MockTokenizer{}
	_, err := gaictx.NewPromptBuilder().
		Budget(gaictx.PromptBudget{
			Tokenizer:           tokenizer,
			ContextWindowTokens: 100,
		}).
		System("system", "system", gaictx.Required()).
		Source(gaictx.SectionContext, "source", gaictx.SourceFunc(func(ctx stdcontext.Context, view gaictx.PromptView, budget gaictx.SourceBudget) ([]gaictx.Part, error) {
			if tokenizer.CountCalls != 1 {
				t.Fatalf("source budget should reuse counted part tokens, got %d token counts before source", tokenizer.CountCalls)
			}
			return []gaictx.Part{gaictx.NewPart("source-part", "source", gaictx.Required(), gaictx.Tokens(1))}, nil
		}), gaictx.Required()).
		User("request", "question", gaictx.Required()).
		BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
}

func TestPromptBuilderDropsEarlierOptionalContextForLaterUserPrompt(t *testing.T) {
	t.Parallel()

	builder := gaictx.NewPromptBuilder().
		Budget(gaictx.PromptBudget{
			Tokenizer:           &mocks.MockTokenizer{},
			ContextWindowTokens: 2,
		}).
		System("system", "system", gaictx.Required()).
		Source(gaictx.SectionContext, "optional", gaictx.SourceFunc(func(ctx stdcontext.Context, view gaictx.PromptView, budget gaictx.SourceBudget) ([]gaictx.Part, error) {
			return []gaictx.Part{gaictx.NewPart("optional-part", "optional", gaictx.Tokens(1))}, nil
		}), gaictx.Optional()).
		User("request", "question", gaictx.Required())
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
	builder := gaictx.NewPromptBuilder().
		Budget(gaictx.PromptBudget{
			Tokenizer:           &mocks.MockTokenizer{},
			ContextWindowTokens: 8,
		}).
		System("system", "system", gaictx.Required()).
		Source(gaictx.SectionContext, "optional", gaictx.SourceFunc(func(ctx stdcontext.Context, view gaictx.PromptView, budget gaictx.SourceBudget) ([]gaictx.Part, error) {
			return []gaictx.Part{gaictx.NewPart("optional-part", "optional content with many extra words", gaictx.Tokens(6))}, nil
		}), gaictx.Optional()).
		Source(gaictx.SectionContext, "required", gaictx.SourceFunc(func(ctx stdcontext.Context, view gaictx.PromptView, budget gaictx.SourceBudget) ([]gaictx.Part, error) {
			requiredSourceCalled = true
			if budget.MaxTokens == 0 {
				t.Fatal("required source should receive budget before optional context consumes it")
			}
			return []gaictx.Part{gaictx.NewPart("required-part", "required", gaictx.Required(), gaictx.Tokens(1))}, nil
		}), gaictx.Required()).
		User("request", "question", gaictx.Required())

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

	prompt, err := gaictx.NewPromptBuilder().
		Context("optional", "optional").
		Context("required", "required", gaictx.Required()).
		User("request", "question", gaictx.Required()).
		BuildPrompt(stdcontext.Background(), emptyConversation{})
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}
	assertOrdered(t, prompt.Context, "required", "optional")
}

func TestPromptBuilderDropsOptionalStaticSystemPartOverBudget(t *testing.T) {
	t.Parallel()

	builder := gaictx.NewPromptBuilder().
		Budget(gaictx.PromptBudget{
			Tokenizer:           &mocks.MockTokenizer{},
			ContextWindowTokens: 7,
		}).
		System("optional-system", "optional system prompt with too many words", gaictx.Optional()).
		User("request", "question", gaictx.Required())

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
	builder := gaictx.NewPromptBuilder().
		Budget(gaictx.PromptBudget{
			Tokenizer:           &mocks.MockTokenizer{},
			ContextWindowTokens: 3,
			Summarizer:          summarizer,
		}).
		System("system", "system", gaictx.Required()).
		User("request", "question", gaictx.Required()).
		User("optional-user", "optional user prompt with too many words", gaictx.Optional())

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

func TestPromptSessionRebuildsSourcesWhenConversationReserveIsExceeded(t *testing.T) {
	t.Parallel()

	buildCount := 0
	builder := gaictx.NewPromptBuilder().
		Budget(gaictx.PromptBudget{
			Tokenizer:                  &mocks.MockTokenizer{},
			ContextWindowTokens:        7,
			ConversationReserveTokens:  1,
			RenderOverheadReserveRatio: 0,
		}).
		System("system", "system", gaictx.Required(), gaictx.Tokens(1)).
		Source(gaictx.SectionContext, "optional", gaictx.SourceFunc(func(ctx stdcontext.Context, view gaictx.PromptView, budget gaictx.SourceBudget) ([]gaictx.Part, error) {
			buildCount++
			return []gaictx.Part{gaictx.NewPart("optional-context", "optional context", gaictx.Tokens(4))}, nil
		}), gaictx.Optional()).
		User("request", "question", gaictx.Required(), gaictx.Tokens(1))

	session, err := builder.StartPrompt(stdcontext.Background())
	if err != nil {
		t.Fatalf("StartPrompt failed: %v", err)
	}
	if !strings.Contains(session.Prompt().Context, "optional context") {
		t.Fatalf("expected optional source before reserve overflow: %q", session.Prompt().Context)
	}

	prompt, err := session.AppendMessages(stdcontext.Background(), []gaictx.Message{
		sessionMessageWithTokens(1, gaictx.RoleAssistant, "tool delta", "mock.tokenizer", 3),
	})
	if err != nil {
		t.Fatalf("AppendMessages failed: %v", err)
	}
	if buildCount != 2 {
		t.Fatalf("expected source rebuild after reserve overflow, got %d builds", buildCount)
	}
	if strings.Contains(prompt.Context, "optional context") {
		t.Fatalf("expected optional source to be dropped after reserve overflow: %q", prompt.Context)
	}
	if !strings.Contains(prompt.Prompt, "tool delta") {
		t.Fatalf("expected appended delta in prompt: %q", prompt.Prompt)
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

func traceEntryStatus(t *testing.T, trace gaictx.BuildTrace, id string) string {
	t.Helper()
	return traceEntry(t, trace, id).Status
}

func traceEntry(t *testing.T, trace gaictx.BuildTrace, id string) gaictx.BuildTraceEntry {
	t.Helper()
	for _, entry := range trace.Entries {
		if entry.ID == id {
			return entry
		}
	}
	t.Fatalf("trace entry %q not found: %+v", id, trace.Entries)
	return gaictx.BuildTraceEntry{}
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

func (sectionNameRenderer) Render(section gaictx.Section, parts []gaictx.Part) string {
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

func (s fakeSummarizer) Summarize(ctx stdcontext.Context, req gaictx.SummaryRequest) (string, error) {
	if s.err != nil {
		return "", s.err
	}
	return s.summary, nil
}
