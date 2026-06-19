package context

import (
	"context"
	"errors"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
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

type messageConversation struct {
	messages []Message
}

func (c messageConversation) Messages() []Message {
	return c.messages
}

func TestBuildPromptRendersStructuredConversationContent(t *testing.T) {
	t.Parallel()

	builder := New(Definition{
		SystemInstructions: []Part{NewTextPart("system")},
		UserPrompt:         "find docs",
	})

	prompt, err := builder.BuildPrompt(context.Background(), messageConversation{
		messages: []Message{
			{
				Role:    RoleAssistant,
				Content: NewToolCallContent("search", `{"q":"lace"}`),
			},
			{
				Role:    RoleTool,
				Content: NewToolResultContent("search", "found <docs>", true, "cached"),
			},
		},
	})
	if err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}

	expected := []string{
		`<user>`,
		`find docs`,
		`</user>`,
		`<assistant>`,
		`<tool_call name="search">`,
		`<arguments>`,
		`{&#34;q&#34;:&#34;lace&#34;}`,
		`<tool>`,
		`<tool_result name="search">`,
		`<result>`,
		`found &lt;docs&gt;`,
	}
	for _, fragment := range expected {
		if !strings.Contains(prompt, fragment) {
			t.Fatalf("expected prompt to contain %q:\n%s", fragment, prompt)
		}
	}
	rejected := []string{
		`<message role=`,
		`<user><text>`,
		`assistant: search`,
		`tool: search result`,
		`{&amp;#34;`,
		`Precomputed`,
		`cached`,
	}
	for _, fragment := range rejected {
		if strings.Contains(prompt, fragment) {
			t.Fatalf("expected prompt not to contain %q:\n%s", fragment, prompt)
		}
	}
}

type debugEventSink struct {
	sensitive bool
	events    []gai.DebugEvent
}

func (s *debugEventSink) Emit(ctx context.Context, e gai.DebugEvent) {
	s.events = append(s.events, e)
}

func (s *debugEventSink) IncludeSensitiveData() bool {
	return s.sensitive
}

type failingPart struct{}

func (failingPart) Name() string {
	return "failing"
}

func (failingPart) Tokens(ctx context.Context, tokenizer ai.Tokenizer) (int, error) {
	return 0, errors.New("token count failed")
}

func (failingPart) Render(ctx context.Context) (RenderNode, error) {
	return RenderNode{}, errors.New("render failed")
}

func TestPromptBuilderEmitsExistingEventsWithoutSensitiveFieldsByDefault(t *testing.T) {
	t.Parallel()

	sink := &debugEventSink{}
	source := &testContextSource{name: "docs", text: "context"}
	builder := New(Definition{
		SystemInstructions: []Part{NewTextPart("system prompt")},
		ContextSources:     []ContextSource{source},
		UserPrompt:         "find docs",
		TokenBudget:        10,
		DebugSink:          sink,
	})
	builder.SetTokenizer(debugTestTokenizer{})

	if _, err := builder.BuildContext(context.Background()); err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	if _, err := builder.BuildPrompt(context.Background(), emptyConversation{}); err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}

	var names []string
	for _, event := range sink.events {
		names = append(names, event.Name)
	}
	want := []string{
		"prompt_builder_context_build_started",
		"prompt_builder_source_included",
		"prompt_builder_context_build_finished",
		"prompt_builder_render_finished",
	}
	if !reflect.DeepEqual(names, want) {
		t.Fatalf("unexpected event names: got %v want %v", names, want)
	}

	renderEvent := sink.events[len(sink.events)-1]
	if _, ok := renderEvent.Fields["prompt"]; ok {
		t.Fatalf("expected prompt field to be omitted without sensitive debug")
	}
	if _, ok := renderEvent.Fields["prompt_structure"]; ok {
		t.Fatalf("expected prompt_structure field to be omitted without sensitive debug")
	}
}

func TestPromptBuilderEmitsSensitiveRenderFieldsWhenEnabled(t *testing.T) {
	t.Parallel()

	sink := &debugEventSink{sensitive: true}
	builder := New(Definition{
		SystemInstructions: []Part{NewTextPart(strings.Repeat("system ", 900))},
		UserPrompt:         "find docs",
		DebugSink:          sink,
	})

	if _, err := builder.BuildPrompt(context.Background(), messageConversation{
		messages: []Message{{Role: RoleAssistant, Content: NewTextContent("assistant reply")}},
	}); err != nil {
		t.Fatalf("BuildPrompt failed: %v", err)
	}

	renderEvent := sink.events[len(sink.events)-1]
	if got := renderEvent.Name; got != "prompt_builder_render_finished" {
		t.Fatalf("expected final render event, got %q", got)
	}
	if got := renderEvent.Fields["prompt_render_mode"]; got != "structured" {
		t.Fatalf("expected structured prompt render mode, got %v", got)
	}
	if _, ok := renderEvent.Fields["prompt_head"]; !ok {
		t.Fatal("expected prompt_head field")
	}
	if _, ok := renderEvent.Fields["prompt_tail"]; !ok {
		t.Fatal("expected prompt_tail field")
	}
	if _, ok := renderEvent.Fields["prompt_structure"]; !ok {
		t.Fatal("expected prompt_structure field")
	}
}

func TestPromptBuilderKeepsTokenErrorEvents(t *testing.T) {
	t.Parallel()

	sink := &debugEventSink{sensitive: true}
	builder := New(Definition{
		SystemInstructions: []Part{failingPart{}},
		DebugSink:          sink,
	})
	builder.SetTokenizer(debugTestTokenizer{})

	promptFields := promptDebugFields(context.Background(), []Part{failingPart{}}, strings.Repeat("p", promptDebugFullLimit+1))
	structure, ok := promptFields["prompt_structure"].([]map[string]any)
	if !ok || len(structure) != 1 {
		t.Fatalf("expected prompt structure entry, got %#v", promptFields["prompt_structure"])
	}
	if got := structure[0]["render_error"]; got != "render failed" {
		t.Fatalf("expected render_error field, got %v", got)
	}

	builder.SystemInstructionsTokens(context.Background())

	names := make([]string, 0, len(sink.events))
	for _, event := range sink.events {
		names = append(names, event.Name)
	}
	if !slices.Contains(names, "prompt_builder_token_count_failed") {
		t.Fatalf("expected token count failure event, got %v", names)
	}
}
