package context_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

type renderTestPart struct {
	name string
	node gaictx.RenderNode
}

type rendererDebugSink struct {
	sensitive bool
	events    []gai.DebugEvent
}

func (s *rendererDebugSink) Emit(_ context.Context, event gai.DebugEvent) {
	s.events = append(s.events, event)
}

func (s *rendererDebugSink) IncludeSensitiveData() bool {
	return s.sensitive
}

func (p renderTestPart) Name() string {
	return p.name
}

func (p renderTestPart) Tokens(ctx context.Context, tokenizer ai.Tokenizer) (int, error) {
	return 0, nil
}

func (p renderTestPart) Render(ctx context.Context) (gaictx.RenderNode, error) {
	return p.node, nil
}

func TestXMLRendererRendersNestedNodesAndEscapesContent(t *testing.T) {
	t.Parallel()

	rendered, err := (gaictx.XMLRenderer{}).Render(context.Background(), []gaictx.Part{
		renderTestPart{
			name: "debug-name",
			node: gaictx.RenderNode{
				Type:   "message",
				Fields: []gaictx.RenderField{{Key: "role", Value: `user&"admin"`}},
				Children: []gaictx.RenderNode{
					{Type: "text", Value: "hello <world> & everyone"},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("Render failed: %v", err)
	}

	want := "<message role=\"user&amp;&#34;admin&#34;\">\n  <text>\n    hello &lt;world&gt; &amp; everyone\n  </text>\n</message>\n"
	if rendered != want {
		t.Fatalf("unexpected render output:\nwant %q\n got %q", want, rendered)
	}
	if strings.Contains(rendered, "debug-name") {
		t.Fatalf("expected part name not to be used as xml tag: %q", rendered)
	}
}

func TestXMLRendererRejectsInvalidNames(t *testing.T) {
	t.Parallel()

	_, err := (gaictx.XMLRenderer{}).Render(context.Background(), []gaictx.Part{
		renderTestPart{
			name: "bad",
			node: gaictx.RenderNode{Type: "1bad"},
		},
	})
	if err == nil {
		t.Fatal("expected invalid tag error")
	}
	if !strings.Contains(err.Error(), `invalid xml tag name: "1bad"`) {
		t.Fatalf("unexpected error: %v", err)
	}

	_, err = (gaictx.XMLRenderer{}).Render(context.Background(), []gaictx.Part{
		renderTestPart{
			name: "bad",
			node: gaictx.RenderNode{
				Type:   "valid",
				Fields: []gaictx.RenderField{{Key: "bad attr", Value: "x"}},
			},
		},
	})
	if err == nil {
		t.Fatal("expected invalid attribute error")
	}
	if !strings.Contains(err.Error(), `invalid xml attribute name: "bad attr"`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestXMLRendererReturnsPartRenderError(t *testing.T) {
	t.Parallel()

	wantErr := errors.New("render failed")
	part := failingRenderPart{err: wantErr}
	_, err := (gaictx.XMLRenderer{}).Render(context.Background(), []gaictx.Part{part})
	if !errors.Is(err, wantErr) {
		t.Fatalf("expected render error %v, got %v", wantErr, err)
	}
}

func TestSimpleRendererRendersInstructionsHistoryAndConversation(t *testing.T) {
	t.Parallel()

	rendered, err := (gaictx.SimpleRenderer{}).Render(context.Background(), []gaictx.Part{
		gaictx.NewSystemPart([]gaictx.Part{
			gaictx.NewTextPart("follow the instructions carefully"),
			renderTestPart{
				name: "system",
				node: gaictx.RenderNode{Type: "text", Value: "be precise <raw> & direct"},
			},
			renderTestPart{
				name: "tool",
				node: gaictx.RenderNode{Type: "text", Value: "always call search(\"x\") if needed"},
			},
		}),
		&historyPartAdapter{
			contents: []gaictx.Message{
				{Role: gaictx.RoleUser, Content: gaictx.NewTextContent("hi <there> & \"quoted\"")},
				{Role: gaictx.RoleAssistant, Content: gaictx.NewToolCallContent("search", `{"q":"lace<&>"}`)},
				{Role: gaictx.RoleTool, Content: gaictx.NewToolResultContent("search", `found <docs> & "notes"`, false, "")},
				{Role: gaictx.RoleAssistant, Content: gaictx.NewTextContent("done")},
			},
		},
		gaictx.NewMessagePart(gaictx.RoleUser, gaictx.NewTextContent("find docs")),
		gaictx.NewMessagePart(gaictx.RoleAssistant, gaictx.NewToolCallContent("search", `{"q":"lace"}`)),
		gaictx.NewMessagePart(gaictx.RoleTool, gaictx.NewToolResultContent("search", "found <docs>", false, "")),
	})
	if err != nil {
		t.Fatalf("Render failed: %v", err)
	}

	want := `<Instructions>

follow the instructions carefully

System:
be precise <raw> & direct

Tool:
always call search("x") if needed

</Instructions>

<history>
user: hi <there> & "quoted"
assistant: {"q":"lace<&>"}
tool res: found <docs> & "notes"
assistant: done
</history>

user: find docs

assistant: {"q":"lace"}

tool res: found <docs>`
	if rendered != want {
		t.Fatalf("unexpected render output:\nwant %q\n got %q", want, rendered)
	}
	if strings.Contains(rendered, "&lt;") || strings.Contains(rendered, "&amp;") || strings.Contains(rendered, "&#34;") {
		t.Fatalf("expected raw characters to be preserved: %q", rendered)
	}
}

func TestRenderersEmitDetailedTruncatedDebugEvents(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		renderer func(*rendererDebugSink) gaictx.Renderer
	}{
		{
			name: "xml",
			renderer: func(sink *rendererDebugSink) gaictx.Renderer {
				return &gaictx.XMLRenderer{DebugSink: sink, DebugPreviewChars: 5}
			},
		},
		{
			name: "simple",
			renderer: func(sink *rendererDebugSink) gaictx.Renderer {
				return &gaictx.SimpleRenderer{DebugSink: sink, DebugPreviewChars: 5}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			sink := &rendererDebugSink{sensitive: true}
			_, err := tt.renderer(sink).Render(context.Background(), []gaictx.Part{
				gaictx.NewMessagePart(gaictx.RoleUser, gaictx.NewTextContent("first long message")),
				gaictx.NewMessagePart(gaictx.RoleAssistant, gaictx.NewTextContent("second long response")),
			})
			if err != nil {
				t.Fatalf("Render failed: %v", err)
			}

			wantNames := []string{
				"renderer_render_started",
				"renderer_part_rendered",
				"renderer_part_rendered",
				"renderer_render_finished",
			}
			if len(sink.events) != len(wantNames) {
				t.Fatalf("unexpected event count: got %d want %d", len(sink.events), len(wantNames))
			}
			for i, want := range wantNames {
				if got := sink.events[i].Name; got != want {
					t.Fatalf("event %d: got %q want %q", i, got, want)
				}
			}

			partEvent := sink.events[1]
			if got := partEvent.Fields["part_index"]; got != 0 {
				t.Fatalf("unexpected part index: %v", got)
			}
			if got := partEvent.Fields["rendered_mode"]; got != "truncated" {
				t.Fatalf("expected truncated part preview, got %v", got)
			}
			if _, ok := partEvent.Fields["rendered_head"]; !ok {
				t.Fatal("expected rendered_head")
			}
			node, ok := partEvent.Fields["node"].(map[string]any)
			if !ok || node["type"] != "user" || node["value_mode"] != "truncated" {
				t.Fatalf("unexpected node structure: %#v", partEvent.Fields["node"])
			}

			finalEvent := sink.events[3]
			if got := finalEvent.Fields["prompt_mode"]; got != "truncated" {
				t.Fatalf("expected truncated prompt preview, got %v", got)
			}
			structure, ok := finalEvent.Fields["structure"].([]map[string]any)
			if !ok || len(structure) != 2 {
				t.Fatalf("unexpected final structure: %#v", finalEvent.Fields["structure"])
			}
		})
	}
}

func TestRendererDebugStructureOmitsContentForNonSensitiveSink(t *testing.T) {
	t.Parallel()

	sink := &rendererDebugSink{}
	_, err := (gaictx.SimpleRenderer{DebugSink: sink}).Render(context.Background(), []gaictx.Part{
		gaictx.NewMessagePart(gaictx.RoleUser, gaictx.NewTextContent("secret prompt content")),
	})
	if err != nil {
		t.Fatalf("Render failed: %v", err)
	}

	partEvent := sink.events[1]
	if _, ok := partEvent.Fields["rendered"]; ok {
		t.Fatal("non-sensitive event contains rendered content")
	}
	node := partEvent.Fields["node"].(map[string]any)
	if _, ok := node["value"]; ok {
		t.Fatal("non-sensitive node contains value content")
	}
	if got := node["value_chars"]; got != len("secret prompt content") {
		t.Fatalf("unexpected value character count: %v", got)
	}

	finalEvent := sink.events[len(sink.events)-1]
	if _, ok := finalEvent.Fields["prompt"]; ok {
		t.Fatal("non-sensitive event contains prompt content")
	}
}

func TestRenderersNotifyRenderResultCallback(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		renderer gaictx.Renderer
	}{
		{name: "xml", renderer: &gaictx.XMLRenderer{}},
		{name: "simple", renderer: &gaictx.SimpleRenderer{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			parts := []gaictx.Part{
				gaictx.NewMessagePart(gaictx.RoleUser, gaictx.NewTextContent("hello")),
			}
			returned := false
			calls := 0
			var callbackPrompt string

			err := tt.renderer.SetRenderResultCallback(context.Background(), func(receivedParts []gaictx.Part, prompt string) {
				if returned {
					t.Fatal("callback ran after Render returned")
				}
				calls++
				callbackPrompt = prompt
				if len(receivedParts) != len(parts) || receivedParts[0].Name() != parts[0].Name() {
					t.Fatalf("unexpected callback parts: %#v", receivedParts)
				}
			})
			if err != nil {
				t.Fatalf("SetRenderResultCallback failed: %v", err)
			}

			prompt, err := tt.renderer.Render(context.Background(), parts)
			returned = true
			if err != nil {
				t.Fatalf("Render failed: %v", err)
			}
			if calls != 1 {
				t.Fatalf("unexpected callback count: %d", calls)
			}
			if callbackPrompt != prompt {
				t.Fatalf("callback prompt differs from returned prompt: got %q want %q", callbackPrompt, prompt)
			}
		})
	}
}

func TestRenderersNotifyRenderResultCallbackForEmptyPrompt(t *testing.T) {
	t.Parallel()

	renderer := &gaictx.SimpleRenderer{}
	called := false
	if err := renderer.SetRenderResultCallback(context.Background(), func(parts []gaictx.Part, prompt string) {
		called = true
		if len(parts) != 0 || prompt != "" {
			t.Fatalf("unexpected empty render result: parts=%d prompt=%q", len(parts), prompt)
		}
	}); err != nil {
		t.Fatalf("SetRenderResultCallback failed: %v", err)
	}

	if _, err := renderer.Render(context.Background(), nil); err != nil {
		t.Fatalf("Render failed: %v", err)
	}
	if !called {
		t.Fatal("expected callback for empty successful render")
	}
}

type failingRenderPart struct {
	err error
}

func (p failingRenderPart) Name() string {
	return "failing"
}

func (p failingRenderPart) Tokens(ctx context.Context, tokenizer ai.Tokenizer) (int, error) {
	return 0, nil
}

func (p failingRenderPart) Render(ctx context.Context) (gaictx.RenderNode, error) {
	return gaictx.RenderNode{}, p.err
}

type historyPartAdapter struct {
	contents []gaictx.Message
}

func (p *historyPartAdapter) Name() string {
	return "history"
}

func (p *historyPartAdapter) Tokens(ctx context.Context, tokenizer ai.Tokenizer) (int, error) {
	return 0, nil
}

func (p *historyPartAdapter) Render(ctx context.Context) (gaictx.RenderNode, error) {
	node := gaictx.RenderNode{Type: "history"}
	for _, message := range p.contents {
		part := gaictx.NewMessagePart(message.Role, message.Content)
		child, err := part.Render(ctx)
		if err != nil {
			return gaictx.RenderNode{}, err
		}
		node.Children = append(node.Children, child)
	}
	return node, nil
}
