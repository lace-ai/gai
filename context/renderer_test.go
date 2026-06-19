package context_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

type renderTestPart struct {
	name string
	node gaictx.RenderNode
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
