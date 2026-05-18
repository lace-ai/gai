package context_test

import (
	"encoding/json"
	"strings"
	"testing"

	aicontext "github.com/lace-ai/gai/context"
)

func TestNewContentFromType(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		contentType string
		input       any
		want        aicontext.Content
		wantString  string
	}{
		{
			name:        "Text content",
			contentType: aicontext.ContentTypeText,
			input:       aicontext.TextContent{Text: "hello"},
			want:        aicontext.TextContent{Text: "hello"},
			wantString:  "hello",
		},
		{
			name:        "Tool call content",
			contentType: aicontext.ContentTypeToolCall,
			input:       aicontext.ToolCallContent{ToolName: "search", Args: `{"query":"docs"}`},
			want:        aicontext.ToolCallContent{ToolName: "search", Args: `{"query":"docs"}`},
			wantString:  `search({"query":"docs"})`,
		},
		{
			name:        "Tool result content",
			contentType: aicontext.ContentTypeToolResult,
			input: aicontext.ToolResultContent{
				ToolName:          "search",
				Result:            "found",
				Precomputed:       true,
				PrecomputedResult: "cached",
			},
			want: aicontext.ToolResultContent{
				ToolName:          "search",
				Result:            "found",
				Precomputed:       true,
				PrecomputedResult: "cached",
			},
			wantString: "search result: found",
		},
		{
			name:        "Tool result error content",
			contentType: aicontext.ContentTypeToolResultErr,
			input:       aicontext.ToolResultErrContent{ToolName: "search", Err: "failed"},
			want:        aicontext.ToolResultErrContent{ToolName: "search", Err: "failed"},
			wantString:  "search error: failed",
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			data, err := json.Marshal(tt.input)
			if err != nil {
				t.Fatalf("Marshal failed: %v", err)
			}

			got, err := aicontext.NewContentFromType(tt.contentType, data)
			if err != nil {
				t.Fatalf("NewContentFromType failed: %v", err)
			}
			if got != tt.want {
				t.Fatalf("content = %#v, want %#v", got, tt.want)
			}
			if got.Type() != tt.contentType {
				t.Fatalf("content type = %q, want %q", got.Type(), tt.contentType)
			}
			if got.String() != tt.wantString {
				t.Fatalf("content string = %q, want %q", got.String(), tt.wantString)
			}
		})
	}
}

func TestContentMarshalRoundTrip(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		content aicontext.Content
		want    aicontext.Content
	}{
		{
			name:    "Text content",
			content: aicontext.TextContent{Text: "hello"},
			want:    aicontext.TextContent{Text: "hello"},
		},
		{
			name:    "Tool call content",
			content: aicontext.ToolCallContent{ToolName: "search", Args: `{"query":"docs"}`},
			want:    aicontext.ToolCallContent{ToolName: "search", Args: `{"query":"docs"}`},
		},
		{
			name: "Tool result content",
			content: aicontext.ToolResultContent{
				ToolName:          "search",
				Result:            "found",
				Precomputed:       true,
				PrecomputedResult: "cached",
			},
			want: aicontext.ToolResultContent{
				ToolName:          "search",
				Result:            "found",
				Precomputed:       true,
				PrecomputedResult: "cached",
			},
		},
		{
			name:    "Tool result error content",
			content: aicontext.ToolResultErrContent{ToolName: "search", Err: "failed"},
			want:    aicontext.ToolResultErrContent{ToolName: "search", Err: "failed"},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			data, err := tt.content.Marshal()
			if err != nil {
				t.Fatalf("Marshal failed: %v", err)
			}

			got, err := aicontext.NewContentFromType(tt.content.Type(), data)
			if err != nil {
				t.Fatalf("NewContentFromType failed: %v", err)
			}
			if got != tt.want {
				t.Fatalf("content = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestNewContentFromTypeRejectsInvalidJSON(t *testing.T) {
	t.Parallel()

	_, err := aicontext.NewContentFromType(aicontext.ContentTypeText, []byte(`{`))
	if err == nil {
		t.Fatal("expected invalid JSON error")
	}
}

func TestNewContentFromTypeRejectsUnknownType(t *testing.T) {
	t.Parallel()

	_, err := aicontext.NewContentFromType("unknown", []byte(`{}`))
	if err == nil {
		t.Fatal("expected unknown content type error")
	}
	if !strings.Contains(err.Error(), "unknown content type: unknown") {
		t.Fatalf("unexpected error: %v", err)
	}
}
