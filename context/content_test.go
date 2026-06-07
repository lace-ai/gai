package context_test

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	gaictx "github.com/lace-ai/gai/context"
)

func TestNewContentFromType(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		contentType string
		input       any
		want        gaictx.Content
		wantString  string
	}{
		{
			name:        "Text content",
			contentType: gaictx.ContentTypeText,
			input:       gaictx.TextContent{Text: "hello"},
			want:        gaictx.TextContent{Text: "hello"},
			wantString:  "hello",
		},
		{
			name:        "Tool call content",
			contentType: gaictx.ContentTypeToolCall,
			input:       gaictx.ToolCallContent{ToolName: "search", Args: `{"query":"docs"}`},
			want:        gaictx.ToolCallContent{ToolName: "search", Args: `{"query":"docs"}`},
			wantString:  `search({"query":"docs"})`,
		},
		{
			name:        "Tool result content",
			contentType: gaictx.ContentTypeToolResult,
			input: gaictx.ToolResultContent{
				ToolName:          "search",
				Result:            "found",
				Precomputed:       true,
				PrecomputedResult: "cached",
			},
			want: gaictx.ToolResultContent{
				ToolName:          "search",
				Result:            "found",
				Precomputed:       true,
				PrecomputedResult: "cached",
			},
			wantString: "search result: found",
		},
		{
			name:        "Tool result error content",
			contentType: gaictx.ContentTypeToolResultErr,
			input:       gaictx.ToolResultErrContent{ToolName: "search", Err: "failed"},
			want:        gaictx.ToolResultErrContent{ToolName: "search", Err: "failed"},
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

			got, err := gaictx.NewContentFromType(tt.contentType, data)
			if err != nil {
				t.Fatalf("NewContentFromType failed: %v", err)
			}
			if !reflect.DeepEqual(got, tt.want) {
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
		content gaictx.Content
		want    gaictx.Content
	}{
		{
			name:    "Text content",
			content: gaictx.TextContent{Text: "hello"},
			want:    gaictx.TextContent{Text: "hello"},
		},
		{
			name:    "Tool call content",
			content: gaictx.ToolCallContent{ToolName: "search", Args: `{"query":"docs"}`},
			want:    gaictx.ToolCallContent{ToolName: "search", Args: `{"query":"docs"}`},
		},
		{
			name: "Tool result content",
			content: gaictx.ToolResultContent{
				ToolName:          "search",
				Result:            "found",
				Precomputed:       true,
				PrecomputedResult: "cached",
			},
			want: gaictx.ToolResultContent{
				ToolName:          "search",
				Result:            "found",
				Precomputed:       true,
				PrecomputedResult: "cached",
			},
		},
		{
			name:    "Tool result error content",
			content: gaictx.ToolResultErrContent{ToolName: "search", Err: "failed"},
			want:    gaictx.ToolResultErrContent{ToolName: "search", Err: "failed"},
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

			got, err := gaictx.NewContentFromType(tt.content.Type(), data)
			if err != nil {
				t.Fatalf("NewContentFromType failed: %v", err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("content = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestNewContentFromTypeRejectsInvalidJSON(t *testing.T) {
	t.Parallel()

	_, err := gaictx.NewContentFromType(gaictx.ContentTypeText, []byte(`{`))
	if err == nil {
		t.Fatal("expected invalid JSON error")
	}
	if !errors.Is(err, gaictx.ErrContentUnmarshal) {
		t.Fatalf("expected ErrContentUnmarshal, got %v", err)
	}
}

func TestNewContentFromTypeRejectsUnknownType(t *testing.T) {
	t.Parallel()

	_, err := gaictx.NewContentFromType("unknown", []byte(`{}`))
	if err == nil {
		t.Fatal("expected unknown content type error")
	}
	if !errors.Is(err, gaictx.ErrUnknownContentType) {
		t.Fatalf("expected ErrUnknownContentType, got %v", err)
	}
	if !strings.Contains(err.Error(), "unknown content type: unknown") {
		t.Fatalf("unexpected error: %v", err)
	}
}
