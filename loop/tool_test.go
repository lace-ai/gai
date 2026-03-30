package loop

import (
	"errors"
	"testing"
)

type testTool struct {
	name string
	resp string
	err  error
}

func (t *testTool) Name() string        { return t.name }
func (t *testTool) Description() string { return "test tool" }
func (t *testTool) Params() string      { return `{"type":"object"}` }
func (t *testTool) Function(_ *ToolRequest) (*ToolResponse, error) {
	if t.err != nil {
		return nil, t.err
	}
	return &ToolResponse{Text: t.resp}, nil
}

func TestDetectToolCallMalformedJSON(t *testing.T) {
	req, ok := detectToolCall(`{"id":"echo","type":"function","arguments":`)
	if !ok {
		t.Fatalf("expected tool-call detection for json-like payload")
	}
	if req != nil {
		t.Fatalf("expected nil request on malformed payload, got %+v", req)
	}
}

func TestDetectToolCallNonJSON(t *testing.T) {
	req, ok := detectToolCall("hello world")
	if ok || req != nil {
		t.Fatalf("expected no tool call for prose input")
	}
}

func TestCallToolNilRequest(t *testing.T) {
	_, err := callTool(nil, nil)
	if !errors.Is(err, ErrInvalidToolRequest) {
		t.Fatalf("expected ErrInvalidToolRequest, got %v", err)
	}
}

func TestCallToolSuccess(t *testing.T) {
	req := &ToolRequest{ID: "echo", Type: "function", Args: []byte(`{"text":"x"}`)}
	res, err := callTool(req, []Tool{&testTool{name: "echo", resp: "ok"}})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if res == nil || res.Text != "ok" {
		t.Fatalf("unexpected response: %+v", res)
	}
}

func TestCallToolValidation(t *testing.T) {
	tests := []struct {
		name string
		req  *ToolRequest
		want error
	}{
		{
			name: "missing type",
			req:  &ToolRequest{ID: "echo", Args: []byte(`{"text":"x"}`)},
			want: ErrToolCallType,
		},
		{
			name: "invalid type",
			req:  &ToolRequest{ID: "echo", Type: "tool", Args: []byte(`{"text":"x"}`)},
			want: ErrToolCallType,
		},
		{
			name: "missing id",
			req:  &ToolRequest{Type: "function", Args: []byte(`{"text":"x"}`)},
			want: ErrToolCallID,
		},
		{
			name: "missing args",
			req:  &ToolRequest{ID: "echo", Type: "function"},
			want: ErrToolArgsMissing,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := callTool(tt.req, []Tool{&testTool{name: "echo", resp: "ok"}})
			if !errors.Is(err, tt.want) {
				t.Fatalf("expected %v, got %v", tt.want, err)
			}
		})
	}
}
