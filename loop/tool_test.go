package loop_test

import (
	"encoding/json"
	"reflect"
	"testing"

	"agent-backend/gai/loop"
)

func TestDetectToolCall(t *testing.T) {
	t.Parallel()

	valideReq := loop.ToolRequest{
		ID:   "test_tool",
		Type: "function",
		Args: json.RawMessage(`{"test":1}`),
	}

	valideInput, err := json.Marshal(valideReq)
	if err != nil {
		t.Fatalf("marshal sample request: %v", err)
		return
	}

	tests := []struct {
		name    string
		input   string
		wantOK  bool
		wantReq loop.ToolRequest
	}{
		{
			name:    "valide tool call",
			input:   string(valideInput),
			wantOK:  true,
			wantReq: valideReq,
		},
		{
			name:   "invalide json",
			input:  `{"id": 1, "}`,
			wantOK: false,
		},
		{
			name:   "plain text",
			input:  "something different",
			wantOK: false,
		},
		{
			name:   "json but not toll call",
			input:  `{"foo":"bar"}`,
			wantOK: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := loop.DetectToolCall(string(tt.input))

			if ok != tt.wantOK {
				t.Fatalf("DetectTollCall() returned ok=%v, want %v, got %v", ok, tt.wantOK, got)
			}
			if tt.wantOK != true {
				return
			}

			if got.ID != tt.wantReq.ID {
				t.Errorf("ID=%q, want %q", got.ID, tt.wantReq.ID)
			}
			if got.Type != tt.wantReq.Type {
				t.Errorf("Type=%q, want %q", got.Type, tt.wantReq.Type)
			}
			if !reflect.DeepEqual(got.Args, tt.wantReq.Args) {
				t.Errorf("Args = %#v, want %#v", got.Args, tt.wantReq.Args)
			}
		})
	}
}
