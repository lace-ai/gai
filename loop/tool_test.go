package loop_test

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/HecoAI/gai/loop"
)

func TestDetectToolCall(t *testing.T) {
	t.Parallel()

	validReq := loop.ToolRequest{
		ID:   "test_tool",
		Type: "function",
		Args: json.RawMessage(`{"test":1}`),
	}

	validInput, err := json.Marshal(validReq)
	if err != nil {
		t.Fatalf("marshal sample request: %v", err)
	}

	tests := []struct {
		name    string
		input   string
		wantOK  bool
		wantReq loop.ToolRequest
	}{
		{
			name:    "valid tool call",
			input:   string(validInput),
			wantOK:  true,
			wantReq: validReq,
		},
		{
			name:   "invalid json",
			input:  `{"id": 1, "}`,
			wantOK: false,
		},
		{
			name:   "plain text",
			input:  "something different",
			wantOK: false,
		},
		{
			name:   "json but not tool call",
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

func TestDecodeToolArgs(t *testing.T) {
	t.Parallel()

	type SampleArgs struct {
		Foo string `json:"foo"`
		Bar int    `json:"bar"`
	}

	ValidArgs := SampleArgs{
		Foo: "hello",
		Bar: 42,
	}

	jsonArgs, err := json.Marshal(ValidArgs)
	if err != nil {
		t.Fatalf("marshal sample args: %v", err)
	}

	req := loop.ToolRequest{
		ID:   "test_tool",
		Type: "function",
	}

	tests := []struct {
		name     string
		input    json.RawMessage
		wantErr  bool
		wantArgs SampleArgs
	}{
		{
			name:     "valid args",
			input:    jsonArgs,
			wantErr:  false,
			wantArgs: ValidArgs,
		},
		{
			name:    "invalid json",
			input:   json.RawMessage(`{"foo": "hello", "bar": }`),
			wantErr: true,
		},
		{
			name:    "invalid Tool Args",
			input:   json.RawMessage(`{"foo": "hello", "bar": "not an int"}`),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotArgs SampleArgs

			tReq := req
			tReq.Args = tt.input
			err := loop.DecodeToolArgs(&tReq, &gotArgs)

			if (err != nil) != tt.wantErr {
				t.Fatalf("DecodeToolArgs() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}

			if !reflect.DeepEqual(gotArgs, tt.wantArgs) {
				t.Errorf("Decoded args = %#v, want %#v", gotArgs, tt.wantArgs)
			}
		})
	}
}
