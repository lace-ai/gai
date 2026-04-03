package loop_test

import (
	"context"
	"errors"
	"testing"

	"agent-backend/gai/ai"
	"agent-backend/gai/loop"
)

func TestLoop(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		iterations []struct {
			res ai.AIResponse
			err error
		}
		wantIterations int
		wantError      bool
		maxIterations  int
	}{
		{
			name: "Single iteration",
			iterations: []struct {
				res ai.AIResponse
				err error
			}{
				{res: ai.AIResponse{Text: "Hello, World!"}, err: nil},
			},
			wantIterations: 1,
			maxIterations:  8,
		},
		{
			name: "single Tool call",
			iterations: []struct {
				res ai.AIResponse
				err error
			}{
				{res: ai.AIResponse{Text: `{"id":"echo","type":"function","arguments":{"text":"test"}}`}, err: nil},
				{res: ai.AIResponse{Text: "How are you?"}, err: nil},
			},
			wantIterations: 2,
			maxIterations:  8,
		},
		{
			name: "Multiple iterations with tool calls",
			iterations: []struct {
				res ai.AIResponse
				err error
			}{
				{res: ai.AIResponse{Text: `{"id":"echo","type":"function","arguments":{"text":"test"}}`}, err: nil},
				{res: ai.AIResponse{Text: `{"id":"echo","type":"function","arguments":{"text":"test"}}`}, err: nil},
				{res: ai.AIResponse{Text: "How are you?"}, err: nil},
			},
			wantIterations: 3,
			maxIterations:  8,
		},
		{
			name: "Exceeding max iterations",
			iterations: []struct {
				res ai.AIResponse
				err error
			}{
				{res: ai.AIResponse{Text: `{"id":"echo","type":"function","arguments":{"text":"test"}}`}, err: nil},
				{res: ai.AIResponse{Text: `{"id":"echo","type":"function","arguments":{"text":"test"}}`}, err: nil},
				{res: ai.AIResponse{Text: `{"id":"echo","type":"function","arguments":{"text":"test"}}`}, err: nil},
				{res: ai.AIResponse{Text: `{"id":"echo","type":"function","arguments":{"text":"test"}}`}, err: nil},
			},
			wantIterations: 2,
			maxIterations:  2,
			wantError:      true,
		},
		{
			name: "Call wrong tool",
			iterations: []struct {
				res ai.AIResponse
				err error
			}{
				{res: ai.AIResponse{Text: `{"id":"nonexistent_tool","type":"function","arguments":{"text":"test"}}`}, err: nil},
			},
			wantIterations: 0,
			maxIterations:  8,
			wantError:      true,
		},
		{
			name: "No tool calls after response",
			iterations: []struct {
				res ai.AIResponse
				err error
			}{
				{res: ai.AIResponse{Text: "Just a normal response."}, err: nil},
				{res: ai.AIResponse{Text: `{"id":"nonexistent_tool","type":"function","arguments":{"text":"test"}}`}, err: nil},
				{res: ai.AIResponse{Text: `{"id":"nonexistent_tool","type":"function","arguments":{"text":"test"}}`}, err: nil},
			},
			wantIterations: 1,
			maxIterations:  8,
		},
		{
			name: "Tool call with error",
			iterations: []struct {
				res ai.AIResponse
				err error
			}{
				{res: ai.AIResponse{Text: `{"id":"echo","type":"function","arguments":{"text":"test"}}`}, err: nil},
				{res: ai.AIResponse{Text: `{"id":"echo","type":"function","arguments":{"text":"test"}}`}, err: errors.New("tool execution failed")},
				{res: ai.AIResponse{Text: `{"id":"echo","type":"function","arguments":{"text":"test"}}`}, err: nil},
			},
			wantIterations: 1,
			maxIterations:  8,
			wantError:      true,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			model := &MockModel{}
			model.responses = tt.iterations
			tools := []loop.Tool{loop.NewEchoTool()}
			l := loop.New(model, tools, "", "")
			l.MaxLoopIterations = tt.maxIterations

			if err := l.Loop(context.Background(), "", func(i []loop.Iteration) string {
				return ""
			}, func(req loop.ToolRequest, res *loop.ToolResponse) error {
				return nil
			}); (err != nil) != tt.wantError {
				t.Fatalf("Loop failed: %v", err)
			}

			if len(l.Iterations) != tt.wantIterations {
				t.Fatalf("Expected %d iteration, got %d", tt.wantIterations, len(l.Iterations))
			}
		})
	}
}
