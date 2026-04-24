package loop_test

import (
	"context"
	"errors"
	"testing"

	"github.com/HecoAI/gai/ai"
	"github.com/HecoAI/gai/loop"
	"github.com/HecoAI/gai/testutil/mocks"
)

type wrapStreamModel struct {
	ai.Model
}

func (m wrapStreamModel) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	return ai.WrapStream(m.Model.GenerateStream(ctx, req))
}

func TestLoop(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		iterations     []mocks.MockModelResponse
		wantIterations int
		wantError      bool
		maxIterations  int
	}{
		{
			name: "Single iteration",
			iterations: []mocks.MockModelResponse{
				{Res: ai.AIResponse{Text: "Hello, World!"}, Err: nil},
			},
			wantIterations: 1,
			maxIterations:  8,
		},
		{
			name: "single Tool call",
			iterations: []mocks.MockModelResponse{
				{Res: ai.AIResponse{Text: `{"id":"echo","name":"function","arguments":{"text":"test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: "How are you?"}, Err: nil},
			},
			wantIterations: 2,
			maxIterations:  8,
		},
		{
			name: "Multiple iterations with tool calls",
			iterations: []mocks.MockModelResponse{
				{Res: ai.AIResponse{Text: `{"id":"echo","name":"function","arguments":{"text":"test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"echo","name":"function","arguments":{"text":"test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: "How are you?"}, Err: nil},
			},
			wantIterations: 3,
			maxIterations:  8,
		},
		{
			name: "Exceeding max iterations",
			iterations: []mocks.MockModelResponse{
				{Res: ai.AIResponse{Text: `{"id":"echo","name":"function","arguments":{"text":"test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"echo","name":"function","arguments":{"text":"test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"echo","name":"function","arguments":{"text":"test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"echo","name":"function","arguments":{"text":"test"}}`}, Err: nil},
			},
			wantIterations: 2,
			maxIterations:  2,
			wantError:      true,
		},
		{
			name: "Call wrong tool",
			iterations: []mocks.MockModelResponse{
				{Res: ai.AIResponse{Text: `{"id":"nonexistent_tool","name":"function","arguments":{"text":"test"}}`}, Err: nil},
			},
			wantIterations: 2,
			maxIterations:  8,
		},
		{
			name: "No tool calls after response",
			iterations: []mocks.MockModelResponse{
				{Res: ai.AIResponse{Text: "Just a normal response."}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"nonexistent_tool","name":"function","arguments":{"text":"test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"nonexistent_tool","name":"function","arguments":{"text":"test"}}`}, Err: nil},
			},
			wantIterations: 1,
			maxIterations:  8,
		},
		{
			name: "Tool call with error",
			iterations: []mocks.MockModelResponse{
				{Res: ai.AIResponse{Text: `{"id":"echo","name":"function","arguments":{"text":"test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"echo","name":"function","arguments":{"text":"test"}}`}, Err: errors.New("tool execution failed")},
				{Res: ai.AIResponse{Text: `{"id":"echo","name":"function","arguments":{"text":"test"}}`}, Err: nil},
			},
			wantIterations: 2,
			maxIterations:  8,
		},
	}

	for _, tt := range tests {
		tt := tt
			t.Run(tt.name, func(t *testing.T) {
				t.Parallel()

				model := &mocks.MockModel{}
				model.Responses = tt.iterations
				tools := []loop.Tool{loop.NewEchoTool()}
				l := loop.New(wrapStreamModel{Model: model}, tools, "Initial prompt", "System prompt", nil, nil)
				l.MaxLoopIterations = tt.maxIterations

				tokenCh, errCh := l.Loop(context.Background())
				for range tokenCh {
				}

				var err error
				for e := range errCh {
					err = e
				}

				if (err != nil) != tt.wantError {
					t.Fatalf("Loop failed: %v", err)
				}

			if len(l.Iterations) != tt.wantIterations {
				t.Fatalf("Expected %d iteration, got %d", tt.wantIterations, len(l.Iterations))
			}
		})
	}
}
