package loop_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/loop"
	"github.com/lace-ai/gai/testutil/mocks"
)

type wrapStreamModel struct {
	ai.Model
}

func (m wrapStreamModel) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	return ai.DetectToolCallsInStream(ctx, m.Model.GenerateStream(ctx, req), nil)
}

type scriptedStreamModel struct {
	sequences [][]ai.Token
	idx       int
}

func (m *scriptedStreamModel) Name() string {
	return "scripted-stream-model"
}

func (m *scriptedStreamModel) Generate(ctx context.Context, req ai.AIRequest) (*ai.AIResponse, error) {
	return &ai.AIResponse{}, nil
}

func (m *scriptedStreamModel) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	out := make(chan ai.Token)

	go func() {
		defer close(out)
		if m.idx >= len(m.sequences) {
			return
		}
		seq := m.sequences[m.idx]
		m.idx++

		for _, tok := range seq {
			select {
			case <-ctx.Done():
				return
			case out <- tok:
			}
		}
	}()

	return out
}

func (m *scriptedStreamModel) Close() error {
	return nil
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
				{Res: ai.AIResponse{Text: `{"id":"call-1","type":"function","name":"echo","arguments":{"text":"test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: "How are you?"}, Err: nil},
			},
			wantIterations: 2,
			maxIterations:  8,
		},
		{
			name: "Multiple iterations with tool calls",
			iterations: []mocks.MockModelResponse{
				{Res: ai.AIResponse{Text: `{"id":"call-1","type":"function","name":"echo","arguments":{"text":"test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"call-2","type":"function","name":"echo","arguments":{"text":"test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: "How are you?"}, Err: nil},
			},
			wantIterations: 3,
			maxIterations:  8,
		},
		{
			name: "Exceeding max iterations",
			iterations: []mocks.MockModelResponse{
				{Res: ai.AIResponse{Text: `{"id":"call-1","type":"function","name":"echo","arguments":{"text":"test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"call-2","type":"function","name":"echo","arguments":{"text":"test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"call-3","type":"function","name":"echo","arguments":{"text":"test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"call-4","type":"function","name":"echo","arguments":{"text":"test"}}`}, Err: nil},
			},
			wantIterations: 2,
			maxIterations:  2,
			wantError:      true,
		},
		{
			name: "Call wrong tool",
			iterations: []mocks.MockModelResponse{
				{Res: ai.AIResponse{Text: `{"id":"call-1","type":"function","name":"nonexistent_tool","arguments":{"text":"test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: "Tool failed, stopping here."}, Err: nil},
			},
			wantIterations: 2,
			maxIterations:  8,
		},
		{
			name: "No tool calls after response",
			iterations: []mocks.MockModelResponse{
				{Res: ai.AIResponse{Text: "Just a normal response."}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"call-1","type":"function","name":"nonexistent_tool","arguments":{"text":"test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"call-2","type":"function","name":"nonexistent_tool","arguments":{"text":"test"}}`}, Err: nil},
			},
			wantIterations: 1,
			maxIterations:  8,
		},
		{
			name: "Tool call with error",
			iterations: []mocks.MockModelResponse{
				{Res: ai.AIResponse{Text: `{"id":"call-1","type":"function","name":"echo","arguments":{"text":"test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"call-2","type":"function","name":"echo","arguments":{"text":"test"}}`}, Err: errors.New("tool execution failed")},
				{Res: ai.AIResponse{Text: `{"id":"call-3","type":"function","name":"echo","arguments":{"text":"test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: "How are you?"}, Err: nil},
			},
			wantIterations: 3,
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

			tokenCh, _, errCh := l.Loop(context.Background())
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

func TestLoopHandlesManyToolCallsInOneIteration(t *testing.T) {
	t.Parallel()

	makeToolCalls := func(t *testing.T, n int, name string) []ai.Token {
		t.Helper()
		calls := make([]ai.Token, 0, n)
		for i := 0; i < n; i++ {
			args, err := json.Marshal(map[string]string{"text": "payload"})
			if err != nil {
				t.Fatalf("marshal args: %v", err)
			}

			calls = append(calls, ai.Token{
				Type: ai.TokenTypeToolCall,
				ToolCall: &ai.ToolCall{
					ID:   fmt.Sprintf("call-%d", i+1),
					Type: "function",
					Name: name,
					Args: args,
				},
			})
		}
		return calls
	}

	tests := []struct {
		name               string
		firstIteration     []ai.Token
		wantFirstParts     int
		wantToolErrors     int
		wantTotalIteration int
	}{
		{
			name:               "Exactly six valid tool calls",
			firstIteration:     makeToolCalls(t, 6, "echo"),
			wantFirstParts:     6,
			wantToolErrors:     0,
			wantTotalIteration: 2,
		},
		{
			name:               "Ten valid tool calls",
			firstIteration:     makeToolCalls(t, 10, "echo"),
			wantFirstParts:     10,
			wantToolErrors:     0,
			wantTotalIteration: 2,
		},
		{
			name:               "Six unknown tool calls produce tool errors",
			firstIteration:     makeToolCalls(t, 6, "unknown_tool"),
			wantFirstParts:     6,
			wantToolErrors:     6,
			wantTotalIteration: 2,
		},
		{
			name: "Mixed text and seven tool calls",
			firstIteration: append(
				[]ai.Token{{Type: ai.TokenTypeText, Data: []byte("prefix")}},
				makeToolCalls(t, 7, "echo")...,
			),
			wantFirstParts:     8,
			wantToolErrors:     0,
			wantTotalIteration: 2,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			model := &scriptedStreamModel{
				sequences: [][]ai.Token{
					tt.firstIteration,
					{
						{Type: ai.TokenTypeText, Data: []byte("done")},
					},
				},
			}

			l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, "Initial prompt", "System prompt", nil, nil)
			l.MaxLoopIterations = 3

			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()

			tokenCh, _, errCh := l.Loop(ctx)
			for range tokenCh {
			}

			for err := range errCh {
				if err != nil {
					t.Fatalf("unexpected loop error: %v", err)
				}
			}

			if len(l.Iterations) != tt.wantTotalIteration {
				t.Fatalf("expected %d iterations, got %d", tt.wantTotalIteration, len(l.Iterations))
			}

			if got := len(l.Iterations[0].Parts); got != tt.wantFirstParts {
				t.Fatalf("expected %d parts in first iteration, got %d", tt.wantFirstParts, got)
			}

			toolErrs := 0
			for i, part := range l.Iterations[0].Parts {
				if part.Type == loop.IterationTypeToolCall {
					if part.ToolResp == nil {
						t.Fatalf("part %d missing tool response", i)
					}
					if part.ToolResp.Err != nil {
						toolErrs++
					}
				}
			}

			if toolErrs != tt.wantToolErrors {
				t.Fatalf("expected %d tool errors, got %d", tt.wantToolErrors, toolErrs)
			}
		})
	}
}
