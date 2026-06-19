package loop_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
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
	mu        sync.Mutex
	requests  []ai.AIRequest
}

type countingPromptBuilder struct {
	count atomic.Int32
}

func (b *countingPromptBuilder) AppendContextSource(ctx context.Context, source gaictx.ContextSource) error {
	return nil
}

func (b *countingPromptBuilder) AppendContextSources(ctx context.Context, sources ...gaictx.ContextSource) error {
	return nil
}

func (b *countingPromptBuilder) AppendSystemInstructions(ctx context.Context, instructions ...gaictx.Part) error {
	return nil
}

func (b *countingPromptBuilder) BuildContext(ctx context.Context) ([]gaictx.Part, error) {
	return nil, nil
}

func (b *countingPromptBuilder) BuildPrompt(ctx context.Context, conv gaictx.Conversation) (string, error) {
	count := b.count.Add(1)
	return fmt.Sprintf("prompt-%d", count), nil
}

func (b *countingPromptBuilder) GetUserPrompt() string {
	return "Initial prompt"
}

func (b *countingPromptBuilder) SetUserPrompt(prompt string) {
}

type stubPromptBuilder struct {
	systemPrompt string
	userPrompt   string
	contextText  string
	buildContext func() string
}

func (b *stubPromptBuilder) AppendContextSource(ctx context.Context, source gaictx.ContextSource) error {
	return nil
}

func (b *stubPromptBuilder) AppendContextSources(ctx context.Context, sources ...gaictx.ContextSource) error {
	return nil
}

func (b *stubPromptBuilder) AppendSystemInstructions(ctx context.Context, instructions ...gaictx.Part) error {
	return nil
}

func (b *stubPromptBuilder) BuildContext(ctx context.Context) ([]gaictx.Part, error) {
	if b.buildContext != nil {
		b.contextText = b.buildContext()
	}
	return nil, nil
}

func (b *stubPromptBuilder) BuildPrompt(ctx context.Context, conv gaictx.Conversation) (string, error) {
	var prompt strings.Builder
	if b.systemPrompt != "" {
		prompt.WriteString(b.systemPrompt)
		prompt.WriteString("\n")
	}
	if b.contextText != "" {
		prompt.WriteString(b.contextText)
		prompt.WriteString("\n")
	}
	if b.userPrompt != "" {
		prompt.WriteString(b.userPrompt)
		prompt.WriteString("\n")
	}
	prompt.WriteString(renderTestMessages(conv.Messages()))
	return prompt.String(), nil
}

func (b *stubPromptBuilder) GetUserPrompt() string {
	return b.userPrompt
}

func (b *stubPromptBuilder) SetUserPrompt(prompt string) {
	b.userPrompt = prompt
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
		m.mu.Lock()
		m.requests = append(m.requests, req)
		m.mu.Unlock()

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

func (m *scriptedStreamModel) Tokenizer() ai.Tokenizer {
	return &mocks.MockTokenizer{}
}

func (m *scriptedStreamModel) Requests() []ai.AIRequest {
	m.mu.Lock()
	defer m.mu.Unlock()

	requests := make([]ai.AIRequest, len(m.requests))
	copy(requests, m.requests)
	return requests
}

func renderTestMessages(messages []gaictx.Message) string {
	var builder strings.Builder
	for i, message := range messages {
		builder.WriteString("<")
		builder.WriteString(string(message.Role))
		builder.WriteString(" key=")
		builder.WriteString(fmt.Sprintf("%d", i))
		builder.WriteString(">\n")
		builder.WriteString(message.Content.String())
		builder.WriteString("\n</")
		builder.WriteString(string(message.Role))
		builder.WriteString(">")
	}
	return builder.String()
}

func testPromptBuilder() gaictx.PromptBuilder {
	return &stubPromptBuilder{
		systemPrompt: "System prompt",
		userPrompt:   "Initial prompt",
	}
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
				{Res: ai.AIResponse{Text: `{"id":"call-2","type":"function","name":"echo","arguments":{"text":"another test"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: "How are you?"}, Err: nil},
			},
			wantIterations: 3,
			maxIterations:  8,
		},
		{
			name: "Exceeding max iterations",
			iterations: []mocks.MockModelResponse{
				{Res: ai.AIResponse{Text: `{"id":"call-1","type":"function","name":"echo","arguments":{"text":"test 1"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"call-2","type":"function","name":"echo","arguments":{"text":"test 2"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"call-3","type":"function","name":"echo","arguments":{"text":"test 3"}}`}, Err: nil},
				{Res: ai.AIResponse{Text: `{"id":"call-4","type":"function","name":"echo","arguments":{"text":"test 4"}}`}, Err: nil},
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
				{Res: ai.AIResponse{Text: `{"id":"call-2","type":"function","name":"echo","arguments":{"text":"second test"}}`}, Err: errors.New("tool execution failed")},
				{Res: ai.AIResponse{Text: `{"id":"call-3","type":"function","name":"echo","arguments":{"text":"third test"}}`}, Err: nil},
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
			l := loop.New(wrapStreamModel{Model: model}, tools, testPromptBuilder(), nil)
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

			l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
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

func TestLoopSuppressesRepeatedCompletedToolCall(t *testing.T) {
	t.Parallel()

	toolCall := ai.Token{
		Type: ai.TokenTypeToolCall,
		ToolCall: &ai.ToolCall{
			ID:   "call-1",
			Type: "function",
			Name: "echo",
			Args: json.RawMessage(`{"text":"repeat"}`),
		},
	}
	formattedToolCall := ai.Token{
		Type: ai.TokenTypeToolCall,
		ToolCall: &ai.ToolCall{
			ID:   "call-2",
			Type: "function",
			Name: "echo",
			Args: json.RawMessage("{\n  \"text\": \"repeat\"\n}"),
		},
	}

	model := &scriptedStreamModel{
		sequences: [][]ai.Token{
			{toolCall},
			{formattedToolCall},
		},
	}

	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.MaxLoopIterations = 3

	tokenCh, _, errCh := l.Loop(context.Background())
	var tokens []ai.Token
	for token := range tokenCh {
		tokens = append(tokens, token)
	}
	for err := range errCh {
		if err != nil {
			t.Fatalf("unexpected loop error: %v", err)
		}
	}

	if len(tokens) != 1 {
		t.Fatalf("expected duplicate tool call to be suppressed, got %d tokens", len(tokens))
	}
	if len(l.Iterations) != 2 {
		t.Fatalf("expected two iterations including suppressed duplicate, got %d", len(l.Iterations))
	}
	if got := len(l.Iterations[1].Parts); got != 0 {
		t.Fatalf("expected suppressed duplicate iteration to have no parts, got %d", got)
	}
}

func TestLoopAppendsIterationMessagesToIncrementalPrompt(t *testing.T) {
	t.Parallel()

	var buildCount atomic.Int32
	promptBuilder := &stubPromptBuilder{
		systemPrompt: "System prompt",
		userPrompt:   "Initial prompt",
		buildContext: func() string {
			count := buildCount.Add(1)
			return fmt.Sprintf("build-%d", count)
		},
	}

	model := &scriptedStreamModel{
		sequences: [][]ai.Token{
			{
				{
					Type: ai.TokenTypeToolCall,
					ToolCall: &ai.ToolCall{
						ID:   "call-1",
						Type: "function",
						Name: "echo",
						Args: json.RawMessage(`{"text":"payload"}`),
					},
				},
			},
			{
				{Type: ai.TokenTypeText, Data: []byte("done")},
			},
		},
	}

	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, promptBuilder, nil)
	l.MaxLoopIterations = 3

	tokenCh, _, errCh := l.Loop(context.Background())
	for range tokenCh {
	}
	for err := range errCh {
		if err != nil {
			t.Fatalf("unexpected loop error: %v", err)
		}
	}

	if got := buildCount.Load(); got != 1 {
		t.Fatalf("expected incremental prompt builder to build sources once, got %d", got)
	}
	requests := model.Requests()
	if len(requests) != 2 {
		t.Fatalf("expected 2 model requests, got %d", len(requests))
	}
	if !strings.Contains(requests[0].Prompt, "System prompt") {
		t.Fatalf("expected system prompt in first request: %q", requests[0].Prompt)
	}
	if !strings.Contains(requests[0].Prompt, "build-1") || !strings.Contains(requests[1].Prompt, "build-1") {
		t.Fatalf("expected dynamic context to be reused: first=%q second=%q", requests[0].Prompt, requests[1].Prompt)
	}
	if !strings.Contains(requests[0].Prompt, "Initial prompt") {
		t.Fatalf("expected user prompt in first request: %q", requests[0].Prompt)
	}
	if strings.Contains(requests[0].Prompt, "payload") {
		t.Fatalf("first request should not contain future tool delta: %q", requests[0].Prompt)
	}
	if !strings.Contains(requests[1].Prompt, "payload") {
		t.Fatalf("second request should include appended tool delta: %q", requests[1].Prompt)
	}
}

func TestLoopFallsBackToBuildPromptEveryIteration(t *testing.T) {
	t.Parallel()

	promptBuilder := &countingPromptBuilder{}
	model := &scriptedStreamModel{
		sequences: [][]ai.Token{
			{
				{
					Type: ai.TokenTypeToolCall,
					ToolCall: &ai.ToolCall{
						ID:   "call-1",
						Type: "function",
						Name: "echo",
						Args: json.RawMessage(`{"text":"payload"}`),
					},
				},
			},
			{
				{Type: ai.TokenTypeText, Data: []byte("done")},
			},
		},
	}

	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, promptBuilder, nil)
	l.MaxLoopIterations = 3

	tokenCh, _, errCh := l.Loop(context.Background())
	for range tokenCh {
	}
	for err := range errCh {
		if err != nil {
			t.Fatalf("unexpected loop error: %v", err)
		}
	}

	if got := promptBuilder.count.Load(); got != 2 {
		t.Fatalf("expected non-incremental prompt builder to run twice, got %d", got)
	}
	requests := model.Requests()
	if len(requests) != 2 {
		t.Fatalf("expected 2 model requests, got %d", len(requests))
	}
	if requests[0].Prompt != "prompt-1" || requests[1].Prompt != "prompt-2" {
		t.Fatalf("expected rebuilt prompts, got first=%q second=%q", requests[0].Prompt, requests[1].Prompt)
	}
}
