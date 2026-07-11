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

type failingToolResponseProcessor struct {
	err error
}

func (p failingToolResponseProcessor) Process(req ai.ToolCall, res *loop.ToolResponse) error {
	return p.err
}

func (b *countingPromptBuilder) PrependContextSource(ctx context.Context, source gaictx.ContextSource) error {
	return nil
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

func (b *countingPromptBuilder) Input() gaictx.PromptInput {
	return gaictx.PromptInput{User: gaictx.NewTextContent("Initial prompt")}
}

func (b *countingPromptBuilder) SetInput(input gaictx.PromptInput) {
}

type stubPromptBuilder struct {
	systemPrompt string
	userPrompt   string
	contextText  string
	buildContext func() string
}

func (b *stubPromptBuilder) PrependContextSource(ctx context.Context, source gaictx.ContextSource) error {
	return nil
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

func (b *stubPromptBuilder) Input() gaictx.PromptInput {
	if b.userPrompt == "" {
		return gaictx.PromptInput{}
	}
	return gaictx.PromptInput{User: gaictx.NewTextContent(b.userPrompt)}
}

func (b *stubPromptBuilder) SetInput(input gaictx.PromptInput) {
	b.userPrompt = ""
	if input.User != nil {
		b.userPrompt = input.User.String()
	}
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

func collectLoopEvents(t *testing.T, l *loop.Loop, ctx context.Context) []loop.Event {
	t.Helper()

	var events []loop.Event
	for event := range l.Run(ctx) {
		events = append(events, event)
	}
	return events
}

func loopError(events []loop.Event) error {
	var err error
	for _, event := range events {
		if event.Type == loop.EventError {
			err = event.Err
		}
	}
	return err
}

func loopEventsOfType(events []loop.Event, eventType loop.EventType) []loop.Event {
	var filtered []loop.Event
	for _, event := range events {
		if event.Type == eventType {
			filtered = append(filtered, event)
		}
	}
	return filtered
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

			err := loopError(collectLoopEvents(t, l, context.Background()))

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

			if err := loopError(collectLoopEvents(t, l, ctx)); err != nil {
				t.Fatalf("unexpected loop error: %v", err)
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

func TestLoopWrapsToolPreprocessErrors(t *testing.T) {
	t.Parallel()

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
		},
	}

	l := loop.New(
		model,
		[]loop.Tool{loop.NewEchoTool()},
		testPromptBuilder(),
		failingToolResponseProcessor{err: errors.New("reject tool response")},
	)

	events := collectLoopEvents(t, l, context.Background())
	err := loopError(events)
	if !errors.Is(err, loop.ErrToolResponseProcess) {
		t.Fatalf("error = %v, want ErrToolResponseProcess", err)
	}
	errorEvents := loopEventsOfType(events, loop.EventError)
	if len(errorEvents) != 1 {
		t.Fatalf("expected 1 error event, got %d", len(errorEvents))
	}
	if errorEvents[0].IterationCount != 1 || errorEvents[0].AttemptID != 1 {
		t.Fatalf("expected attempt metadata on error event, got %#v", errorEvents[0])
	}
	if len(l.Iterations) != 0 {
		t.Fatalf("expected preprocess failure to skip persisted iteration, got %d", len(l.Iterations))
	}
}

func TestLoopRetriesDoNotConsumeIterations(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{
		sequences: [][]ai.Token{
			{{Err: errors.New("temporary 1")}},
			{{Err: errors.New("temporary 2")}},
			{{Err: errors.New("temporary 3")}},
			{{Type: ai.TokenTypeText, Data: []byte("done")}},
		},
	}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1
	l.RetryCount = 3

	events := collectLoopEvents(t, l, context.Background())
	if err := loopError(events); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}

	if len(l.Iterations) != 1 {
		t.Fatalf("expected retry attempts to complete one iteration, got %d", len(l.Iterations))
	}
	if got := len(model.Requests()); got != 4 {
		t.Fatalf("expected 4 model attempts, got %d", got)
	}
	retries := loopEventsOfType(events, loop.EventRetry)
	iterations := loopEventsOfType(events, loop.EventIterationDone)
	if len(retries) != 3 || len(iterations) != 1 {
		t.Fatalf("expected 3 retries and 1 completed iteration, got retries=%d iterations=%d events=%#v", len(retries), len(iterations), events)
	}
	for i := 0; i < 3; i++ {
		event := retries[i]
		if event.IterationCount != 1 {
			t.Fatalf("retry event %d consumed an iteration: %#v", i, event)
		}
		if event.AttemptID != i+1 {
			t.Fatalf("retry event %d expected attempt %d, got %d", i, i+1, event.AttemptID)
		}
		if event.Iteration == nil || event.Iteration.UserMessage == nil {
			t.Fatalf("retry event %d should retain user message: %#v", i, event.Iteration)
		}
	}
	finalEvent := iterations[0]
	if finalEvent.IterationCount != 1 {
		t.Fatalf("expected final iteration count 1, got %d", finalEvent.IterationCount)
	}
	if finalEvent.AttemptID != 4 {
		t.Fatalf("expected final attempt 4, got %d", finalEvent.AttemptID)
	}
	if finalEvent.RetryCount != 3 {
		t.Fatalf("expected final retry count 3, got %d", finalEvent.RetryCount)
	}
	if l.Iterations[0].UserMessage == nil {
		t.Fatal("expected completed first iteration to retain user message")
	}
}

func TestLoopStreamErrorsIncludeAttemptMetadata(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{
		sequences: [][]ai.Token{
			{{Err: errors.New("fatal stream error")}},
		},
	}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1
	l.RetryCount = 0

	events := collectLoopEvents(t, l, context.Background())
	err := loopError(events)
	if !errors.Is(err, loop.ErrMaxRetries) {
		t.Fatalf("error = %v, want ErrMaxRetries", err)
	}
	errorEvents := loopEventsOfType(events, loop.EventError)
	if len(errorEvents) != 1 {
		t.Fatalf("expected 1 error event, got %d", len(errorEvents))
	}
	if errorEvents[0].IterationCount != 1 || errorEvents[0].AttemptID != 1 || errorEvents[0].RetryCount != 0 {
		t.Fatalf("expected attempt metadata on error event, got %#v", errorEvents[0])
	}
	if errorEvents[0].Iteration == nil || errorEvents[0].Iteration.UserMessage == nil {
		t.Fatalf("expected failed attempt snapshot to retain user message, got %#v", errorEvents[0].Iteration)
	}
}

func TestLoopTerminalStreamErrorIncludesPartialAttemptIteration(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{
		sequences: [][]ai.Token{
			{
				{Type: ai.TokenTypeText, Data: []byte("partial")},
				{Err: errors.New("fatal stream error")},
			},
		},
	}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1
	l.RetryCount = 0

	events := collectLoopEvents(t, l, context.Background())
	err := loopError(events)
	if !errors.Is(err, loop.ErrMaxRetries) {
		t.Fatalf("error = %v, want ErrMaxRetries", err)
	}
	errorEvents := loopEventsOfType(events, loop.EventError)
	if len(errorEvents) != 1 {
		t.Fatalf("expected 1 error event, got %d", len(errorEvents))
	}
	errorEvent := errorEvents[0]
	if errorEvent.PartCount != 1 {
		t.Fatalf("expected partial attempt part count 1, got %d", errorEvent.PartCount)
	}
	if errorEvent.Iteration == nil {
		t.Fatal("expected partial attempt iteration on error event")
	}
	if got := errorEvent.Iteration.Parts[0].Response.Text; got != "partial" {
		t.Fatalf("expected partial attempt text, got %q", got)
	}
	if len(l.Iterations) != 0 {
		t.Fatalf("expected terminal stream failure to skip persisted iteration, got %d", len(l.Iterations))
	}
}

func TestLoopRetryStatusMarksPartialTokensDiscardable(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{
		sequences: [][]ai.Token{
			{
				{Type: ai.TokenTypeText, Data: []byte("partial")},
				{Err: errors.New("temporary")},
			},
			{
				{Type: ai.TokenTypeText, Data: []byte("final")},
			},
		},
	}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1
	l.RetryCount = 1

	events := collectLoopEvents(t, l, context.Background())
	if err := loopError(events); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
	}

	tokenEvents := loopEventsOfType(events, loop.EventToken)
	if len(tokenEvents) != 2 {
		t.Fatalf("expected partial and final token to stream, got %d", len(tokenEvents))
	}
	if string(tokenEvents[0].Token.Data) != "partial" || string(tokenEvents[1].Token.Data) != "final" {
		t.Fatalf("unexpected token events: %#v", tokenEvents)
	}
	if tokenEvents[0].AttemptID != 1 || tokenEvents[1].AttemptID != 2 {
		t.Fatalf("expected token attempt IDs 1 and 2, got %#v", tokenEvents)
	}
	retries := loopEventsOfType(events, loop.EventRetry)
	iterations := loopEventsOfType(events, loop.EventIterationDone)
	if len(retries) != 1 || len(iterations) != 1 {
		t.Fatalf("expected retry and final iteration events, got retries=%d iterations=%d", len(retries), len(iterations))
	}
	retryEvent := retries[0]
	if retryEvent.AttemptID != 1 {
		t.Fatalf("expected retry event for attempt 1, got %#v", retryEvent)
	}
	if retryEvent.PartCount != 1 {
		t.Fatalf("expected retry event to report partial attempt part count, got %d", retryEvent.PartCount)
	}
	if got := l.Iterations[0].Parts[0].Response.Text; got != "final" {
		t.Fatalf("expected persisted iteration to use successful attempt only, got %q", got)
	}
}

func TestLoopDoesNotRetryCanceledStream(t *testing.T) {
	t.Parallel()

	model := &scriptedStreamModel{
		sequences: [][]ai.Token{
			{{Err: context.Canceled}},
			{{Type: ai.TokenTypeText, Data: []byte("should not run")}},
		},
	}
	l := loop.New(model, []loop.Tool{loop.NewEchoTool()}, testPromptBuilder(), nil)
	l.MaxLoopIterations = 1
	l.RetryCount = 3

	events := collectLoopEvents(t, l, context.Background())
	err := loopError(events)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if got := len(model.Requests()); got != 1 {
		t.Fatalf("expected no retry after cancellation, got %d model requests", got)
	}
	if retries := loopEventsOfType(events, loop.EventRetry); len(retries) != 0 {
		t.Fatalf("expected no retry events after cancellation, got %#v", retries)
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

	if err := loopError(collectLoopEvents(t, l, context.Background())); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
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
	if len(l.Iterations) != 2 {
		t.Fatalf("expected 2 stored iterations, got %d", len(l.Iterations))
	}
	if l.Iterations[0].UserMessage == nil {
		t.Fatal("expected first stored iteration to retain user message")
	}
	if l.Iterations[1].UserMessage != nil {
		t.Fatalf("expected later stored iterations to omit user message, got %#v", l.Iterations[1].UserMessage)
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

	if err := loopError(collectLoopEvents(t, l, context.Background())); err != nil {
		t.Fatalf("unexpected loop error: %v", err)
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
	for index, request := range requests {
		if len(request.Tools) != 1 {
			t.Fatalf("request %d expected 1 tool definition, got %d", index, len(request.Tools))
		}
		if request.Tools[0].Name != "echo" {
			t.Fatalf("request %d expected echo tool definition, got %#v", index, request.Tools[0])
		}
	}
}

func TestIterationCountsLeadingThoughtTokens(t *testing.T) {
	t.Parallel()

	var iteration loop.Iteration
	iteration.AppendToken(ai.Token{Type: ai.TokenTypeThought, Text: "thinking", TokenUsage: 7})
	iteration.AppendToken(ai.Token{Type: ai.TokenTypeThought, Text: " more", TokenUsage: 3})
	iteration.AppendToken(ai.Token{Type: ai.TokenTypeText, Text: "answer", TokenUsage: 2})

	if len(iteration.Parts) != 1 {
		t.Fatalf("expected one response part, got %d", len(iteration.Parts))
	}
	response := iteration.Parts[0].Response
	if response == nil {
		t.Fatal("expected response part")
	}
	if response.Text != "answer" {
		t.Fatalf("unexpected visible text: %q", response.Text)
	}
	if response.Reasoning != "thinking more" {
		t.Fatalf("unexpected reasoning: %q", response.Reasoning)
	}
	if response.ReasoningTokens != 10 {
		t.Fatalf("expected reasoning tokens to include leading thought, got %d", response.ReasoningTokens)
	}
	if response.OutputTokens != 12 {
		t.Fatalf("unexpected output tokens: %d", response.OutputTokens)
	}
}

func TestIterationDeltaMessagesSkipsThoughtOnlyResponses(t *testing.T) {
	t.Parallel()

	var iteration loop.Iteration
	iteration.AppendToken(ai.Token{Type: ai.TokenTypeThought, Text: "thinking"})

	if messages := iteration.DeltaMessages(); len(messages) != 0 {
		t.Fatalf("expected no messages for thought-only response, got %#v", messages)
	}

	iteration.AppendToken(ai.Token{Type: ai.TokenTypeText, Text: "answer"})

	messages := iteration.DeltaMessages()
	if len(messages) != 1 {
		t.Fatalf("expected one assistant message after visible text, got %#v", messages)
	}
	if messages[0].Role != gaictx.RoleAssistant {
		t.Fatalf("expected assistant role, got %q", messages[0].Role)
	}
	if got := messages[0].Content.String(); got != "answer" {
		t.Fatalf("unexpected assistant content: %q", got)
	}
}
