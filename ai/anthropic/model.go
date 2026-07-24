package anthropic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"go.opentelemetry.io/otel/attribute"
)

const (
	anthropicTracerName = "github.com/lace-ai/gai/ai/anthropic"
	defaultMaxTokens    = 4096
	maxResponseBody     = 1 << 20
	maxSSEEvent         = 1 << 20
	minThinkingTokens   = 1024
)

type Model struct {
	name   string
	client *Provider
	debug  gai.DebugSink
}

var _ ai.Model = (*Model)(nil)

func (m *Model) Name() string { return m.name }
func (m *Model) Close() error { return nil }
func (m *Model) Tokenizer() ai.Tokenizer {
	return &Tokenizer{modelName: m.name, client: m.client, debug: m.debug}
}

type messagesRequest struct {
	Model        string               `json:"model"`
	MaxTokens    int                  `json:"max_tokens"`
	Messages     []messageRequest     `json:"messages"`
	Stream       bool                 `json:"stream,omitempty"`
	Tools        []toolRequest        `json:"tools,omitempty"`
	ToolChoice   *toolChoiceRequest   `json:"tool_choice,omitempty"`
	OutputConfig *outputConfigRequest `json:"output_config,omitempty"`
	Thinking     *thinkingRequest     `json:"thinking,omitempty"`
}
type messageRequest struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}
type toolRequest struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}
type toolChoiceRequest struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
}
type outputConfigRequest struct {
	Format *outputFormatRequest `json:"format,omitempty"`
	Effort string               `json:"effort,omitempty"`
}
type outputFormatRequest struct {
	Type   string          `json:"type"`
	Schema json.RawMessage `json:"schema"`
}
type thinkingRequest struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
	Display      string `json:"display,omitempty"`
}

func buildMessagesRequest(req ai.AIRequest, model string, stream bool) (messagesRequest, error) {
	if err := req.ResponseFormat.Validate(); err != nil {
		return messagesRequest{}, err
	}
	p := messagesRequest{Model: model, MaxTokens: req.MaxTokens, Messages: []messageRequest{{Role: "user", Content: req.Prompt}}, Stream: stream}
	if p.MaxTokens <= 0 {
		p.MaxTokens = defaultMaxTokens
	}
	if len(req.Tools) > 0 {
		tools, err := mapTools(req.Tools)
		if err != nil {
			return messagesRequest{}, err
		}
		p.Tools = tools
		choice, err := mapToolChoice(req.ToolChoice)
		if err != nil {
			return messagesRequest{}, err
		}
		p.ToolChoice = choice
	}
	format, err := mapResponseFormat(req.ResponseFormat)
	if err != nil {
		return messagesRequest{}, err
	}
	p.OutputConfig = format
	if req.Reasoning.Effort != "" {
		if !supportsAdaptiveThinking(model) {
			return messagesRequest{}, fmt.Errorf("%w: anthropic reasoning effort is unsupported by %s", ai.ErrUnsupportedCapability, model)
		}
		switch req.Reasoning.Effort {
		case ai.ReasoningEffortLow, ai.ReasoningEffortMedium, ai.ReasoningEffortHigh:
			if p.OutputConfig == nil {
				p.OutputConfig = &outputConfigRequest{}
			}
			p.OutputConfig.Effort = string(req.Reasoning.Effort)
		default:
			return messagesRequest{}, fmt.Errorf("%w: %s", ai.ErrUnsupportedCapability, req.Reasoning.Effort)
		}
	}
	if req.Reasoning.Enabled || req.Reasoning.IncludeThoughts || req.Reasoning.BudgetTokens > 0 || req.Reasoning.Effort != "" {
		display := "omitted"
		if req.Reasoning.IncludeThoughts {
			display = "summarized"
		}
		if req.Reasoning.BudgetTokens > 0 {
			if req.Reasoning.BudgetTokens < minThinkingTokens || req.Reasoning.BudgetTokens >= p.MaxTokens {
				return messagesRequest{}, fmt.Errorf("%w: anthropic thinking budget must be at least %d and less than max_tokens", ai.ErrUnsupportedCapability, minThinkingTokens)
			}
			p.Thinking = &thinkingRequest{Type: "enabled", BudgetTokens: req.Reasoning.BudgetTokens, Display: display}
		} else if supportsAdaptiveThinking(model) {
			p.Thinking = &thinkingRequest{Type: "adaptive", Display: display}
		} else {
			return messagesRequest{}, fmt.Errorf("%w: anthropic adaptive thinking is unsupported by %s", ai.ErrUnsupportedCapability, model)
		}
		if p.ToolChoice != nil && (p.ToolChoice.Type == "any" || p.ToolChoice.Type == "tool") {
			return messagesRequest{}, fmt.Errorf("%w: anthropic thinking is incompatible with required tool choice", ai.ErrUnsupportedCapability)
		}
	}
	return p, nil
}

func mapTools(defs []ai.ToolDefinition) ([]toolRequest, error) {
	tools := make([]toolRequest, 0, len(defs))
	for _, def := range defs {
		if err := def.Validate(); err != nil {
			return nil, err
		}
		tools = append(tools, toolRequest{Name: def.Name, Description: def.Description, InputSchema: append(json.RawMessage(nil), def.Parameters...)})
	}
	return tools, nil
}
func mapToolChoice(choice ai.ToolChoice) (*toolChoiceRequest, error) {
	names := choice.Names
	switch choice.Mode {
	case "", ai.ToolChoiceAuto:
		if len(names) != 0 {
			return nil, fmt.Errorf("%w: anthropic auto tool choice cannot restrict names", ai.ErrUnsupportedCapability)
		}
		return &toolChoiceRequest{Type: "auto"}, nil
	case ai.ToolChoiceNone:
		if len(names) != 0 {
			return nil, fmt.Errorf("%w: anthropic none tool choice cannot name tools", ai.ErrUnsupportedCapability)
		}
		return &toolChoiceRequest{Type: "none"}, nil
	case ai.ToolChoiceRequired:
		if len(names) == 0 {
			return &toolChoiceRequest{Type: "any"}, nil
		}
		if len(names) == 1 && strings.TrimSpace(names[0]) != "" {
			return &toolChoiceRequest{Type: "tool", Name: strings.TrimSpace(names[0])}, nil
		}
		return nil, fmt.Errorf("%w: anthropic required tool choice accepts at most one named tool", ai.ErrUnsupportedCapability)
	default:
		return nil, fmt.Errorf("unsupported anthropic tool choice mode %q", choice.Mode)
	}
}
func mapResponseFormat(format ai.ResponseFormat) (*outputConfigRequest, error) {
	switch format.Type {
	case "", ai.ResponseFormatText:
		return nil, nil
	case ai.ResponseFormatJSONObject:
		return nil, fmt.Errorf("%w: anthropic JSON object responses require a JSON schema", ai.ErrUnsupportedCapability)
	case ai.ResponseFormatJSONSchema:
		return &outputConfigRequest{Format: &outputFormatRequest{Type: "json_schema", Schema: append(json.RawMessage(nil), format.Schema...)}}, nil
	default:
		return nil, fmt.Errorf("%w: %s", ai.ErrInvalidResponseFormat, format.Type)
	}
}

func (m *Model) Generate(ctx context.Context, req ai.AIRequest) (response *ai.AIResponse, err error) {
	ctx, span := gai.StartOperationSpan(ctx, anthropicTracerName, "ai.anthropic", "ai.operation", "model.generate", attribute.String("ai.provider", "anthropic"), attribute.String("ai.model", m.name), attribute.Int("ai.max_tokens", req.MaxTokens), attribute.Int("ai.prompt_length", len(req.Prompt)))
	defer func() { gai.EndSpan(span, err) }()
	payload, err := buildMessagesRequest(req, m.name, false)
	if err != nil {
		return nil, err
	}
	if m.debug != nil && m.debug.IncludeSensitiveData() {
		m.debug.Emit(ctx, gai.DebugEvent{Name: "anthropic_generate_request", Source: "ai:anthropic.Model.Generate", Fields: map[string]any{"payload": payload}})
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	httpReq, err := m.newRequest(ctx, "/v1/messages", body, false)
	if err != nil {
		return nil, err
	}
	res, err := m.client.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	span.SetAttributes(attribute.Int("http.response.status_code", res.StatusCode))
	raw, err := readBounded(res.Body)
	if err != nil {
		return nil, err
	}
	if res.StatusCode >= http.StatusMultipleChoices {
		return nil, parseError(res.StatusCode, raw)
	}
	var parsed messageResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("decode anthropic message: %w", err)
	}
	text, thinking, calls, err := mapMessageContent(parsed.Content)
	if err != nil {
		return nil, err
	}
	input := parsed.Usage.InputTokens + parsed.Usage.CacheCreationInputTokens + parsed.Usage.CacheReadInputTokens
	span.SetAttributes(attribute.Int("ai.input_tokens", input), attribute.Int("ai.output_tokens", parsed.Usage.OutputTokens))
	if m.debug != nil {
		fields := map[string]any{"input_tokens": input, "output_tokens": parsed.Usage.OutputTokens}
		if m.debug.IncludeSensitiveData() {
			fields["response"] = string(raw)
		}
		m.debug.Emit(ctx, gai.DebugEvent{Name: "anthropic_generate_success", Source: "ai:anthropic.Model.Generate", Fields: fields})
	}
	return &ai.AIResponse{Text: text, Reasoning: thinking, ToolCalls: calls, Raw: append(json.RawMessage(nil), raw...), FinishReason: parsed.StopReason, InputTokens: input, OutputTokens: parsed.Usage.OutputTokens, ReasoningTokens: parsed.Usage.OutputTokensDetails.ThinkingTokens}, nil
}

func (m *Model) newRequest(ctx context.Context, path string, body []byte, stream bool) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.client.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("x-api-key", m.client.apiKey)
	req.Header.Set("anthropic-version", apiVersion)
	req.Header.Set("Content-Type", "application/json")
	if stream {
		req.Header.Set("Accept", "text/event-stream")
	}
	return req, nil
}

type messageResponse struct {
	Content    []contentBlock `json:"content"`
	StopReason string         `json:"stop_reason"`
	Usage      usage          `json:"usage"`
}
type usage struct {
	InputTokens              int `json:"input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	OutputTokensDetails      struct {
		ThinkingTokens int `json:"thinking_tokens"`
	} `json:"output_tokens_details"`
}
type contentBlock struct {
	Type     string          `json:"type"`
	Text     string          `json:"text"`
	Thinking string          `json:"thinking"`
	ID       string          `json:"id"`
	Name     string          `json:"name"`
	Input    json.RawMessage `json:"input"`
}

func mapMessageContent(blocks []contentBlock) (string, string, []ai.ToolCall, error) {
	var text, thinking strings.Builder
	calls := make([]ai.ToolCall, 0)
	for i, block := range blocks {
		switch block.Type {
		case "text":
			text.WriteString(block.Text)
		case "thinking":
			thinking.WriteString(block.Thinking)
		case "tool_use":
			name := strings.TrimSpace(block.Name)
			if name == "" {
				return "", "", nil, fmt.Errorf("content[%d]: tool use name empty", i)
			}
			args := block.Input
			if len(args) == 0 || string(args) == "null" {
				args = json.RawMessage("{}")
			}
			if !json.Valid(args) {
				return "", "", nil, fmt.Errorf("content[%d]: tool use input is invalid JSON", i)
			}
			id := strings.TrimSpace(block.ID)
			if id == "" {
				id = ai.GenerateToolCallID(name)
			}
			calls = append(calls, ai.ToolCall{ID: id, Type: "function", Name: name, Args: append(json.RawMessage(nil), args...)})
		}
	}
	return text.String(), thinking.String(), calls, nil
}

func readBounded(reader io.Reader) ([]byte, error) {
	raw, err := io.ReadAll(io.LimitReader(reader, maxResponseBody+1))
	if err != nil {
		return nil, err
	}
	if len(raw) > maxResponseBody {
		return nil, fmt.Errorf("anthropic response body exceeds %d bytes", maxResponseBody)
	}
	return raw, nil
}
func parseError(status int, raw []byte) error {
	var payload struct {
		Type  string `json:"type"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(raw, &payload) == nil && payload.Error.Message != "" {
		return &Error{StatusCode: status, Type: payload.Error.Type, Message: payload.Error.Message}
	}
	return &Error{StatusCode: status, Type: payload.Type, Message: strings.TrimSpace(string(raw))}
}

func (m *Model) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	out := make(chan ai.Token, 1)
	go func() {
		ctx, span := gai.StartOperationSpan(ctx, anthropicTracerName, "ai.anthropic", "ai.operation", "model.generate_stream", attribute.String("ai.provider", "anthropic"), attribute.String("ai.model", m.name), attribute.Int("ai.max_tokens", req.MaxTokens), attribute.Int("ai.prompt_length", len(req.Prompt)))
		var streamErr error
		defer func() { gai.EndSpan(span, streamErr); close(out) }()
		emit := func(token ai.Token) bool {
			if ai.SendToken(ctx, out, token) {
				return true
			}
			streamErr = ctx.Err()
			return false
		}
		payload, err := buildMessagesRequest(req, m.name, true)
		if err != nil {
			streamErr = err
			emit(ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
			return
		}
		body, err := json.Marshal(payload)
		if err != nil {
			streamErr = err
			emit(ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
			return
		}
		httpReq, err := m.newRequest(ctx, "/v1/messages", body, true)
		if err != nil {
			streamErr = err
			emit(ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
			return
		}
		res, err := m.client.httpClient.Do(httpReq)
		if err != nil {
			streamErr = err
			emit(ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
			return
		}
		defer res.Body.Close()
		span.SetAttributes(attribute.Int("http.response.status_code", res.StatusCode))
		if res.StatusCode >= http.StatusMultipleChoices {
			raw, readErr := readBounded(res.Body)
			if readErr != nil {
				streamErr = readErr
			} else {
				streamErr = parseError(res.StatusCode, raw)
			}
			emit(ai.Token{Type: ai.TokenTypeErr, Err: streamErr, Text: streamErr.Error()})
			return
		}
		if err := consumeSSE(res.Body, emit); err != nil && !errors.Is(err, context.Canceled) {
			streamErr = err
			emit(ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
		}
	}()
	return ai.DetectToolCallsInStream(ctx, out, m.debug)
}

type streamBlock struct {
	typ, id, name string
	input         strings.Builder
}

func consumeSSE(body io.Reader, emit func(ai.Token) bool) error {
	reader := bufio.NewReader(body)
	var data strings.Builder
	blocks := map[int]*streamBlock{}
	messageStopped := false
	flush := func() error {
		if data.Len() == 0 {
			return nil
		}
		raw := []byte(data.String())
		data.Reset()
		var event struct {
			Type         string       `json:"type"`
			Index        int          `json:"index"`
			ContentBlock contentBlock `json:"content_block"`
			Delta        struct {
				Type        string `json:"type"`
				Text        string `json:"text"`
				Thinking    string `json:"thinking"`
				PartialJSON string `json:"partial_json"`
			} `json:"delta"`
			Error struct {
				Type    string `json:"type"`
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal(raw, &event); err != nil {
			return fmt.Errorf("decode anthropic stream event: %w", err)
		}
		switch event.Type {
		case "message_stop":
			messageStopped = true
		case "error":
			return &Error{Type: event.Error.Type, Message: event.Error.Message}
		case "content_block_start":
			blocks[event.Index] = &streamBlock{typ: event.ContentBlock.Type, id: event.ContentBlock.ID, name: event.ContentBlock.Name}
		case "content_block_delta":
			block := blocks[event.Index]
			if block == nil {
				return fmt.Errorf("anthropic stream delta for unknown block %d", event.Index)
			}
			switch event.Delta.Type {
			case "text_delta":
				if event.Delta.Text != "" && !emit(ai.Token{Type: ai.TokenTypeText, Text: event.Delta.Text, Data: []byte(event.Delta.Text)}) {
					return context.Canceled
				}
			case "thinking_delta":
				if event.Delta.Thinking != "" && !emit(ai.Token{Type: ai.TokenTypeThought, Text: event.Delta.Thinking, Data: []byte(event.Delta.Thinking)}) {
					return context.Canceled
				}
			case "input_json_delta":
				block.input.WriteString(event.Delta.PartialJSON)
			}
		case "content_block_stop":
			block := blocks[event.Index]
			if block == nil {
				return fmt.Errorf("anthropic stream stop for unknown block %d", event.Index)
			}
			delete(blocks, event.Index)
			if block.typ == "tool_use" {
				name := strings.TrimSpace(block.name)
				if name == "" {
					return fmt.Errorf("anthropic stream tool use name empty")
				}
				args := json.RawMessage(block.input.String())
				if len(args) == 0 {
					args = json.RawMessage("{}")
				}
				if !json.Valid(args) {
					return fmt.Errorf("anthropic stream tool use input is invalid JSON")
				}
				id := strings.TrimSpace(block.id)
				if id == "" {
					id = ai.GenerateToolCallID(name)
				}
				call := &ai.ToolCall{ID: id, Type: "function", Name: name, Args: append(json.RawMessage(nil), args...)}
				if !emit(ai.Token{Type: ai.TokenTypeToolCall, Data: append([]byte(nil), args...), ToolCall: call}) {
					return context.Canceled
				}
			}
		}
		return nil
	}
	for {
		line, err := readSSELine(reader)
		if err != nil && !errors.Is(err, io.EOF) {
			return fmt.Errorf("read anthropic stream: %w", err)
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			if flushErr := flush(); flushErr != nil {
				return flushErr
			}
		} else if strings.HasPrefix(line, "data:") {
			if data.Len() > 0 {
				data.WriteByte('\n')
			}
			data.WriteString(strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			if data.Len() > maxSSEEvent {
				return fmt.Errorf("anthropic stream event exceeds %d bytes", maxSSEEvent)
			}
		}
		if errors.Is(err, io.EOF) {
			if flushErr := flush(); flushErr != nil {
				return flushErr
			}
			if len(blocks) != 0 {
				return fmt.Errorf("anthropic stream ended with %d open content block(s)", len(blocks))
			}
			if !messageStopped {
				return errors.New("anthropic stream ended before message_stop")
			}
			return nil
		}
	}
}

func readSSELine(reader *bufio.Reader) (string, error) {
	var line strings.Builder
	for {
		fragment, err := reader.ReadSlice('\n')
		if line.Len()+len(fragment) > maxSSEEvent {
			return "", fmt.Errorf("anthropic stream line exceeds %d bytes", maxSSEEvent)
		}
		line.Write(fragment)
		if errors.Is(err, bufio.ErrBufferFull) {
			continue
		}
		return line.String(), err
	}
}

type Tokenizer struct {
	modelName string
	client    *Provider
	debug     gai.DebugSink
}

func (t *Tokenizer) ID() string { return "anthropic." + t.modelName }
func (t *Tokenizer) Tokenize(ctx context.Context, text string) (tokens []string, err error) {
	_, span := gai.StartOperationSpan(ctx, anthropicTracerName, "ai.anthropic", "ai.operation", "tokenizer.tokenize", attribute.String("ai.provider", "anthropic"), attribute.String("ai.model", t.modelName))
	err = ai.ErrTokenizerUnsupported
	defer gai.EndSpan(span, err)
	return nil, err
}

type countTokensRequest struct {
	Model    string           `json:"model"`
	Messages []messageRequest `json:"messages"`
}

func (t *Tokenizer) CountTokens(ctx context.Context, text string) (tokens int, err error) {
	ctx, span := gai.StartOperationSpan(ctx, anthropicTracerName, "ai.anthropic", "ai.operation", "tokenizer.count_tokens", attribute.String("ai.provider", "anthropic"), attribute.String("ai.model", t.modelName), attribute.Int("ai.input_length", len(text)))
	defer func() { span.SetAttributes(attribute.Int("ai.input_tokens", tokens)); gai.EndSpan(span, err) }()
	body, err := json.Marshal(countTokensRequest{Model: t.modelName, Messages: []messageRequest{{Role: "user", Content: text}}})
	if err != nil {
		return 0, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.client.baseURL+"/v1/messages/count_tokens", bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	req.Header.Set("x-api-key", t.client.apiKey)
	req.Header.Set("anthropic-version", apiVersion)
	req.Header.Set("Content-Type", "application/json")
	res, err := t.client.httpClient.Do(req)
	if err != nil {
		return 0, err
	}
	defer res.Body.Close()
	span.SetAttributes(attribute.Int("http.response.status_code", res.StatusCode))
	raw, err := readBounded(res.Body)
	if err != nil {
		return 0, err
	}
	if res.StatusCode >= http.StatusMultipleChoices {
		return 0, parseError(res.StatusCode, raw)
	}
	var parsed struct {
		InputTokens int `json:"input_tokens"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return 0, fmt.Errorf("decode anthropic token count: %w", err)
	}
	return parsed.InputTokens, nil
}
