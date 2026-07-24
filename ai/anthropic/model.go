package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	antropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"go.opentelemetry.io/otel/attribute"
)

const (
	anthropicTracerName = "github.com/lace-ai/gai/ai/anthropic"
	defaultMaxTokens    = 4096
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

// sdkClient is deliberately created from the provider's fields for each call.
// Those fields are package-level test seams and callers must never inherit SDK
// environment defaults or retries.
func (p *Provider) sdkClient() antropic.Client {
	return antropic.NewClient(
		option.WithoutEnvironmentDefaults(),
		option.WithAPIKey(p.apiKey),
		option.WithBaseURL(p.baseURL),
		option.WithHTTPClient(p.httpClient),
		option.WithMaxRetries(0),
	)
}

func buildMessagesRequest(req ai.AIRequest, model string) (antropic.MessageNewParams, error) {
	if err := req.ResponseFormat.Validate(); err != nil {
		return antropic.MessageNewParams{}, err
	}
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}
	p := antropic.MessageNewParams{
		Model:     antropic.Model(model),
		MaxTokens: int64(maxTokens),
		Messages:  []antropic.MessageParam{antropic.NewUserMessage(antropic.NewTextBlock(req.Prompt))},
	}
	if len(req.Tools) > 0 {
		tools, err := mapTools(req.Tools)
		if err != nil {
			return antropic.MessageNewParams{}, err
		}
		p.Tools = tools
		choice, err := mapToolChoice(req.ToolChoice)
		if err != nil {
			return antropic.MessageNewParams{}, err
		}
		p.ToolChoice = choice
	}
	format, err := mapResponseFormat(req.ResponseFormat)
	if err != nil {
		return antropic.MessageNewParams{}, err
	}
	if format != nil {
		p.OutputConfig = *format
	}
	if req.Reasoning.Effort != "" {
		if !supportsAdaptiveThinking(model) {
			return antropic.MessageNewParams{}, fmt.Errorf("%w: anthropic reasoning effort is unsupported by %s", ai.ErrUnsupportedCapability, model)
		}
		switch req.Reasoning.Effort {
		case ai.ReasoningEffortLow:
			p.OutputConfig.Effort = antropic.OutputConfigEffortLow
		case ai.ReasoningEffortMedium:
			p.OutputConfig.Effort = antropic.OutputConfigEffortMedium
		case ai.ReasoningEffortHigh:
			p.OutputConfig.Effort = antropic.OutputConfigEffortHigh
		default:
			return antropic.MessageNewParams{}, fmt.Errorf("%w: %s", ai.ErrUnsupportedCapability, req.Reasoning.Effort)
		}
	}
	if req.Reasoning.Enabled || req.Reasoning.IncludeThoughts || req.Reasoning.BudgetTokens > 0 || req.Reasoning.Effort != "" {
		if req.Reasoning.BudgetTokens > 0 {
			if req.Reasoning.BudgetTokens < minThinkingTokens || req.Reasoning.BudgetTokens >= maxTokens {
				return antropic.MessageNewParams{}, fmt.Errorf("%w: anthropic thinking budget must be at least %d and less than max_tokens", ai.ErrUnsupportedCapability, minThinkingTokens)
			}
			thinking := antropic.ThinkingConfigParamOfEnabled(int64(req.Reasoning.BudgetTokens))
			if req.Reasoning.IncludeThoughts {
				thinking.OfEnabled.Display = antropic.ThinkingConfigEnabledDisplaySummarized
			} else {
				thinking.OfEnabled.Display = antropic.ThinkingConfigEnabledDisplayOmitted
			}
			p.Thinking = thinking
		} else if supportsAdaptiveThinking(model) {
			display := antropic.ThinkingConfigAdaptiveDisplayOmitted
			if req.Reasoning.IncludeThoughts {
				display = antropic.ThinkingConfigAdaptiveDisplaySummarized
			}
			p.Thinking = antropic.ThinkingConfigParamUnion{OfAdaptive: &antropic.ThinkingConfigAdaptiveParam{Display: display}}
		} else {
			return antropic.MessageNewParams{}, fmt.Errorf("%w: anthropic adaptive thinking is unsupported by %s", ai.ErrUnsupportedCapability, model)
		}
		if p.ToolChoice.OfAny != nil || p.ToolChoice.OfTool != nil {
			return antropic.MessageNewParams{}, fmt.Errorf("%w: anthropic thinking is incompatible with required tool choice", ai.ErrUnsupportedCapability)
		}
	}
	return p, nil
}

func mapTools(defs []ai.ToolDefinition) ([]antropic.ToolUnionParam, error) {
	tools := make([]antropic.ToolUnionParam, 0, len(defs))
	for _, def := range defs {
		if err := def.Validate(); err != nil {
			return nil, err
		}
		var schema map[string]any
		if err := json.Unmarshal(def.Parameters, &schema); err != nil {
			return nil, fmt.Errorf("decode anthropic tool schema: %w", err)
		}
		tool := antropic.ToolUnionParamOfTool(antropic.ToolInputSchemaParam{ExtraFields: schema}, def.Name)
		if def.Description != "" {
			tool.OfTool.Description = param.NewOpt(def.Description)
		}
		tools = append(tools, tool)
	}
	return tools, nil
}

func mapToolChoice(choice ai.ToolChoice) (antropic.ToolChoiceUnionParam, error) {
	names := choice.Names
	switch choice.Mode {
	case "", ai.ToolChoiceAuto:
		if len(names) != 0 {
			return antropic.ToolChoiceUnionParam{}, fmt.Errorf("%w: anthropic auto tool choice cannot restrict names", ai.ErrUnsupportedCapability)
		}
		return antropic.ToolChoiceUnionParam{OfAuto: &antropic.ToolChoiceAutoParam{}}, nil
	case ai.ToolChoiceNone:
		if len(names) != 0 {
			return antropic.ToolChoiceUnionParam{}, fmt.Errorf("%w: anthropic none tool choice cannot name tools", ai.ErrUnsupportedCapability)
		}
		none := antropic.NewToolChoiceNoneParam()
		return antropic.ToolChoiceUnionParam{OfNone: &none}, nil
	case ai.ToolChoiceRequired:
		if len(names) == 0 {
			return antropic.ToolChoiceUnionParam{OfAny: &antropic.ToolChoiceAnyParam{}}, nil
		}
		if len(names) == 1 && strings.TrimSpace(names[0]) != "" {
			return antropic.ToolChoiceParamOfTool(strings.TrimSpace(names[0])), nil
		}
		return antropic.ToolChoiceUnionParam{}, fmt.Errorf("%w: anthropic required tool choice accepts at most one named tool", ai.ErrUnsupportedCapability)
	default:
		return antropic.ToolChoiceUnionParam{}, fmt.Errorf("unsupported anthropic tool choice mode %q", choice.Mode)
	}
}

func mapResponseFormat(format ai.ResponseFormat) (*antropic.OutputConfigParam, error) {
	switch format.Type {
	case "", ai.ResponseFormatText:
		return nil, nil
	case ai.ResponseFormatJSONObject:
		return nil, fmt.Errorf("%w: anthropic JSON object responses require a JSON schema", ai.ErrUnsupportedCapability)
	case ai.ResponseFormatJSONSchema:
		var schema map[string]any
		if err := json.Unmarshal(format.Schema, &schema); err != nil {
			return nil, fmt.Errorf("decode anthropic response schema: %w", err)
		}
		return &antropic.OutputConfigParam{Format: antropic.JSONOutputFormatParam{Schema: schema}}, nil
	default:
		return nil, fmt.Errorf("%w: %s", ai.ErrInvalidResponseFormat, format.Type)
	}
}

func (m *Model) Generate(ctx context.Context, req ai.AIRequest) (response *ai.AIResponse, err error) {
	ctx, span := gai.StartOperationSpan(ctx, anthropicTracerName, "ai.anthropic", "ai.operation", "model.generate", attribute.String("ai.provider", "anthropic"), attribute.String("ai.model", m.name), attribute.Int("ai.max_tokens", req.MaxTokens), attribute.Int("ai.prompt_length", len(req.Prompt)))
	defer func() { gai.EndSpan(span, err) }()
	payload, err := buildMessagesRequest(req, m.name)
	if err != nil {
		return nil, err
	}
	if m.debug != nil && m.debug.IncludeSensitiveData() {
		m.debug.Emit(ctx, gai.DebugEvent{Name: "anthropic_generate_request", Source: "ai:anthropic.Model.Generate", Fields: map[string]any{"payload": payload}})
	}
	client := m.client.sdkClient()
	message, err := client.Messages.New(ctx, payload)
	if err != nil {
		return nil, localError(err)
	}
	text, thinking, calls, err := mapMessageContent(message.Content)
	if err != nil {
		return nil, err
	}
	input := int(message.Usage.InputTokens + message.Usage.CacheCreationInputTokens + message.Usage.CacheReadInputTokens)
	output := int(message.Usage.OutputTokens)
	span.SetAttributes(attribute.Int("ai.input_tokens", input), attribute.Int("ai.output_tokens", output))
	if m.debug != nil {
		fields := map[string]any{"input_tokens": input, "output_tokens": output}
		if m.debug.IncludeSensitiveData() {
			fields["response"] = message.RawJSON()
		}
		m.debug.Emit(ctx, gai.DebugEvent{Name: "anthropic_generate_success", Source: "ai:anthropic.Model.Generate", Fields: fields})
	}
	return &ai.AIResponse{Text: text, Reasoning: thinking, ToolCalls: calls, Raw: json.RawMessage(message.RawJSON()), FinishReason: string(message.StopReason), InputTokens: input, OutputTokens: output, ReasoningTokens: int(message.Usage.OutputTokensDetails.ThinkingTokens)}, nil
}

func mapMessageContent(blocks []antropic.ContentBlockUnion) (string, string, []ai.ToolCall, error) {
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

func localError(err error) error {
	var sdkErr *antropic.Error
	if !errors.As(err, &sdkErr) {
		return err
	}
	message := strings.TrimSpace(sdkErr.RawJSON())
	var payload antropic.ErrorResponse
	if json.Unmarshal([]byte(sdkErr.RawJSON()), &payload) == nil && payload.Error.Message != "" {
		message = payload.Error.Message
	}
	return &Error{StatusCode: sdkErr.StatusCode, Type: string(sdkErr.Type()), Message: message}
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
		payload, err := buildMessagesRequest(req, m.name)
		if err != nil {
			streamErr = err
			emit(ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
			return
		}
		client := m.client.sdkClient()
		stream := client.Messages.NewStreaming(ctx, payload)
		defer stream.Close()
		blocks := map[int64]*streamBlock{}
		for stream.Next() {
			event := stream.Current()
			switch event.Type {
			case "content_block_start":
				blocks[event.Index] = &streamBlock{typ: event.ContentBlock.Type, id: event.ContentBlock.ID, name: event.ContentBlock.Name}
			case "content_block_delta":
				block := blocks[event.Index]
				if block == nil {
					streamErr = fmt.Errorf("anthropic stream delta for unknown block %d", event.Index)
					break
				}
				switch event.Delta.Type {
				case "text_delta":
					if event.Delta.Text != "" && !emit(ai.Token{Type: ai.TokenTypeText, Text: event.Delta.Text, Data: []byte(event.Delta.Text)}) {
						return
					}
				case "thinking_delta":
					if event.Delta.Thinking != "" && !emit(ai.Token{Type: ai.TokenTypeThought, Text: event.Delta.Thinking, Data: []byte(event.Delta.Thinking)}) {
						return
					}
				case "input_json_delta":
					block.input.WriteString(event.Delta.PartialJSON)
				}
			case "content_block_stop":
				block := blocks[event.Index]
				if block == nil {
					streamErr = fmt.Errorf("anthropic stream stop for unknown block %d", event.Index)
					break
				}
				delete(blocks, event.Index)
				if block.typ == "tool_use" {
					call, callErr := streamToolCall(block)
					if callErr != nil {
						streamErr = callErr
						break
					}
					if !emit(ai.Token{Type: ai.TokenTypeToolCall, Data: append([]byte(nil), call.Args...), ToolCall: call}) {
						return
					}
				}
			}
			if streamErr != nil {
				break
			}
		}
		if streamErr == nil && len(blocks) != 0 {
			streamErr = fmt.Errorf("anthropic stream ended with %d open content block(s)", len(blocks))
		}
		if streamErr == nil {
			streamErr = localError(stream.Err())
		}
		if streamErr == nil && len(blocks) != 0 {
			streamErr = errors.New("anthropic stream ended with open content block")
		}
		if streamErr != nil && !errors.Is(streamErr, context.Canceled) {
			emit(ai.Token{Type: ai.TokenTypeErr, Err: streamErr, Text: streamErr.Error()})
		}
	}()
	return ai.DetectToolCallsInStream(ctx, out, m.debug)
}

type streamBlock struct {
	typ, id, name string
	input         strings.Builder
}

func streamToolCall(block *streamBlock) (*ai.ToolCall, error) {
	name := strings.TrimSpace(block.name)
	if name == "" {
		return nil, errors.New("anthropic stream tool use name empty")
	}
	args := json.RawMessage(block.input.String())
	if len(args) == 0 {
		args = json.RawMessage("{}")
	}
	if !json.Valid(args) {
		return nil, errors.New("anthropic stream tool use input is invalid JSON")
	}
	id := strings.TrimSpace(block.id)
	if id == "" {
		id = ai.GenerateToolCallID(name)
	}
	return &ai.ToolCall{ID: id, Type: "function", Name: name, Args: append(json.RawMessage(nil), args...)}, nil
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
func (t *Tokenizer) CountTokens(ctx context.Context, text string) (tokens int, err error) {
	ctx, span := gai.StartOperationSpan(ctx, anthropicTracerName, "ai.anthropic", "ai.operation", "tokenizer.count_tokens", attribute.String("ai.provider", "anthropic"), attribute.String("ai.model", t.modelName), attribute.Int("ai.input_length", len(text)))
	defer func() { span.SetAttributes(attribute.Int("ai.input_tokens", tokens)); gai.EndSpan(span, err) }()
	client := t.client.sdkClient()
	count, err := client.Messages.CountTokens(ctx, antropic.MessageCountTokensParams{Model: antropic.Model(t.modelName), Messages: []antropic.MessageParam{antropic.NewUserMessage(antropic.NewTextBlock(text))}})
	if err != nil {
		return 0, localError(err)
	}
	return int(count.InputTokens), nil
}
