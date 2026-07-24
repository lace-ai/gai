package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"go.opentelemetry.io/otel/attribute"
	"google.golang.org/genai"
	genaitokenizer "google.golang.org/genai/tokenizer"
)

const geminiTracerName = "github.com/lace-ai/gai/ai/gemini"

type Model struct {
	name   string
	client *Provider
	debug  gai.DebugSink
	mu     sync.Mutex
	api    *genai.Client
}

var _ ai.Model = (*Model)(nil)

func (m *Model) Name() string {
	return m.name
}

func (m *Model) Tokenizer() ai.Tokenizer {
	return &Tokenizer{
		modelName: m.name,
	}
}

func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.api = nil
	return nil
}

func (m *Model) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	out := make(chan ai.Token, 1)
	prompt := req.Prompt

	go func() {
		ctx, span := gai.StartOperationSpan(ctx, geminiTracerName, "ai.gemini", "ai.operation", "model.generate_stream",
			attribute.String("ai.provider", "gemini"),
			attribute.String("ai.model", m.name),
			attribute.Int("ai.max_tokens", req.MaxTokens),
			attribute.Int("ai.prompt_length", len(prompt)),
		)
		var streamErr error
		if m.debug != nil && m.debug.IncludeSensitiveData() {
			m.debug.Emit(ctx, gai.DebugEvent{
				Name:   "gemini_stream_request",
				Source: "ai:gemini.Model.GenerateStream",
				Fields: map[string]any{
					"prompt":     req.Prompt,
					"max_tokens": req.MaxTokens,
				},
			})
		}
		textTokenCount := 0
		thoughtTokenCount := 0
		toolCallCount := 0
		defer func() {
			span.SetAttributes(
				attribute.Int("ai.text_token_count", textTokenCount),
				attribute.Int("ai.thought_token_count", thoughtTokenCount),
				attribute.Int("ai.tool_call_count", toolCallCount),
			)
			gai.EndSpan(span, streamErr)
		}()
		defer close(out)

		client, err := m.getClient(ctx)
		if err != nil {
			streamErr = err
			if m.debug != nil {
				m.debug.Emit(ctx, gai.DebugEvent{
					Name:   "gemini_get_client_failed",
					Source: "ai:gemini.Model.GenerateStream",
					Fields: map[string]any{
						"error": err.Error(),
					},
					Err: err,
				})
			}
			ai.SendToken(ctx, out, ai.Token{Err: err, Type: ai.TokenTypeErr, Text: err.Error()})
			return
		}

		config, err := buildGenerateContentConfig(req)
		if err != nil {
			streamErr = err
			ai.SendToken(ctx, out, ai.Token{Err: err, Type: ai.TokenTypeErr, Text: err.Error()})
			return
		}

		contents, err := nativeContents(req)
		if err != nil {
			streamErr = err
			ai.SendToken(ctx, out, ai.Token{Err: err, Type: ai.TokenTypeErr, Text: err.Error()})
			return
		}

		for resp, err := range client.Models.GenerateContentStream(ctx, m.name, contents, config) {
			if err != nil {
				streamErr = fmt.Errorf("error generating content stream: %w", err)
				if m.debug != nil {
					m.debug.Emit(ctx, gai.DebugEvent{
						Name:   "gemini_stream_generation_failed",
						Source: "ai:gemini.Model.GenerateStream",
						Fields: map[string]any{
							"error": err.Error(),
						},
						Err: err,
					})
				}
				ai.SendToken(ctx, out, ai.Token{Err: streamErr, Type: ai.TokenTypeErr, Text: streamErr.Error()})
				return
			}

			if resp == nil || len(resp.Candidates) == 0 || resp.Candidates[0] == nil || resp.Candidates[0].Content == nil {
				continue
			}

			for _, part := range resp.Candidates[0].Content.Parts {
				if part == nil {
					continue
				}

				switch {
				case part.Text != "":
					token := buildTextToken(part)
					switch token.Type {
					case ai.TokenTypeThought:
						thoughtTokenCount++
					default:
						textTokenCount++
					}
					if !ai.SendToken(ctx, out, token) {
						return
					}
				case part.FunctionCall != nil:
					rawPart, err := json.Marshal(part)
					if err != nil {
						encodeErr := fmt.Errorf("error encoding part: %w", err)
						streamErr = encodeErr
						if m.debug != nil {
							fields := map[string]any{
								"error": err.Error(),
							}
							if m.debug.IncludeSensitiveData() {
								fields["part"] = string(rawPart)
							}
							m.debug.Emit(ctx, gai.DebugEvent{
								Name:   "gemini_stream_part_encoding_failed",
								Source: "ai:gemini.Model.GenerateStream",
								Fields: fields,
								Err:    err,
							})
						}
						ai.SendToken(ctx, out, ai.Token{Err: encodeErr, Type: ai.TokenTypeErr, Text: encodeErr.Error()})
						return
					}
					toolCall, err := mapFunctionCall(part.FunctionCall)
					if err != nil {
						mapErr := fmt.Errorf("error mapping function call: %w", err)
						streamErr = mapErr
						if m.debug != nil {
							fields := map[string]any{
								"error": err.Error(),
							}
							if m.debug.IncludeSensitiveData() {
								fields["function_call_name"] = part.FunctionCall.Name
								fields["function_call_args"] = fmt.Sprintf("%v", part.FunctionCall.Args)
							}
							m.debug.Emit(ctx, gai.DebugEvent{
								Name:   "gemini_stream_function_call_mapping_failed",
								Source: "ai:gemini.Model.GenerateStream",
								Fields: fields,
								Err:    err,
							})
						}
						ai.SendToken(ctx, out, ai.Token{Err: mapErr, Type: ai.TokenTypeErr, Text: mapErr.Error()})
						return
					}
					if m.debug != nil {
						fields := map[string]any{}
						if m.debug.IncludeSensitiveData() {
							fields["tool_call_id"] = toolCall.ID
							fields["tool_call_args"] = string(toolCall.Args)
						}
						m.debug.Emit(ctx, gai.DebugEvent{
							Name:   "gemini_stream_function_call_mapped",
							Source: "ai:gemini.Model.GenerateStream",
							Fields: fields,
						})
					}
					if !ai.SendToken(ctx, out, ai.Token{
						Type:     ai.TokenTypeToolCall,
						Data:     rawPart,
						ToolCall: toolCall,
					}) {
						return
					}
					toolCallCount++
				}
			}
		}
	}()

	return out
}

func (m *Model) Generate(ctx context.Context, req ai.AIRequest) (response *ai.AIResponse, err error) {
	prompt := req.Prompt
	ctx, span := gai.StartOperationSpan(ctx, geminiTracerName, "ai.gemini", "ai.operation", "model.generate",
		attribute.String("ai.provider", "gemini"),
		attribute.String("ai.model", m.name),
		attribute.Int("ai.max_tokens", req.MaxTokens),
		attribute.Int("ai.prompt_length", len(prompt)),
	)
	defer func() { gai.EndSpan(span, err) }()
	if m.debug != nil && m.debug.IncludeSensitiveData() {
		m.debug.Emit(ctx, gai.DebugEvent{
			Name:   "gemini_generate_request",
			Source: "ai:gemini.Model.Generate",
			Fields: map[string]any{
				"prompt":     req.Prompt,
				"max_tokens": req.MaxTokens,
			},
		})
	}

	client, err := m.getClient(ctx)
	if err != nil {
		if m.debug != nil {
			m.debug.Emit(ctx, gai.DebugEvent{
				Name:   "gemini_get_client_failed",
				Source: "ai:gemini.Model.Generate",
				Fields: map[string]any{
					"error": err.Error(),
				},
				Err: err,
			})
		}
		return nil, err
	}

	config, err := buildGenerateContentConfig(req)
	if err != nil {
		return nil, err
	}

	contents, err := nativeContents(req)
	if err != nil {
		return nil, err
	}
	result, err := client.Models.GenerateContent(
		ctx,
		m.name,
		contents,
		config,
	)
	if err != nil {
		return nil, err
	}
	if err != nil {
		if m.debug != nil {
			m.debug.Emit(ctx, gai.DebugEvent{
				Name:   "gemini_generate_content_failed",
				Source: "ai:gemini.Model.Generate",
				Fields: map[string]any{
					"error": err.Error(),
				},
				Err: err,
			})
		}
		return nil, err
	}

	inputTokens := 0
	outputTokens := 0
	if result.UsageMetadata != nil {
		inputTokens = int(result.UsageMetadata.PromptTokenCount)
		outputTokens = int(result.UsageMetadata.CandidatesTokenCount)
	}
	reasoningTokens := 0
	if result.UsageMetadata != nil {
		reasoningTokens = int(result.UsageMetadata.ThoughtsTokenCount)
	}
	span.SetAttributes(
		attribute.Int("ai.input_tokens", inputTokens),
		attribute.Int("ai.output_tokens", outputTokens),
		attribute.Int("ai.reasoning_tokens", reasoningTokens),
	)

	if m.debug != nil {
		fields := map[string]any{
			"input_tokens":  inputTokens,
			"output_tokens": outputTokens,
		}
		if m.debug.IncludeSensitiveData() {
			fields["response_text"] = result.Text()
		}
		m.debug.Emit(ctx, gai.DebugEvent{
			Name:   "gemini_generate_content_success",
			Source: "ai:gemini.Model.Generate",
			Fields: fields,
		})
	}

	text, reasoning, toolCalls, err := mapGenerateContentResponse(result)
	if err != nil {
		return nil, err
	}
	raw, _ := json.Marshal(result)
	return &ai.AIResponse{
		Text:            text,
		Reasoning:       reasoning,
		ToolCalls:       toolCalls,
		Raw:             raw,
		InputTokens:     inputTokens,
		OutputTokens:    outputTokens,
		ReasoningTokens: reasoningTokens,
	}, nil
}

func nativeContents(req ai.AIRequest) ([]*genai.Content, error) {
	if err := req.ValidateMessages(); err != nil {
		return nil, err
	}
	if len(req.Messages) == 0 {
		return genai.Text(req.Prompt), nil
	}
	seen := map[string]struct{}{}
	out := make([]*genai.Content, 0, len(req.Messages))
	for _, m := range req.Messages {
		if m.Role == ai.RequestMessageRoleUser {
			out = append(out, &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: m.Text}}})
			continue
		}
		if m.Role == ai.RequestMessageRoleTool {
			r := m.ToolResult
			response := map[string]any{"output": r.Content}
			if r.IsError {
				response = map[string]any{"error": r.Content}
			}
			out = append(out, &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{Name: r.Name, Response: response}}}})
			delete(seen, r.Name)
			continue
		}
		parts := []*genai.Part{}
		if m.Text != "" {
			parts = append(parts, &genai.Part{Text: m.Text})
		}
		for _, c := range m.ToolCalls {
			if _, ok := seen[c.Name]; ok {
				return nil, fmt.Errorf("gemini native history has duplicate function %q", c.Name)
			}
			seen[c.Name] = struct{}{}
			var args map[string]any
			if err := json.Unmarshal(c.Arguments, &args); err != nil {
				return nil, err
			}
			parts = append(parts, &genai.Part{FunctionCall: &genai.FunctionCall{Name: c.Name, Args: args}})
		}
		out = append(out, &genai.Content{Role: genai.RoleModel, Parts: parts})
	}
	return out, nil
}

func buildGenerateContentConfig(req ai.AIRequest) (*genai.GenerateContentConfig, error) {
	if err := req.ResponseFormat.Validate(); err != nil {
		return nil, err
	}

	var config *genai.GenerateContentConfig
	ensureConfig := func() *genai.GenerateContentConfig {
		if config == nil {
			config = &genai.GenerateContentConfig{}
		}
		return config
	}

	if req.MaxTokens > 0 {
		ensureConfig().MaxOutputTokens = int32(req.MaxTokens)
	}
	if len(req.Tools) > 0 {
		tools, err := mapGenerateContentTools(req.Tools)
		if err != nil {
			return nil, err
		}
		ensureConfig().Tools = tools
		toolConfig, err := mapGenerateContentToolConfig(req.ToolChoice)
		if err != nil {
			return nil, err
		}
		if toolConfig != nil {
			ensureConfig().ToolConfig = toolConfig
		}
	}
	if err := applyGenerateContentResponseFormat(ensureConfig, req.ResponseFormat); err != nil {
		return nil, err
	}
	if thinking := mapGenerateContentThinkingConfig(req.Reasoning); thinking != nil {
		ensureConfig().ThinkingConfig = thinking
	}
	return config, nil
}

func mapGenerateContentTools(definitions []ai.ToolDefinition) ([]*genai.Tool, error) {
	declarations := make([]*genai.FunctionDeclaration, 0, len(definitions))
	for _, definition := range definitions {
		if err := definition.Validate(); err != nil {
			return nil, err
		}
		var schema any
		if err := json.Unmarshal(definition.Parameters, &schema); err != nil {
			return nil, err
		}
		declarations = append(declarations, &genai.FunctionDeclaration{
			Name:                 definition.Name,
			Description:          definition.Description,
			ParametersJsonSchema: schema,
		})
	}
	return []*genai.Tool{{FunctionDeclarations: declarations}}, nil
}

func mapGenerateContentToolConfig(choice ai.ToolChoice) (*genai.ToolConfig, error) {
	if len(choice.Names) > 0 {
		switch choice.Mode {
		case ai.ToolChoiceAuto, "":
			return nil, fmt.Errorf("gemini tool choice %q with specific tool names is unsupported: Gemini SDK cannot enforce allowed tool names in auto mode", ai.ToolChoiceAuto)
		case ai.ToolChoiceNone:
			return nil, fmt.Errorf("gemini tool choice %q with specific tool names is invalid: no tools may be called", ai.ToolChoiceNone)
		}
	}

	var mode genai.FunctionCallingConfigMode
	switch choice.Mode {
	case ai.ToolChoiceNone:
		mode = genai.FunctionCallingConfigModeNone
	case ai.ToolChoiceRequired:
		mode = genai.FunctionCallingConfigModeAny
	case ai.ToolChoiceAuto, "":
		return nil, nil
	default:
		return nil, fmt.Errorf("unsupported gemini tool choice mode %q", choice.Mode)
	}
	return &genai.ToolConfig{
		FunctionCallingConfig: &genai.FunctionCallingConfig{
			Mode:                 mode,
			AllowedFunctionNames: append([]string(nil), choice.Names...),
		},
	}, nil
}

func applyGenerateContentResponseFormat(ensureConfig func() *genai.GenerateContentConfig, format ai.ResponseFormat) error {
	switch format.Type {
	case "", ai.ResponseFormatText:
		return nil
	case ai.ResponseFormatJSONObject:
		ensureConfig().ResponseMIMEType = "application/json"
		return nil
	case ai.ResponseFormatJSONSchema:
		var schema any
		if err := json.Unmarshal(format.Schema, &schema); err != nil {
			return err
		}
		cfg := ensureConfig()
		cfg.ResponseMIMEType = "application/json"
		cfg.ResponseJsonSchema = schema
		return nil
	default:
		return fmt.Errorf("%w: %s", ai.ErrInvalidResponseFormat, format.Type)
	}
}

func mapGenerateContentThinkingConfig(reasoning ai.ReasoningConfig) *genai.ThinkingConfig {
	if !reasoning.Enabled && !reasoning.IncludeThoughts && reasoning.BudgetTokens <= 0 && reasoning.Effort == "" {
		return nil
	}
	config := &genai.ThinkingConfig{
		IncludeThoughts: reasoning.IncludeThoughts,
	}
	if reasoning.BudgetTokens > 0 {
		budget := int32(reasoning.BudgetTokens)
		config.ThinkingBudget = &budget
	}
	switch reasoning.Effort {
	case ai.ReasoningEffortLow:
		config.ThinkingLevel = genai.ThinkingLevelLow
	case ai.ReasoningEffortMedium:
		config.ThinkingLevel = genai.ThinkingLevelMedium
	case ai.ReasoningEffortHigh:
		config.ThinkingLevel = genai.ThinkingLevelHigh
	}
	return config
}

func mapGenerateContentResponse(result *genai.GenerateContentResponse) (string, string, []ai.ToolCall, error) {
	if result == nil || len(result.Candidates) == 0 || result.Candidates[0] == nil || result.Candidates[0].Content == nil {
		return "", "", nil, nil
	}
	var text strings.Builder
	var reasoning strings.Builder
	var toolCalls []ai.ToolCall
	for _, part := range result.Candidates[0].Content.Parts {
		if part == nil {
			continue
		}
		switch {
		case part.Text != "":
			if part.Thought {
				reasoning.WriteString(part.Text)
			} else {
				text.WriteString(part.Text)
			}
		case part.FunctionCall != nil:
			toolCall, err := mapFunctionCall(part.FunctionCall)
			if err != nil {
				return "", "", nil, err
			}
			toolCalls = append(toolCalls, *toolCall)
		}
	}
	return text.String(), reasoning.String(), toolCalls, nil
}

func mapFunctionCall(functionCall *genai.FunctionCall) (*ai.ToolCall, error) {
	toolName := strings.TrimSpace(functionCall.Name)
	if toolName == "" {
		return nil, fmt.Errorf("function call name empty")
	}

	args, err := marshalArgs(functionCall.Args)
	if err != nil {
		return nil, err
	}
	return &ai.ToolCall{
		ID:   ai.GenerateToolCallID(toolName),
		Type: "function",
		Name: toolName,
		Args: args,
	}, nil
}

func buildTextToken(part *genai.Part) ai.Token {
	tokenType := ai.TokenTypeText
	if part.Thought {
		tokenType = ai.TokenTypeThought
	}
	return ai.Token{
		Type: tokenType,
		Data: []byte(part.Text),
		Text: part.Text,
	}
}

func marshalArgs(args map[string]any) (json.RawMessage, error) {
	if args == nil {
		return json.RawMessage("{}"), nil
	}

	raw, err := json.Marshal(args)
	if err != nil {
		return nil, err
	}

	return json.RawMessage(raw), nil
}

func (m *Model) getClient(ctx context.Context) (*genai.Client, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.api != nil {
		return m.api, nil
	}

	client, err := m.client.getClient(ctx)
	if err != nil {
		return nil, err
	}

	m.api = client
	return m.api, nil
}

type Tokenizer struct {
	modelName string
	mu        sync.Mutex
	local     *genaitokenizer.LocalTokenizer
}

func (t *Tokenizer) ID() string {
	return "gemini." + t.modelName
}

func (t *Tokenizer) CountTokens(ctx context.Context, text string) (tokens int, err error) {
	ctx, span := gai.StartOperationSpan(ctx, geminiTracerName, "ai.gemini", "ai.operation", "tokenizer.count_tokens",
		attribute.String("ai.provider", "gemini"),
		attribute.String("ai.model", t.modelName),
		attribute.String("ai.tokenizer", t.ID()),
		attribute.Int("ai.input_length", len(text)),
	)
	defer func() {
		span.SetAttributes(attribute.Int("ai.input_tokens", tokens))
		gai.EndSpan(span, err)
	}()

	local, err := t.getLocal()
	if err != nil {
		return 0, err
	}

	result, err := local.CountTokens(genai.Text(text), nil)
	if err != nil {
		return 0, err
	}
	return int(result.TotalTokens), nil
}

func (t *Tokenizer) Tokenize(ctx context.Context, text string) (tokens []string, err error) {
	ctx, span := gai.StartOperationSpan(ctx, geminiTracerName, "ai.gemini", "ai.operation", "tokenizer.tokenize",
		attribute.String("ai.provider", "gemini"),
		attribute.String("ai.model", t.modelName),
		attribute.String("ai.tokenizer", t.ID()),
		attribute.Int("ai.input_length", len(text)),
	)
	defer func() {
		span.SetAttributes(attribute.Int("ai.input_tokens", len(tokens)))
		gai.EndSpan(span, err)
	}()

	local, err := t.getLocal()
	if err != nil {
		return nil, err
	}

	result, err := local.ComputeTokens(genai.Text(text))
	if err != nil {
		return nil, err
	}

	for _, info := range result.TokensInfo {
		for _, token := range info.Tokens {
			tokens = append(tokens, string(token))
		}
	}
	return tokens, nil
}

func (t *Tokenizer) getLocal() (*genaitokenizer.LocalTokenizer, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.local != nil {
		return t.local, nil
	}

	local, err := genaitokenizer.NewLocalTokenizer(t.modelName)
	if err != nil {
		return nil, err
	}
	t.local = local
	return t.local, nil
}
