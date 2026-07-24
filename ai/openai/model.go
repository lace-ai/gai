package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	sdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/shared"
)

// Model is an OpenAI chat-completions model.
type Model struct {
	name     string
	provider *Provider
}

var _ ai.Model = (*Model)(nil)

func (m *Model) Name() string { return m.name }

func (m *Model) Close() error { return nil }

func (m *Model) Tokenizer() ai.Tokenizer { return tokenizer{modelName: m.name} }

func (m *Model) Generate(ctx context.Context, req ai.AIRequest) (*ai.AIResponse, error) {
	params, err := buildChatCompletionParams(m.name, req, false)
	if err != nil {
		return nil, err
	}
	response, err := m.client(false).Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, err
	}
	result := &ai.AIResponse{
		Raw:             json.RawMessage(response.RawJSON()),
		InputTokens:     int(response.Usage.PromptTokens),
		OutputTokens:    int(response.Usage.CompletionTokens),
		ReasoningTokens: int(response.Usage.CompletionTokensDetails.ReasoningTokens),
	}
	if len(response.Choices) == 0 {
		return result, nil
	}
	message := response.Choices[0].Message
	result.Text = message.Content
	for _, call := range message.ToolCalls {
		args := json.RawMessage(strings.TrimSpace(call.Function.Arguments))
		if len(args) == 0 {
			args = json.RawMessage("{}")
		}
		if !json.Valid(args) {
			return nil, fmt.Errorf("invalid JSON arguments for tool %q", call.Function.Name)
		}
		result.ToolCalls = append(result.ToolCalls, ai.ToolCall{
			ID: call.ID, Type: "function", Name: call.Function.Name, Args: args,
		})
	}
	return result, nil
}

func (m *Model) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	out := make(chan ai.Token, 1)
	go func() {
		defer close(out)
		params, err := buildChatCompletionParams(m.name, req, true)
		if err != nil {
			ai.SendToken(ctx, out, ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
			return
		}
		stream := m.client(true).Chat.Completions.NewStreaming(ctx, params)
		defer func() {
			if err := stream.Close(); err != nil && m.provider.debug != nil {
				m.provider.debug.Emit(ctx, gai.DebugEvent{
					Name:   "stream_close_failed",
					Source: "ai:openai.Model.GenerateStream",
					Err:    err,
				})
			}
		}()
		calls := map[int64]*streamToolCall{}
		for stream.Next() {
			chunk := stream.Current()
			if len(chunk.Choices) == 0 {
				continue
			}
			choice := chunk.Choices[0]
			if text := choice.Delta.Content; text != "" {
				if !ai.SendToken(ctx, out, ai.Token{Type: ai.TokenTypeText, Data: []byte(text), Text: text}) {
					return
				}
			}
			for _, delta := range choice.Delta.ToolCalls {
				call := calls[delta.Index]
				if call == nil {
					call = &streamToolCall{}
					calls[delta.Index] = call
				}
				call.id.WriteString(delta.ID)
				call.typ = firstNonEmpty(call.typ, delta.Type)
				call.name.WriteString(delta.Function.Name)
				call.arguments.WriteString(delta.Function.Arguments)
			}
			if choice.FinishReason == "tool_calls" {
				if !sendStreamToolCalls(ctx, out, calls) {
					return
				}
			}
		}
		if err := stream.Err(); err != nil {
			ai.SendToken(ctx, out, ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
			return
		}
		sendStreamToolCalls(ctx, out, calls)
	}()
	return out
}

type streamToolCall struct {
	id, name, arguments strings.Builder
	typ                 string
	emitted             bool
}

func sendStreamToolCalls(ctx context.Context, out chan<- ai.Token, calls map[int64]*streamToolCall) bool {
	indices := make([]int64, 0, len(calls))
	for index := range calls {
		indices = append(indices, index)
	}
	sort.Slice(indices, func(i, j int) bool { return indices[i] < indices[j] })
	for _, index := range indices {
		call := calls[index]
		if call.emitted {
			continue
		}
		call.emitted = true
		name := call.name.String()
		if strings.TrimSpace(name) == "" {
			continue
		}
		args := call.arguments.String()
		if !json.Valid([]byte(args)) {
			err := fmt.Errorf("invalid JSON arguments for tool %q", name)
			return ai.SendToken(ctx, out, ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
		}
		toolCall := &ai.ToolCall{ID: call.id.String(), Type: firstNonEmpty(call.typ, "function"), Name: name, Args: json.RawMessage(args)}
		if !ai.SendToken(ctx, out, ai.Token{Type: ai.TokenTypeToolCall, Data: []byte(args), ToolCall: toolCall}) {
			return false
		}
	}
	return true
}

func buildChatCompletionParams(model string, req ai.AIRequest, stream bool) (sdk.ChatCompletionNewParams, error) {
	if err := req.ResponseFormat.Validate(); err != nil {
		return sdk.ChatCompletionNewParams{}, err
	}
	params := sdk.ChatCompletionNewParams{Model: shared.ChatModel(model), Messages: []sdk.ChatCompletionMessageParamUnion{sdk.UserMessage(req.Prompt)}}
	if req.MaxTokens > 0 {
		params.MaxCompletionTokens = param.NewOpt(int64(req.MaxTokens))
	}
	if stream {
		params.StreamOptions.IncludeUsage = param.NewOpt(true)
	}
	if err := applyTools(&params, req.Tools, req.ToolChoice); err != nil {
		return sdk.ChatCompletionNewParams{}, err
	}
	if err := applyResponseFormat(&params, req.ResponseFormat); err != nil {
		return sdk.ChatCompletionNewParams{}, err
	}
	if req.Reasoning.Effort != "" && !isReasoningModel(model) {
		return sdk.ChatCompletionNewParams{}, fmt.Errorf("%w: reasoning effort is unsupported for model %q", ai.ErrUnsupportedCapability, model)
	}
	switch req.Reasoning.Effort {
	case "", ai.ReasoningEffortLow, ai.ReasoningEffortMedium, ai.ReasoningEffortHigh:
		params.ReasoningEffort = shared.ReasoningEffort(req.Reasoning.Effort)
	default:
		return sdk.ChatCompletionNewParams{}, fmt.Errorf("%w: OpenAI reasoning effort %q", ai.ErrUnsupportedCapability, req.Reasoning.Effort)
	}
	return params, nil
}

func isReasoningModel(model string) bool {
	switch model {
	case O3, O3Mini, O4Mini:
		return true
	default:
		return false
	}
}

func applyTools(params *sdk.ChatCompletionNewParams, definitions []ai.ToolDefinition, choice ai.ToolChoice) error {
	if len(definitions) == 0 {
		return nil
	}
	params.Tools = make([]sdk.ChatCompletionToolParam, 0, len(definitions))
	available := make(map[string]struct{}, len(definitions))
	for _, definition := range definitions {
		if err := definition.Validate(); err != nil {
			return err
		}
		var schema shared.FunctionParameters
		if err := json.Unmarshal(definition.Parameters, &schema); err != nil {
			return fmt.Errorf("decode tool %q schema: %w", definition.Name, err)
		}
		params.Tools = append(params.Tools, sdk.ChatCompletionToolParam{Function: shared.FunctionDefinitionParam{Name: definition.Name, Description: param.NewOpt(definition.Description), Parameters: schema}})
		available[definition.Name] = struct{}{}
	}
	if choice.Mode == ai.ToolChoiceRequired && len(choice.Names) > 0 {
		allowed := make(map[string]struct{}, len(choice.Names))
		for _, name := range choice.Names {
			if _, ok := available[name]; !ok {
				return fmt.Errorf("required tool %q is not defined", name)
			}
			allowed[name] = struct{}{}
		}
		if len(choice.Names) > 1 {
			filtered := make([]sdk.ChatCompletionToolParam, 0, len(allowed))
			for _, tool := range params.Tools {
				if _, ok := allowed[tool.Function.Name]; ok {
					filtered = append(filtered, tool)
				}
			}
			params.Tools = filtered
		}
	}
	switch choice.Mode {
	case "", ai.ToolChoiceAuto, ai.ToolChoiceNone, ai.ToolChoiceRequired:
		mode := string(choice.Mode)
		if mode == "" {
			mode = string(ai.ToolChoiceAuto)
		}
		params.ToolChoice.OfAuto = param.NewOpt(mode)
		if choice.Mode == ai.ToolChoiceRequired && len(choice.Names) == 1 {
			params.ToolChoice = sdk.ChatCompletionToolChoiceOptionParamOfChatCompletionNamedToolChoice(sdk.ChatCompletionNamedToolChoiceFunctionParam{Name: choice.Names[0]})
		}
		return nil
	default:
		return fmt.Errorf("unsupported openai tool choice mode %q", choice.Mode)
	}
}

func applyResponseFormat(params *sdk.ChatCompletionNewParams, format ai.ResponseFormat) error {
	switch format.Type {
	case "", ai.ResponseFormatText:
		return nil
	case ai.ResponseFormatJSONObject:
		params.ResponseFormat.OfJSONObject = &shared.ResponseFormatJSONObjectParam{}
		return nil
	case ai.ResponseFormatJSONSchema:
		var schema any
		if err := json.Unmarshal(format.Schema, &schema); err != nil {
			return err
		}
		params.ResponseFormat.OfJSONSchema = &shared.ResponseFormatJSONSchemaParam{JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{Name: format.Name, Schema: schema}}
		return nil
	default:
		return fmt.Errorf("%w: %s", ai.ErrInvalidResponseFormat, format.Type)
	}
}

func (m *Model) client(streaming bool) *sdk.Client {
	httpClient := m.provider.httpClient
	if streaming {
		httpClient = m.provider.streamingHTTPClient()
	}
	client := sdk.NewClient(option.WithAPIKey(m.provider.apiKey), option.WithBaseURL(m.provider.baseURL), option.WithHTTPClient(httpClient), option.WithMaxRetries(0))
	return &client
}

type tokenizer struct{ modelName string }

func (t tokenizer) ID() string { return "openai." + t.modelName }

func (tokenizer) CountTokens(context.Context, string) (int, error) {
	return 0, ai.ErrTokenizerUnsupported
}

func (tokenizer) Tokenize(context.Context, string) ([]string, error) {
	return nil, ai.ErrTokenizerUnsupported
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
