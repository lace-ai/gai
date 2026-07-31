package openai

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"
)

func (m *Model) generateResponses(ctx context.Context, req ai.AIRequest) (*ai.AIResponse, error) {
	params, err := buildResponsesParams(m.name, req)
	if err != nil {
		return nil, err
	}
	response, err := m.client(false).Responses.New(ctx, params)
	if err != nil {
		return nil, err
	}
	if response.Error.Message != "" {
		return nil, fmt.Errorf("OpenAI Responses API: %s", response.Error.Message)
	}
	return responseFromResponses(response), nil
}

func (m *Model) generateResponsesStream(ctx context.Context, out chan<- ai.Token, req ai.AIRequest) {
	params, err := buildResponsesParams(m.name, req)
	if err != nil {
		ai.SendToken(ctx, out, ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
		return
	}
	stream := m.client(true).Responses.NewStreaming(ctx, params)
	defer func() {
		if err := stream.Close(); err != nil && m.provider.debug != nil {
			m.provider.debug.Emit(ctx, gai.DebugEvent{
				Name:   "stream_close_failed",
				Source: "ai:openai.Model.generateResponsesStream",
				Err:    err,
			})
		}
	}()
	for stream.Next() {
		event := stream.Current()
		switch event.Type {
		case "response.output_text.delta":
			if text := event.Delta.OfString; text != "" && !ai.SendToken(ctx, out, ai.Token{Type: ai.TokenTypeText, Data: []byte(text), Text: text}) {
				return
			}
		case "response.output_item.done":
			if event.Item.Type != "function_call" {
				continue
			}
			args := json.RawMessage(event.Item.Arguments)
			if !json.Valid(args) {
				err := fmt.Errorf("invalid JSON arguments for tool %q", event.Item.Name)
				ai.SendToken(ctx, out, ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
				return
			}
			call := &ai.ToolCall{ID: event.Item.CallID, Type: "function", Name: event.Item.Name, Args: args}
			if !ai.SendToken(ctx, out, ai.Token{Type: ai.TokenTypeToolCall, Data: []byte(args), ToolCall: call}) {
				return
			}
		case "error":
			err := fmt.Errorf("OpenAI Responses API: %s", event.Message)
			ai.SendToken(ctx, out, ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
			return
		case "response.failed":
			err := fmt.Errorf("OpenAI Responses API: %s", event.Response.Error.Message)
			ai.SendToken(ctx, out, ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
			return
		}
	}
	if err := stream.Err(); err != nil {
		ai.SendToken(ctx, out, ai.Token{Type: ai.TokenTypeErr, Err: err, Text: err.Error()})
	}
}

func buildResponsesParams(model string, req ai.AIRequest) (responses.ResponseNewParams, error) {
	params := responses.ResponseNewParams{Model: model}
	if len(req.Messages) == 0 {
		params.Input.OfString = param.NewOpt(req.Prompt)
	} else {
		input, err := mapResponsesMessages(req.Messages)
		if err != nil {
			return responses.ResponseNewParams{}, err
		}
		params.Input.OfInputItemList = input
	}
	if req.MaxTokens > 0 {
		params.MaxOutputTokens = param.NewOpt(int64(req.MaxTokens))
	}
	switch req.ResponseFormat.Type {
	case "", ai.ResponseFormatText:
	case ai.ResponseFormatJSONObject:
		params.Text.Format.OfJSONObject = &shared.ResponseFormatJSONObjectParam{}
	case ai.ResponseFormatJSONSchema:
		var schema map[string]any
		if err := json.Unmarshal(req.ResponseFormat.Schema, &schema); err != nil {
			return responses.ResponseNewParams{}, err
		}
		params.Text.Format = responses.ResponseFormatTextConfigParamOfJSONSchema(req.ResponseFormat.Name, schema)
	default:
		return responses.ResponseNewParams{}, fmt.Errorf("%w: %s", ai.ErrInvalidResponseFormat, req.ResponseFormat.Type)
	}
	if req.Reasoning.Effort != "" {
		switch req.Reasoning.Effort {
		case ai.ReasoningEffortNone, ai.ReasoningEffortLow, ai.ReasoningEffortMedium, ai.ReasoningEffortHigh:
			params.Reasoning.Effort = shared.ReasoningEffort(req.Reasoning.Effort)
		default:
			return responses.ResponseNewParams{}, fmt.Errorf("%w: OpenAI reasoning effort %q", ai.ErrUnsupportedCapability, req.Reasoning.Effort)
		}
	}
	for _, definition := range req.Tools {
		if err := definition.Validate(); err != nil {
			return responses.ResponseNewParams{}, err
		}
		var schema map[string]any
		if err := json.Unmarshal(definition.Parameters, &schema); err != nil {
			return responses.ResponseNewParams{}, fmt.Errorf("decode tool %q schema: %w", definition.Name, err)
		}
		params.Tools = append(params.Tools, responses.ToolUnionParam{OfFunction: &responses.FunctionToolParam{
			Name: definition.Name, Description: param.NewOpt(definition.Description), Parameters: schema, Strict: param.NewOpt(false),
		}})
	}
	if len(req.Tools) > 0 {
		switch req.ToolChoice.Mode {
		case "", ai.ToolChoiceAuto:
			params.ToolChoice.OfToolChoiceMode = param.NewOpt(responses.ToolChoiceOptionsAuto)
		case ai.ToolChoiceNone:
			params.ToolChoice.OfToolChoiceMode = param.NewOpt(responses.ToolChoiceOptionsNone)
		case ai.ToolChoiceRequired:
			if len(req.ToolChoice.Names) == 1 {
				params.ToolChoice.OfFunctionTool = &responses.ToolChoiceFunctionParam{Name: req.ToolChoice.Names[0]}
				break
			}
			if len(req.ToolChoice.Names) > 1 {
				return responses.ResponseNewParams{}, fmt.Errorf("%w: Responses API supports one named function tool", ai.ErrUnsupportedCapability)
			}
			params.ToolChoice.OfToolChoiceMode = param.NewOpt(responses.ToolChoiceOptionsRequired)
		default:
			return responses.ResponseNewParams{}, fmt.Errorf("unsupported OpenAI tool choice mode %q", req.ToolChoice.Mode)
		}
	}
	return params, nil
}

func mapResponsesMessages(messages []ai.RequestMessage) (responses.ResponseInputParam, error) {
	input := make(responses.ResponseInputParam, 0, len(messages))
	for _, message := range messages {
		switch message.Role {
		case ai.RequestMessageRoleUser:
			input = append(input, responses.ResponseInputItemParamOfMessage(message.Text, responses.EasyInputMessageRoleUser))
		case ai.RequestMessageRoleAssistant:
			if message.Text != "" {
				input = append(input, responses.ResponseInputItemParamOfMessage(message.Text, responses.EasyInputMessageRoleAssistant))
			}
			for _, call := range message.ToolCalls {
				input = append(input, responses.ResponseInputItemParamOfFunctionCall(string(call.Arguments), call.ID, call.Name))
			}
		case ai.RequestMessageRoleTool:
			content := message.ToolResult.Content
			if message.ToolResult.IsError {
				encoded, _ := json.Marshal(map[string]string{"error": content})
				content = string(encoded)
			}
			input = append(input, responses.ResponseInputItemParamOfFunctionCallOutput(message.ToolResult.ToolCallID, content))
		default:
			return nil, fmt.Errorf("unsupported native message role %q", message.Role)
		}
	}
	return input, nil
}

func responseFromResponses(response *responses.Response) *ai.AIResponse {
	result := &ai.AIResponse{
		Raw: json.RawMessage(response.RawJSON()), Text: response.OutputText(), FinishReason: string(response.Status),
		InputTokens: int(response.Usage.InputTokens), OutputTokens: int(response.Usage.OutputTokens), ReasoningTokens: int(response.Usage.OutputTokensDetails.ReasoningTokens),
	}
	for _, output := range response.Output {
		if output.Type != "function_call" {
			continue
		}
		args := json.RawMessage(output.Arguments)
		if !json.Valid(args) {
			continue
		}
		result.ToolCalls = append(result.ToolCalls, ai.ToolCall{ID: output.CallID, Type: "function", Name: output.Name, Args: args})
	}
	return result
}
