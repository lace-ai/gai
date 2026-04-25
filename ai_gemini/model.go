package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/lace-ai/gai/ai"
	"google.golang.org/genai"
)

type Model struct {
	name   string
	client *Provider
	mu     sync.Mutex
	api    *genai.Client
}

func (m *Model) Name() string {
	return m.name
}

func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.api = nil
	return nil
}

func (m *Model) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	out := make(chan ai.Token, 1)

	go func() {
		defer close(out)

		client, err := m.getClient(ctx)
		if err != nil {
			out <- ai.Token{Err: err, Type: ai.TokenTypeErr, Text: err.Error()}
			return
		}

		var config *genai.GenerateContentConfig
		if req.MaxTokens > 0 {
			config = &genai.GenerateContentConfig{
				MaxOutputTokens: int32(req.MaxTokens),
			}
		}

		contents := genai.Text(req.Prompt.CombinedPrompt())

		for resp, err := range client.Models.GenerateContentStream(ctx, m.name, contents, config) {
			if err != nil {
				streamErr := fmt.Errorf("error generating content stream: %w", err)
				out <- ai.Token{Err: streamErr, Type: ai.TokenTypeErr, Text: streamErr.Error()}
				return
			}

			if resp == nil || len(resp.Candidates) == 0 || resp.Candidates[0] == nil || resp.Candidates[0].Content == nil {
				continue
			}

			for _, part := range resp.Candidates[0].Content.Parts {
				if part == nil {
					continue
				}

				rawPart, err := json.Marshal(part)
				if err != nil {
					encodeErr := fmt.Errorf("error encoding part: %w", err)
					out <- ai.Token{Err: encodeErr, Type: ai.TokenTypeErr, Text: encodeErr.Error()}
					return
				}

				switch {
				case part.Text != "":
					tokenType := ai.TokenTypeText
					if part.Thought {
						tokenType = ai.TokenTypeThought
					}
					out <- ai.Token{
						Type: tokenType,
						Data: rawPart,
						Text: part.Text,
					}
				case part.ToolCall != nil:
					toolCall, err := mapServerToolCall(part.ToolCall)
					if err != nil {
						mapErr := fmt.Errorf("error mapping tool call: %w", err)
						out <- ai.Token{Err: mapErr, Type: ai.TokenTypeErr, Text: mapErr.Error()}
						return
					}
					out <- ai.Token{
						Type:     ai.TokenTypeToolCall,
						Data:     rawPart,
						ToolCall: toolCall,
					}
				case part.FunctionCall != nil:
					toolCall, err := mapFunctionCall(part.FunctionCall)
					if err != nil {
						mapErr := fmt.Errorf("error mapping function call: %w", err)
						out <- ai.Token{Err: mapErr, Type: ai.TokenTypeErr, Text: mapErr.Error()}
						return
					}
					out <- ai.Token{
						Type:     ai.TokenTypeToolCall,
						Data:     rawPart,
						ToolCall: toolCall,
					}
				}
			}
		}
	}()

	return out
}

func (m *Model) Generate(ctx context.Context, req ai.AIRequest) (*ai.AIResponse, error) {
	client, err := m.getClient(ctx)
	if err != nil {
		return nil, err
	}

	result, err := client.Models.GenerateContent(
		ctx,
		m.name,
		genai.Text(req.Prompt.CombinedPrompt()),
		nil,
	)
	if err != nil {
		return nil, err
	}

	return &ai.AIResponse{
		Text:         result.Text(),
		InputTokens:  int(result.UsageMetadata.PromptTokenCount),
		OutputTokens: int(result.UsageMetadata.CandidatesTokenCount),
	}, nil
}

func mapServerToolCall(toolCall *genai.ToolCall) (*ai.ToolCall, error) {
	args, err := marshalArgs(toolCall.Args)
	if err != nil {
		return nil, err
	}

	return &ai.ToolCall{
		ID:   toolCall.ID,
		Name: string(toolCall.ToolType),
		Args: args,
	}, nil
}

func mapFunctionCall(functionCall *genai.FunctionCall) (*ai.ToolCall, error) {
	args, err := marshalArgs(functionCall.Args)
	if err != nil {
		return nil, err
	}

	return &ai.ToolCall{
		ID:   functionCall.ID,
		Name: functionCall.Name,
		Args: args,
	}, nil
}

func marshalArgs(args map[string]any) (json.RawMessage, error) {
	if args == nil {
		return nil, nil
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

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey: m.client.apiKey,
	})
	if err != nil {
		return nil, err
	}

	m.api = client
	return m.api, nil
}
