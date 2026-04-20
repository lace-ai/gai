package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/HecoAI/gai/ai"
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
			out <- ai.Token{Err: err, Type: ai.TokenTypeErr}
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
				out <- ai.Token{Err: fmt.Errorf("error generating content stream: %w", err), Type: ai.TokenTypeErr}
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
					tokenType := ai.TokenTypeText
					if part.Thought {
						tokenType = ai.TokenTypeTought
					}
					out <- ai.Token{
						Type: tokenType,
						Data: []byte(part.Text),
					}
				case part.ToolCall != nil:
					toolCall, err := json.Marshal(part.ToolCall)
					if err != nil {
						out <- ai.Token{Err: fmt.Errorf("error encoding tool call: %w", err), Type: ai.TokenTypeErr}
						return
					}
					out <- ai.Token{
						Type: ai.TokenTypeToolCall,
						Data: toolCall,
					}
				case part.FunctionCall != nil:
					functionCall, err := json.Marshal(part.FunctionCall)
					if err != nil {
						out <- ai.Token{Err: fmt.Errorf("error encoding function call: %w", err), Type: ai.TokenTypeErr}
						return
					}
					out <- ai.Token{
						Type: ai.TokenTypeToolCall,
						Data: functionCall,
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
