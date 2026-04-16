package gemini

import (
	"context"
	"sync"

	"agent-backend/gai/ai"

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
