package mistral

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"agent-backend/gai/ai"
)

type Model struct {
	name   string
	client *Provider
}

func (m *Model) Name() string {
	return m.name
}

func (m *Model) Close() error {
	return nil
}

type chatCompletionRequest struct {
	Model     string               `json:"model"`
	Messages  []chatMessageRequest `json:"messages"`
	MaxTokens *int                 `json:"max_tokens,omitempty"`
}

type chatMessageRequest struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatCompletionResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

func (m *Model) Generate(ctx context.Context, req ai.AIRequest) (*ai.AIResponse, error) {
	payload := chatCompletionRequest{
		Model: m.name,
		Messages: []chatMessageRequest{
			{
				Role:    "user",
				Content: req.Prompt.CombinedPrompt(),
			},
		},
	}
	if req.MaxTokens > 0 {
		payload.MaxTokens = &req.MaxTokens
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.client.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Authorization", "Bearer "+m.client.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")

	res, err := m.client.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	resBody, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, err
	}

	if res.StatusCode >= http.StatusMultipleChoices {
		return nil, fmt.Errorf("mistral chat completion failed (status %d): %s", res.StatusCode, string(resBody))
	}

	var parsed chatCompletionResponse
	if err := json.Unmarshal(resBody, &parsed); err != nil {
		return nil, err
	}
	if len(parsed.Choices) == 0 {
		return nil, ErrNoChoices
	}

	return &ai.AIResponse{
		Text:         parsed.Choices[0].Message.Content,
		InputTokens:  parsed.Usage.PromptTokens,
		OutputTokens: parsed.Usage.CompletionTokens,
	}, nil
}
