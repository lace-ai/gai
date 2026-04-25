package mistral

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

	"github.com/lace-ai/gai/ai"
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
	Stream    bool                 `json:"stream,omitempty"`
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

type chatCompletionStreamResponse struct {
	Choices []struct {
		Delta struct {
			Content   json.RawMessage `json:"content"`
			ToolCalls json.RawMessage `json:"tool_calls"`
		} `json:"delta"`
	} `json:"choices"`
}

type mistralStreamFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type mistralStreamToolCallEntry struct {
	ID       string                    `json:"id"`
	Type     string                    `json:"type"`
	Function mistralStreamFunctionCall `json:"function"`
}

func mapMistralToolCalls(raw json.RawMessage) ([]ai.ToolCall, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}

	var calls []mistralStreamToolCallEntry
	if err := json.Unmarshal(raw, &calls); err != nil {
		return nil, fmt.Errorf("decode tool_calls: %w", err)
	}

	result := make([]ai.ToolCall, 0, len(calls))
	for _, c := range calls {
		args := json.RawMessage(c.Function.Arguments)
		if len(args) == 0 {
			args = json.RawMessage("{}")
		}
		result = append(result, ai.ToolCall{
			ID:   c.Function.Name,
			Name: "function",
			Args: args,
		})
	}
	return result, nil
}

func (m *Model) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	out := make(chan ai.Token, 1)

	go func() {
		defer close(out)

		payload := chatCompletionRequest{
			Model: m.name,
			Messages: []chatMessageRequest{
				{
					Role:    "user",
					Content: req.Prompt.CombinedPrompt(),
				},
			},
			Stream: true,
		}
		if req.MaxTokens > 0 {
			payload.MaxTokens = &req.MaxTokens
		}

		body, err := json.Marshal(payload)
		if err != nil {
			out <- ai.Token{Err: err, Type: ai.TokenTypeErr}
			return
		}

		httpReq, err := http.NewRequestWithContext(
			ctx,
			http.MethodPost,
			m.client.baseURL+"/v1/chat/completions",
			bytes.NewReader(body),
		)
		if err != nil {
			out <- ai.Token{Err: err, Type: ai.TokenTypeErr}
			return
		}
		httpReq.Header.Set("Authorization", "Bearer "+m.client.apiKey)
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "text/event-stream")

		res, err := m.client.httpClient.Do(httpReq)
		if err != nil {
			out <- ai.Token{Err: err, Type: ai.TokenTypeErr}
			return
		}
		defer res.Body.Close()

		if res.StatusCode >= http.StatusMultipleChoices {
			const maxResponseBody = 1 << 20 // 1MB
			resBody, readErr := io.ReadAll(io.LimitReader(res.Body, maxResponseBody))
			if readErr != nil {
				out <- ai.Token{
					Err:  fmt.Errorf("mistral chat stream failed (status %d): %w", res.StatusCode, readErr),
					Type: ai.TokenTypeErr,
				}
				return
			}
			out <- ai.Token{
				Err:  fmt.Errorf("mistral chat stream failed (status %d): %s", res.StatusCode, string(resBody)),
				Type: ai.TokenTypeErr,
			}
			return
		}

		reader := bufio.NewReader(res.Body)
		var eventData strings.Builder

		flushEvent := func() error {
			raw := strings.TrimSpace(eventData.String())
			eventData.Reset()
			if raw == "" {
				return nil
			}
			if raw == "[DONE]" {
				return io.EOF
			}

			var chunk chatCompletionStreamResponse
			if err := json.Unmarshal([]byte(raw), &chunk); err != nil {
				return fmt.Errorf("decode stream chunk: %w", err)
			}
			if len(chunk.Choices) == 0 {
				return nil
			}

			text, err := extractStreamText(chunk.Choices[0].Delta.Content)
			if err != nil {
				return err
			}
			if text != "" {
				out <- ai.Token{Type: ai.TokenTypeText, Data: []byte(text)}
			}

			toolCalls := strings.TrimSpace(string(chunk.Choices[0].Delta.ToolCalls))
			if toolCalls != "" && toolCalls != "null" {
				calls, mapErr := mapMistralToolCalls(chunk.Choices[0].Delta.ToolCalls)
				if mapErr != nil {
					return fmt.Errorf("map tool_calls: %w", mapErr)
				}
				for _, tc := range calls {
					tcCopy := tc
					out <- ai.Token{
						Type:     ai.TokenTypeToolCall,
						Data:     append([]byte(nil), chunk.Choices[0].Delta.ToolCalls...),
						ToolCall: &tcCopy,
					}
				}
			}

			return nil
		}

		for {
			line, err := reader.ReadString('\n')
			if err != nil && !errors.Is(err, io.EOF) {
				out <- ai.Token{Err: fmt.Errorf("read stream: %w", err), Type: ai.TokenTypeErr}
				return
			}

			line = strings.TrimRight(line, "\r\n")
			if line == "" {
				if flushErr := flushEvent(); flushErr != nil {
					if errors.Is(flushErr, io.EOF) {
						return
					}
					out <- ai.Token{Err: flushErr, Type: ai.TokenTypeErr}
					return
				}
			} else if strings.HasPrefix(line, "data:") {
				if eventData.Len() > 0 {
					eventData.WriteByte('\n')
				}
				eventData.WriteString(strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			}

			if errors.Is(err, io.EOF) {
				if flushErr := flushEvent(); flushErr != nil && !errors.Is(flushErr, io.EOF) {
					out <- ai.Token{Err: flushErr, Type: ai.TokenTypeErr}
				}
				return
			}
		}
	}()

	return out
}

func extractStreamText(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}

	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, nil
	}

	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err == nil {
		var builder strings.Builder
		for _, p := range parts {
			if p.Text == "" {
				continue
			}
			if p.Type == "" || p.Type == "text" {
				builder.WriteString(p.Text)
			}
		}
		return builder.String(), nil
	}

	return "", fmt.Errorf("unsupported stream content payload: %s", string(raw))
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

	httpReq, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		m.client.baseURL+"/v1/chat/completions",
		bytes.NewReader(body),
	)
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

	const maxResponseBody = 1 << 20 // 10MB
	resBody, err := io.ReadAll(io.LimitReader(res.Body, maxResponseBody))
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
