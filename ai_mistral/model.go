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

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
)

type Model struct {
	name   string
	client *Provider
	debug  gai.DebugSink
}

var _ ai.Model = (*Model)(nil)

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
		toolName := strings.TrimSpace(c.Function.Name)
		args := json.RawMessage(c.Function.Arguments)
		if len(args) == 0 {
			args = json.RawMessage("{}")
		}
		callType := strings.TrimSpace(c.Type)
		if callType == "" {
			callType = "function"
		}
		result = append(result, ai.ToolCall{
			ID:   ai.GenerateToolCallID(toolName),
			Type: callType,
			Name: toolName,
			Args: args,
		})
	}
	return result, nil
}

func (m *Model) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	raw := make(chan ai.Token, 1)

	go func() {
		defer close(raw)

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
			if m.debug != nil {
				fields := map[string]any{
					"error": err.Error(),
				}
				if m.debug.IncludeSensitiveData() {
					fields["payload"] = payload
				}
				m.debug.Emit(ctx, gai.DebugEvent{
					Name:   "mistral_stream_request_payload",
					Source: "ai:mistral.Model.GenerateStream",
					Fields: fields,
					Err:    err,
				})
			}
			raw <- ai.Token{Err: err, Type: ai.TokenTypeErr}
			return
		}

		httpReq, err := http.NewRequestWithContext(
			ctx,
			http.MethodPost,
			m.client.baseURL+"/v1/chat/completions",
			bytes.NewReader(body),
		)
		if err != nil {
			if m.debug != nil {
				m.debug.Emit(ctx, gai.DebugEvent{
					Name:   "mistral_stream_request_creation_failed",
					Source: "ai:mistral.Model.GenerateStream",
					Fields: map[string]any{
						"error": err.Error(),
					},
					Err: err,
				})
			}
			raw <- ai.Token{Err: err, Type: ai.TokenTypeErr}
			return
		}
		httpReq.Header.Set("Authorization", "Bearer "+m.client.apiKey)
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "text/event-stream")

		res, err := m.client.httpClient.Do(httpReq)
		if err != nil {
			if m.debug != nil {
				m.debug.Emit(ctx, gai.DebugEvent{
					Name:   "mistral_stream_request_failed",
					Source: "ai:mistral.Model.GenerateStream",
					Fields: map[string]any{
						"error": err.Error(),
					},
					Err: err,
				})
			}
			raw <- ai.Token{Err: err, Type: ai.TokenTypeErr}
			return
		}
		defer res.Body.Close()

		if res.StatusCode >= http.StatusMultipleChoices {
			const maxResponseBody = 1 << 20 // 1MB
			resBody, readErr := io.ReadAll(io.LimitReader(res.Body, maxResponseBody))
			if readErr != nil {
				if m.debug != nil {
					m.debug.Emit(ctx, gai.DebugEvent{
						Name:   "mistral_stream_request_failed_with_unreadable_body",
						Source: "ai:mistral.Model.GenerateStream",
						Fields: map[string]any{
							"status_code": res.StatusCode,
							"error":       readErr.Error(),
						},
						Err: readErr,
					})
				}
				raw <- ai.Token{
					Err:  fmt.Errorf("mistral chat stream failed (status %d): %w", res.StatusCode, readErr),
					Type: ai.TokenTypeErr,
				}
				return
			}
			if m.debug != nil {
				fields := map[string]any{
					"status_code": res.StatusCode,
				}
				if m.debug.IncludeSensitiveData() {
					fields["response"] = string(resBody)
				}
				m.debug.Emit(ctx, gai.DebugEvent{
					Name:   "mistral_stream_request_failed",
					Source: "ai:mistral.Model.GenerateStream",
					Fields: fields,
				})
			}
			raw <- ai.Token{
				Err:  fmt.Errorf("mistral chat stream failed (status %d): %s", res.StatusCode, string(resBody)),
				Type: ai.TokenTypeErr,
			}
			return
		}

		reader := bufio.NewReader(res.Body)
		var eventData strings.Builder

		flushEvent := func() error {
			event := strings.TrimSpace(eventData.String())
			eventData.Reset()
			if event == "" {
				return nil
			}
			if event == "[DONE]" {
				return io.EOF
			}

			var chunk chatCompletionStreamResponse
			if err := json.Unmarshal([]byte(event), &chunk); err != nil {
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
				raw <- ai.Token{Type: ai.TokenTypeText, Text: text, Data: []byte(text)}
			}

			toolCalls := strings.TrimSpace(string(chunk.Choices[0].Delta.ToolCalls))
			if toolCalls != "" && toolCalls != "null" {
				calls, mapErr := mapMistralToolCalls(chunk.Choices[0].Delta.ToolCalls)
				if mapErr != nil {
					if m.debug != nil {
						fields := map[string]any{
							"error": mapErr.Error(),
						}
						if m.debug.IncludeSensitiveData() {
							fields["tool_calls"] = string(chunk.Choices[0].Delta.ToolCalls)
						}
						m.debug.Emit(ctx, gai.DebugEvent{
							Name:   "mistral_stream_tool_calls_mapping_failed",
							Source: "ai:mistral.Model.GenerateStream",
							Fields: fields,
							Err:    mapErr,
						})
					}
					return fmt.Errorf("map tool_calls: %w", mapErr)
				}
				if m.debug != nil {
					fields := map[string]any{}
					if m.debug.IncludeSensitiveData() {
						fields["tool_calls"] = calls
					}
					m.debug.Emit(ctx, gai.DebugEvent{
						Name:   "mistral_stream_tool_calls_mapped",
						Source: "ai:mistral.Model.GenerateStream",
						Fields: fields,
					})
				}
				for _, tc := range calls {
					tcCopy := tc
					raw <- ai.Token{
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
				if m.debug != nil {
					m.debug.Emit(ctx, gai.DebugEvent{
						Name:   "mistral_stream_read_failed",
						Source: "ai:mistral.Model.GenerateStream",
						Fields: map[string]any{
							"error": err.Error(),
						},
						Err: err,
					})
				}
				raw <- ai.Token{Err: fmt.Errorf("read stream: %w", err), Type: ai.TokenTypeErr}
				return
			}

			line = strings.TrimRight(line, "\r\n")
			if line == "" {
				if flushErr := flushEvent(); flushErr != nil {
					if errors.Is(flushErr, io.EOF) {
						if m.debug != nil {
							m.debug.Emit(ctx, gai.DebugEvent{
								Name:   "mistral_stream_read_eof",
								Source: "ai:mistral.Model.GenerateStream",
								Fields: map[string]any{},
							})
						}
						return
					}
					if m.debug != nil {
						m.debug.Emit(ctx, gai.DebugEvent{
							Name:   "mistral_stream_chunk_processing_failed",
							Source: "ai:mistral.Model.GenerateStream",
							Fields: map[string]any{
								"error": flushErr.Error(),
							},
							Err: flushErr,
						})
					}
					raw <- ai.Token{Err: flushErr, Type: ai.TokenTypeErr}
					return
				}
			} else if strings.HasPrefix(line, "data:") {
				if eventData.Len() > 0 {
					eventData.WriteByte('\n')
				}
				eventData.WriteString(strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			}

			if errors.Is(err, io.EOF) {
				if m.debug != nil {
					m.debug.Emit(ctx, gai.DebugEvent{
						Name:   "mistral_stream_read_eof",
						Source: "ai:mistral.Model.GenerateStream",
						Fields: map[string]any{},
					})
				}
				if flushErr := flushEvent(); flushErr != nil && !errors.Is(flushErr, io.EOF) {
					if m.debug != nil {
						m.debug.Emit(ctx, gai.DebugEvent{
							Name:   "mistral_stream_final_flush_failed",
							Source: "ai:mistral.Model.GenerateStream",
							Fields: map[string]any{
								"error": flushErr.Error(),
							},
							Err: flushErr,
						})
					}
					raw <- ai.Token{Err: flushErr, Type: ai.TokenTypeErr}
				}
				return
			}
		}
	}()

	return ai.WrapStream(ctx, raw, m.debug)
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
