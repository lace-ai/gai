package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"google.golang.org/genai"
)

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

				switch {
				case part.Text != "":
					out <- buildTextToken(part)
				case part.FunctionCall != nil:
					rawPart, err := json.Marshal(part)
					if err != nil {
						encodeErr := fmt.Errorf("error encoding part: %w", err)
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
						out <- ai.Token{Err: encodeErr, Type: ai.TokenTypeErr, Text: encodeErr.Error()}
						return
					}
					toolCall, err := mapFunctionCall(part.FunctionCall)
					if err != nil {
						mapErr := fmt.Errorf("error mapping function call: %w", err)
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
						out <- ai.Token{Err: mapErr, Type: ai.TokenTypeErr, Text: mapErr.Error()}
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

	result, err := client.Models.GenerateContent(
		ctx,
		m.name,
		genai.Text(req.Prompt.CombinedPrompt()),
		nil,
	)
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

	if m.debug != nil {
		m.debug.Emit(ctx, gai.DebugEvent{
			Name:   "gemini_generate_content_success",
			Source: "ai:gemini.Model.Generate",
			Fields: map[string]any{
				"input_tokens":  inputTokens,
				"output_tokens": outputTokens,
			},
		})
	}

	return &ai.AIResponse{
		Text:         result.Text(),
		InputTokens:  inputTokens,
		OutputTokens: outputTokens,
	}, nil
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

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey: m.client.apiKey,
	})
	if err != nil {
		return nil, err
	}

	m.api = client
	return m.api, nil
}
