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
	prompt := req.Prompt.CombinedPrompt()

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
					"prompt":          req.Prompt,
					"combined_prompt": prompt,
					"max_tokens":      req.MaxTokens,
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
			out <- ai.Token{Err: err, Type: ai.TokenTypeErr, Text: err.Error()}
			return
		}

		var config *genai.GenerateContentConfig
		if req.MaxTokens > 0 {
			config = &genai.GenerateContentConfig{
				MaxOutputTokens: int32(req.MaxTokens),
			}
		}

		contents := genai.Text(prompt)

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
					token := buildTextToken(part)
					switch token.Type {
					case ai.TokenTypeThought:
						thoughtTokenCount++
					default:
						textTokenCount++
					}
					out <- token
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
						out <- ai.Token{Err: encodeErr, Type: ai.TokenTypeErr, Text: encodeErr.Error()}
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
					toolCallCount++
				}
			}
		}
	}()

	return out
}

func (m *Model) Generate(ctx context.Context, req ai.AIRequest) (response *ai.AIResponse, err error) {
	prompt := req.Prompt.CombinedPrompt()
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
				"prompt":          req.Prompt,
				"combined_prompt": prompt,
				"max_tokens":      req.MaxTokens,
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

	result, err := client.Models.GenerateContent(
		ctx,
		m.name,
		genai.Text(prompt),
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
	span.SetAttributes(
		attribute.Int("ai.input_tokens", inputTokens),
		attribute.Int("ai.output_tokens", outputTokens),
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
