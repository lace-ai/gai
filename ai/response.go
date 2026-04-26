package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/lace-ai/gai"
)

type AIResponse struct {
	Text         string
	InputTokens  int
	OutputTokens int
}

type TokenType string

var (
	TokenTypeText     TokenType = "text"
	TokenTypeThought  TokenType = "thought"
	TokenTypeToolCall TokenType = "tool_call"
	TokenTypeErr      TokenType = "error"
)

type Token struct {
	Type       TokenType
	Data       []byte
	TokenUsage int

	ToolCall *ToolCall
	Text     string
	Err      error
}

type ToolCall struct {
	ID   string
	Name string
	Args json.RawMessage
}

func (t Token) String() string {
	return string(t.Data)
}

func (tc *ToolCall) Validate() error {
	if tc == nil {
		return fmt.Errorf("%w: tool call nil", ErrInvalidToolCall)
	}
	if strings.TrimSpace(tc.ID) == "" {
		return fmt.Errorf("%w: id empty", ErrInvalidToolCall)
	}
	if tc.Name != "function" {
		return fmt.Errorf("%w: name not function", ErrInvalidToolCall)
	}
	return nil
}

func (r *AIResponse) AppendToken(t Token) {
	switch t.Type {
	case TokenTypeText:
		if len(t.Text) > 0 {
			r.Text += t.Text
		} else {
			r.Text += string(t.Data)
		}
	case TokenTypeThought:
		if len(t.Text) > 0 {
			r.Text += t.Text
		} else {
			r.Text += string(t.Data)
		}
	case TokenTypeErr:
		if t.Err != nil {
			r.Text += string(t.Err.Error())
		} else {
			r.Text += string(t.Data)
		}
	case TokenTypeToolCall:
		r.Text += string(t.Data)
	}

	r.OutputTokens += t.TokenUsage
}

func (tc *ToolCall) String() string {
	var builder strings.Builder
	builder.WriteString("id: ")
	builder.WriteString(tc.ID)
	builder.WriteString(",type: ")
	builder.WriteString(tc.Name)
	builder.WriteString(",arguments: ")
	builder.Write(tc.Args)

	return builder.String()
}

// WrapStream detects a leading JSON object in the text stream.
// If it is a valid tool call, emits one ToolCall token.
// Otherwise, replays buffered tokens and continues passthrough.
// Use it if your provider doesn't have a native tool call detection
func WrapStream(ctx context.Context, in <-chan Token, debug gai.DebugSink) <-chan Token {
	out := make(chan Token, 8)

	go func() {
		defer close(out)

		var pending []Token

		// JSON tracking state
		seenNonWS := false
		newLines := 0
		isJSONCandidate := false
		objDepth := 0
		arrDepth := 0
		inString := false
		escape := false

		flushPending := func() {
			for _, t := range pending {
				out <- t
			}
			newLines = 0
			isJSONCandidate = false
			objDepth = 0
			arrDepth = 0
			inString = false
			escape = false
			pending = nil
		}

		maybeToolCall := func(last string) bool {
			if !isJSONCandidate {
				return false
			}
			if inString || objDepth != 0 || arrDepth != 0 {
				if debug != nil {
					debug.Emit(ctx, gai.DebugEvent{
						Name:   "wrap_stream_non_tool_call",
						Source: "ai:WrapStream.maybeToolCall",
						Fields: map[string]any{
							"reason": fmt.Sprintf("inString=%v objDepth=%d arrDepth=%d", inString, objDepth, arrDepth),
							"data":   string(joinTokenData(pending)),
						},
					})
				}
				return false
			}

			payload := append(joinTokenData(pending[:len(pending)-1]), []byte(last)...)
			if tc, ok := parseToolCall(payload); ok {
				if debug != nil {
					debug.Emit(ctx, gai.DebugEvent{
						Name:   "wrap_stream_tool_call_detected",
						Source: "ai:WrapStream.maybeToolCall",
						Fields: map[string]any{
							"id":   tc.ID,
							"name": tc.Name,
							"args": string(tc.Args),
						},
					})
				}
				out <- Token{
					Type:     TokenTypeToolCall,
					Data:     payload,
					ToolCall: tc,
				}
			} else {
				if debug != nil {
					debug.Emit(ctx, gai.DebugEvent{
						Name:   "wrap_stream_tool_call_parse_failed",
						Source: "ai:WrapStream.maybeToolCall",
						Fields: map[string]any{
							"reason": "parse failed",
							"data":   string(payload),
						},
					})
				}
				flushPending()
			}

			pending = nil
			return true
		}

		for t := range in {
			// non-text tokens: passthrough.
			if t.Type != TokenTypeText {
				pending = append(pending, t)
				flushPending()
				continue
			}

			pending = append(pending, t)

			var tokenStr strings.Builder
			for _, b := range t.Data {
				tokenStr.WriteByte(b)

				if !seenNonWS && !isJSONCandidate {
					if isWS(b) {
						continue
					}
					seenNonWS = true
					if b == '{' {
						if debug != nil {
							debug.Emit(ctx, gai.DebugEvent{
								Name:   "wrap_stream_json_candidate",
								Source: "ai:WrapStream",
								Fields: map[string]any{
									"data": string(tokenStr.String()),
								},
							})
						}
						isJSONCandidate = true
						objDepth = 1
					}
					continue
				}

				if !isJSONCandidate {
					if b == '\n' {
						newLines++
					}
					if newLines >= 2 && b == '{' {
						if debug != nil {
							debug.Emit(ctx, gai.DebugEvent{
								Name:   "wrap_stream_json_candidate_after_newlines",
								Source: "ai:WrapStream",
								Fields: map[string]any{
									"data": string(tokenStr.String()),
								},
							})
						}
						isJSONCandidate = true
						objDepth = 1
						newLines = 0
					}
					continue
				}

				if inString {
					if escape {
						escape = false
						continue
					}
					if b == '\\' {
						escape = true
						continue
					}
					if b == '"' {
						inString = false
					}
					continue
				}

				switch b {
				case '"':
					inString = true
				case '{':
					objDepth++
				case '}':
					objDepth--
				case '[':
					arrDepth++
				case ']':
					arrDepth--
				}
			}

			if maybeToolCall(tokenStr.String()) {
				continue
			}
		}

		if debug != nil {
			debug.Emit(ctx, gai.DebugEvent{
				Name:   "wrap_stream_end_of_stream",
				Source: "ai:WrapStream",
				Fields: map[string]any{
					"pending_data": string(joinTokenData(pending)),
				},
			})
		}

		// End of stream: unresolved buffer is not a tool call, replay it.
		if len(pending) > 0 {
			flushPending()
		}
	}()

	return out
}

func parseToolCall(payload []byte) (*ToolCall, bool) {
	s := strings.TrimSpace(string(payload))
	if s == "" || !strings.HasPrefix(s, "{") {
		return nil, false
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal([]byte(s), &raw); err != nil {
		return nil, false
	}

	var id, typ string
	var args json.RawMessage

	if v, ok := raw["id"]; ok {
		if err := json.Unmarshal(v, &id); err != nil {
			return nil, false
		}
	}
	if v, ok := raw["name"]; ok {
		if err := json.Unmarshal(v, &typ); err != nil {
			return nil, false
		}
	}
	if v, ok := raw["arguments"]; ok {
		args = v
	}

	if strings.TrimSpace(id) == "" {
		return nil, false
	}
	if typ != "function" {
		return nil, false
	}
	if len(args) == 0 {
		args = json.RawMessage("{}")
	}

	return &ToolCall{
		ID:   id,
		Name: typ,
		Args: args,
	}, true
}

func joinTokenData(tokens []Token) []byte {
	var b bytes.Buffer
	for _, t := range tokens {
		b.Write(t.Data)
	}
	return b.Bytes()
}

func isWS(b byte) bool {
	return b == ' ' || b == '\n' || b == '\r' || b == '\t'
}
