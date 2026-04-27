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

func (t Token) String() string {
	return string(t.Data)
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

		resetTracking := func() {
			newLines = 0
			isJSONCandidate = false
			objDepth = 0
			arrDepth = 0
			inString = false
			escape = false
		}

		flushPending := func() {
			for _, t := range pending {
				out <- t
			}
			resetTracking()
			pending = nil
		}

		maybeToolCall := func(last string) bool {
			if !isJSONCandidate {
				return false
			}
			if inString || objDepth != 0 || arrDepth != 0 {
				if debug != nil {
					fields := map[string]any{
						"reason": fmt.Sprintf("inString=%v objDepth=%d arrDepth=%d", inString, objDepth, arrDepth),
					}
					if debug.IncludeSensitiveData() {
						fields["data"] = string(joinTokenData(pending))
					}
					debug.Emit(ctx, gai.DebugEvent{
						Name:   "wrap_stream_non_tool_call",
						Source: "ai:WrapStream.maybeToolCall",
						Fields: fields,
					})
				}
				return false
			}

			payload := []byte(last)
			if len(pending) > 0 {
				payload = append(joinTokenData(pending[:len(pending)-1]), payload...)
			}
			if tc, ok := parseToolCall(payload); ok {
				if debug != nil {
					fields := map[string]any{
						"id":   tc.ID,
						"type": tc.Type,
						"name": tc.Name,
					}
					if debug.IncludeSensitiveData() {
						fields["args"] = string(tc.Args)
					}
					debug.Emit(ctx, gai.DebugEvent{
						Name:   "wrap_stream_tool_call_detected",
						Source: "ai:WrapStream.maybeToolCall",
						Fields: fields,
					})
				}
				out <- Token{
					Type:     TokenTypeToolCall,
					Data:     payload,
					ToolCall: tc,
				}
				resetTracking()
			} else {
				if debug != nil {
					fields := map[string]any{
						"reason": "parse failed",
					}
					if debug.IncludeSensitiveData() {
						fields["data"] = string(payload)
					}
					debug.Emit(ctx, gai.DebugEvent{
						Name:   "wrap_stream_tool_call_parse_failed",
						Source: "ai:WrapStream.maybeToolCall",
						Fields: fields,
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

			remaining := t.Data
			for len(remaining) > 0 {
				pending = append(pending, Token{Type: TokenTypeText, Data: remaining})

				var tokenStr strings.Builder
				handledCandidate := false
				for idx, b := range remaining {
					tokenStr.WriteByte(b)

					if !seenNonWS && !isJSONCandidate {
						if isWS(b) {
							continue
						}
						seenNonWS = true
						if b == '{' {
							if debug != nil {
								fields := map[string]any{}
								if debug.IncludeSensitiveData() {
									fields["data"] = string(tokenStr.String())
								}
								debug.Emit(ctx, gai.DebugEvent{
									Name:   "wrap_stream_json_candidate",
									Source: "ai:WrapStream",
									Fields: fields,
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
								fields := map[string]any{}
								if debug.IncludeSensitiveData() {
									fields["data"] = string(tokenStr.String())
								}
								debug.Emit(ctx, gai.DebugEvent{
									Name:   "wrap_stream_json_candidate_after_newlines",
									Source: "ai:WrapStream",
									Fields: fields,
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

					// If JSON candidate is balanced at this byte, decide now.
					if isJSONCandidate && !inString && objDepth == 0 && arrDepth == 0 {
						if maybeToolCall(tokenStr.String()) {
							handledCandidate = true
							if idx+1 < len(remaining) {
								remaining = append([]byte(nil), remaining[idx+1:]...)
								seenNonWS = false
							} else {
								remaining = nil
							}
							break
						}
					}
				}

				if !handledCandidate {
					break
				}
			}
		}

		if debug != nil {
			fields := map[string]any{}
			if debug.IncludeSensitiveData() {
				fields["pending_data"] = string(joinTokenData(pending))
			}
			debug.Emit(ctx, gai.DebugEvent{
				Name:   "wrap_stream_end_of_stream",
				Source: "ai:WrapStream",
				Fields: fields,
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

	var typ, name string
	var args json.RawMessage

	if v, ok := raw["type"]; ok {
		if err := json.Unmarshal(v, &typ); err != nil {
			return nil, false
		}
	}
	if v, ok := raw["name"]; ok {
		if err := json.Unmarshal(v, &name); err != nil {
			return nil, false
		}
	}
	if v, ok := raw["arguments"]; ok {
		args = v
	}

	if typ != "function" {
		return nil, false
	}
	if strings.TrimSpace(name) == "" {
		return nil, false
	}
	if len(args) == 0 {
		args = json.RawMessage("{}")
	}

	return &ToolCall{
		ID:   GenerateToolCallID(name),
		Type: typ,
		Name: name,
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
