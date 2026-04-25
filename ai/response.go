package ai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

type AIResponse struct {
	Text         string
	InputTokens  int
	OutputTokens int
}

type TokenType string

var (
	TokenTypeText     TokenType = "text"
	TokenTypeTought   TokenType = "tought"
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
	case TokenTypeTought:
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
func WrapStream(in <-chan Token) <-chan Token {
	out := make(chan Token, 8)

	go func() {
		defer close(out)

		deciding := true
		var pending []Token

		// JSON tracking state
		seenNonWS := false
		isJSONCandidate := false
		objDepth := 0
		arrDepth := 0
		inString := false
		escape := false

		flushPending := func() {
			for _, t := range pending {
				out <- t
			}
			pending = nil
		}

		maybeFinish := func() bool {
			if !isJSONCandidate {
				return false
			}
			if inString || objDepth != 0 || arrDepth != 0 {
				return false
			}

			payload := joinTokenData(pending)
			if tc, ok := parseToolCall(payload); ok {
				out <- Token{
					Type:     TokenTypeToolCall,
					Data:     payload,
					ToolCall: tc,
				}
			} else {
				flushPending()
			}

			pending = nil
			deciding = false
			return true
		}

		for t := range in {
			if !deciding {
				out <- t
				continue
			}

			// If non-text appears before decision, replay and passthrough.
			if t.Type != TokenTypeText {
				pending = append(pending, t)
				flushPending()
				deciding = false
				continue
			}

			pending = append(pending, t)

			for _, b := range t.Data {
				if !seenNonWS {
					if isWS(b) {
						continue
					}
					seenNonWS = true
					if b == '{' {
						isJSONCandidate = true
						objDepth = 1
					} else {
						isJSONCandidate = false
						flushPending()
						deciding = false
						break
					}
					continue
				}

				if !isJSONCandidate {
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

			if !deciding {
				continue
			}

			if maybeFinish() {
				continue
			}
		}

		// End of stream: unresolved buffer is not a tool call, replay it.
		if deciding && len(pending) > 0 {
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
