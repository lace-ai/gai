package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/lace-ai/gai"
)

type ToolCall struct {
	ID   string
	Type string
	Name string
	Args json.RawMessage
}

func (tc *ToolCall) Validate() error {
	if tc == nil {
		return fmt.Errorf("%w: tool call nil", ErrInvalidToolCall)
	}
	if strings.TrimSpace(tc.ID) == "" {
		return fmt.Errorf("%w: id empty", ErrInvalidToolCall)
	}
	if tc.Type != "function" {
		return fmt.Errorf("%w: type not function", ErrInvalidToolCall)
	}
	if strings.TrimSpace(tc.Name) == "" {
		return fmt.Errorf("%w: name empty", ErrInvalidToolCall)
	}
	return nil
}

func (tc *ToolCall) String() string {
	var builder strings.Builder
	builder.WriteString("id: ")
	builder.WriteString(tc.ID)
	builder.WriteString(",type: ")
	builder.WriteString(tc.Type)
	builder.WriteString(",name: ")
	builder.WriteString(tc.Name)
	builder.WriteString(",arguments: ")
	builder.Write(tc.Args)

	return builder.String()
}

var toolCallCounter uint64

func GenerateToolCallID(toolName string) string {
	name := strings.TrimSpace(toolName)
	if name == "" {
		name = "tool"
	}
	seq := atomic.AddUint64(&toolCallCounter, 1)
	return fmt.Sprintf("call_%s_%d_%d", name, time.Now().UnixNano(), seq)
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

// DetectToolCallsInStream scans a token stream for text-encoded tool calls.
// When it detects a valid tool-call JSON object, it emits a ToolCall token.
// Otherwise, buffered tokens are replayed unchanged.
func DetectToolCallsInStream(ctx context.Context, in <-chan Token, debug gai.DebugSink) <-chan Token {
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
						Name:   "tool_call_stream_non_tool_call",
						Source: "ai:DetectToolCallsInStream.maybeToolCall",
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
						Name:   "tool_call_stream_tool_call_detected",
						Source: "ai:DetectToolCallsInStream.maybeToolCall",
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
						Name:   "tool_call_stream_tool_call_parse_failed",
						Source: "ai:DetectToolCallsInStream.maybeToolCall",
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
									Name:   "tool_call_stream_json_candidate",
									Source: "ai:DetectToolCallsInStream",
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
									Name:   "tool_call_stream_json_candidate_after_newlines",
									Source: "ai:DetectToolCallsInStream",
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
				Name:   "tool_call_stream_end_of_stream",
				Source: "ai:DetectToolCallsInStream",
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
