package ai

import "encoding/json"

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
	Type TokenType
	Data []byte

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
