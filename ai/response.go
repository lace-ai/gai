package ai

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
	Err  error
}

func (t Token) String() string {
	return string(t.Data)
}
