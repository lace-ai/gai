package ai

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
