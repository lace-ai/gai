package anthropic

const (
	ClaudeOpus4_6   = "claude-opus-4-6"
	ClaudeSonnet4_6 = "claude-sonnet-4-6"
	ClaudeHaiku4_5  = "claude-haiku-4-5"
)

func supportsAdaptiveThinking(model string) bool {
	return model == ClaudeOpus4_6 || model == ClaudeSonnet4_6
}

var models = []string{ClaudeOpus4_6, ClaudeSonnet4_6, ClaudeHaiku4_5}
