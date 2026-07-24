package anthropic

const (
	ClaudeFable5             = "claude-fable-5"
	ClaudeMythos5            = "claude-mythos-5"
	ClaudeOpus4_8            = "claude-opus-4-8"
	ClaudeSonnet5            = "claude-sonnet-5"
	ClaudeOpus4_7            = "claude-opus-4-7"
	ClaudeOpus4_6            = "claude-opus-4-6"
	ClaudeSonnet4_6          = "claude-sonnet-4-6"
	ClaudeOpus4_5            = "claude-opus-4-5"
	ClaudeSonnet4_5          = "claude-sonnet-4-5"
	ClaudeHaiku4_5           = "claude-haiku-4-5"
	ClaudeOpus4_1            = "claude-opus-4-1"
	ClaudeOpus4_5_20251101   = "claude-opus-4-5-20251101"
	ClaudeSonnet4_5_20250929 = "claude-sonnet-4-5-20250929"
	ClaudeHaiku4_5_20251001  = "claude-haiku-4-5-20251001"
	ClaudeOpus4_1_20250805   = "claude-opus-4-1-20250805"
)

func supportsAdaptiveThinking(model string) bool {
	return model == ClaudeOpus4_6 || model == ClaudeSonnet4_6
}

// models is the maintained Anthropic Messages API catalog. It includes current
// aliases and still-supported dated/legacy IDs so callers can pin a model
// version deliberately.
var models = []string{
	ClaudeFable5,
	ClaudeMythos5,
	ClaudeOpus4_8,
	ClaudeSonnet5,
	ClaudeOpus4_7,
	ClaudeOpus4_6,
	ClaudeSonnet4_6,
	ClaudeOpus4_5,
	ClaudeSonnet4_5,
	ClaudeHaiku4_5,
	ClaudeOpus4_1,
	ClaudeOpus4_5_20251101,
	ClaudeSonnet4_5_20250929,
	ClaudeHaiku4_5_20251001,
	ClaudeOpus4_1_20250805,
}
