package openai

const (
	GPT56      = "gpt-5.6"
	GPT56Terra = "gpt-5.6-terra"
	GPT56Sol   = "gpt-5.6-sol"
	GPT56Luna  = "gpt-5.6-luna"
	GPT41      = "gpt-4.1"
	GPT41Mini  = "gpt-4.1-mini"
	GPT41Nano  = "gpt-4.1-nano"
	GPT4o      = "gpt-4o"
	GPT4oMini  = "gpt-4o-mini"
	O3         = "o3"
	O3Mini     = "o3-mini"
	O4Mini     = "o4-mini"
)

// models contains text-capable models supported by the chat-completions adapter.
// Image-only and automatic model-selection IDs are deliberately excluded.
var models = []string{
	GPT56, GPT56Terra, GPT56Sol, GPT56Luna,
	GPT41, GPT41Mini, GPT41Nano, GPT4o, GPT4oMini, O3, O3Mini, O4Mini,
}
