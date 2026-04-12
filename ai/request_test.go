package ai_test

import (
	"testing"

	"agent-backend/gai/ai"
)

func TestAIRequestCombinedPrompt(t *testing.T) {
	req := ai.AIRequest{
		Prompt: ai.Prompt{
			System:  "system",
			Prompt:  "user prompt",
			Context: "<conversation>...</conversation>",
		},
	}

	got := req.Prompt.CombinedPrompt()
	want := "system\n\nuser prompt\n\n<conversation>...</conversation>"
	if got != want {
		t.Fatalf("unexpected combined prompt:\nwant: %q\ngot:  %q", want, got)
	}
}

func TestAIRequestCombinedPromptSkipsEmptySections(t *testing.T) {
	req := ai.AIRequest{Prompt: ai.Prompt{
		Prompt: "only prompt",
	}}
	got := req.Prompt.CombinedPrompt()
	if got != "only prompt\n\n" {
		t.Fatalf("unexpected combined prompt for prompt-only request: %q", got)
	}
}
