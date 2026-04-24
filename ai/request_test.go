package ai_test

import (
	"testing"

	"github.com/lace-ai/gai/ai"
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
	want := "system\n\n<conversation>...</conversation>user prompt\n\n"
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
