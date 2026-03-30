package ai

import "testing"

func TestAIRequestCombinedPrompt(t *testing.T) {
	req := AIRequest{
		SystemPrompt: "system",
		Prompt:       "user prompt",
		Context:      "<conversation>...</conversation>",
	}

	got := req.CombinedPrompt()
	want := "system\n\nuser prompt\n\n<conversation>...</conversation>"
	if got != want {
		t.Fatalf("unexpected combined prompt:\nwant: %q\ngot:  %q", want, got)
	}
}

func TestAIRequestCombinedPromptSkipsEmptySections(t *testing.T) {
	req := AIRequest{Prompt: "only prompt"}
	got := req.CombinedPrompt()
	if got != "only prompt\n\n" {
		t.Fatalf("unexpected combined prompt for prompt-only request: %q", got)
	}
}
