package ai_test

import (
	"testing"

	"github.com/lace-ai/gai/ai"
)

func TestAIRequestStoresPromptString(t *testing.T) {
	req := ai.AIRequest{
		Prompt:    "system\n\n<context>...</context>\n\nuser prompt",
		MaxTokens: 42,
	}

	if req.Prompt != "system\n\n<context>...</context>\n\nuser prompt" {
		t.Fatalf("unexpected prompt: %q", req.Prompt)
	}
	if req.MaxTokens != 42 {
		t.Fatalf("unexpected max tokens: %d", req.MaxTokens)
	}
}
