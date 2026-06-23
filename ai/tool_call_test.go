package ai_test

import (
	"testing"

	"github.com/lace-ai/gai/ai"
)

func TestToolCallStringNilReceiver(t *testing.T) {
	var tc *ai.ToolCall

	if got := tc.String(); got != "<nil>" {
		t.Fatalf("expected <nil>, got %q", got)
	}
}
