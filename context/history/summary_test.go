package history

import (
	"testing"

	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/testutil/mocks"
)

func TestSummaryTokenCountRecountsNegativeCachedValue(t *testing.T) {
	t.Parallel()

	tokenizer := &mocks.MockTokenizer{Count: 5}
	summary := &Summary{
		Content:    gaictx.NewTextContent("older turns"),
		tokenCount: map[string]int{"mock.tokenizer": -1},
	}

	tokens, err := summary.TokenCount(tokenizer)
	if err != nil {
		t.Fatalf("TokenCount failed: %v", err)
	}
	if tokens != 5 {
		t.Fatalf("expected tokenizer to recount invalid cached value, got %d", tokens)
	}
	if tokenizer.CountCalls != 1 {
		t.Fatalf("expected one tokenizer call, got %d", tokenizer.CountCalls)
	}
	if summary.tokenCount["mock.tokenizer"] != 5 {
		t.Fatalf("expected cache to be updated, got %+v", summary.tokenCount)
	}
}
