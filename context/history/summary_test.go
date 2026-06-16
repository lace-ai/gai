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

func TestSummarySetTokenCountStoresAndClearsCache(t *testing.T) {
	t.Parallel()

	summary := &Summary{}
	summary.SetTokenCount("mock.tokenizer", 7)

	if summary.tokenCount["mock.tokenizer"] != 7 {
		t.Fatalf("expected cache to store token count, got %+v", summary.tokenCount)
	}

	summary.SetTokenCount("mock.tokenizer", -1)
	if _, ok := summary.tokenCount["mock.tokenizer"]; ok {
		t.Fatalf("expected negative token count to clear cache entry, got %+v", summary.tokenCount)
	}
}

func TestSummarySetTokenCountsReplacesCache(t *testing.T) {
	t.Parallel()

	summary := &Summary{
		tokenCount: map[string]int{
			"old.tokenizer": 9,
		},
	}

	summary.SetTokenCounts(map[string]int{
		"mock.tokenizer": 7,
		"bad.tokenizer":  -1,
	})

	if _, ok := summary.tokenCount["old.tokenizer"]; ok {
		t.Fatalf("expected old cache entry to be replaced, got %+v", summary.tokenCount)
	}
	if summary.tokenCount["mock.tokenizer"] != 7 {
		t.Fatalf("expected new cache entry to be stored, got %+v", summary.tokenCount)
	}
	if _, ok := summary.tokenCount["bad.tokenizer"]; ok {
		t.Fatalf("expected negative cache entry to be omitted, got %+v", summary.tokenCount)
	}
}
