package history

import (
	"context"
	"testing"

	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/testutil/mocks"
)

func TestPartTokensRecountsNegativeCachedValue(t *testing.T) {
	t.Parallel()

	tokenizer := &mocks.MockTokenizer{Count: 6}
	part := &Part{
		Contents: []Content{
			{Value: gaictx.NewTextContent("hello")},
		},
		TokenCount: map[string]int{"mock.tokenizer": -1},
	}

	tokens, err := part.Tokens(context.Background(), tokenizer)
	if err != nil {
		t.Fatalf("Tokens failed: %v", err)
	}
	if tokens != 6 {
		t.Fatalf("expected tokenizer to recount invalid cached value, got %d", tokens)
	}
	if tokenizer.CountCalls != 1 {
		t.Fatalf("expected one tokenizer call, got %d", tokenizer.CountCalls)
	}
	if part.TokenCount["mock.tokenizer"] != 6 {
		t.Fatalf("expected cache to be updated, got %+v", part.TokenCount)
	}
}
