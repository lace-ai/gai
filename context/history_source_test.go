package context_test

import (
	"context"
	"testing"

	gaictx "github.com/lace-ai/gai/context"
)

func TestHistoryPartMarshalJoinsContentStrings(t *testing.T) {
	t.Parallel()

	part := &gaictx.HistoryPart{
		Contents: []gaictx.Content{
			gaictx.NewTextContent("hello"),
			gaictx.NewToolCallContent("search", `{"q":"lace"}`),
			gaictx.NewToolResultContent("search", "found docs", false, ""),
		},
	}

	got, err := part.Marshal(context.Background())
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	want := "hello\nsearch({\"q\":\"lace\"})\nsearch result: found docs"
	if string(got) != want {
		t.Fatalf("unexpected history marshal output:\nwant %q\n got %q", want, string(got))
	}
}

func TestHistoryPartMarshalEmpty(t *testing.T) {
	t.Parallel()

	var part *gaictx.HistoryPart
	got, err := part.Marshal(context.Background())
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	if string(got) != "" {
		t.Fatalf("expected empty history output, got %q", string(got))
	}
}
