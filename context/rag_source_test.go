package context_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	gaictx "github.com/lace-ai/gai/context"
)

func TestRAGSourceBudgetsDocumentsInsideGroup(t *testing.T) {
	t.Parallel()

	store := &fakeRAGStore{
		docs: []gaictx.Document{
			{ID: 1, Content: "one two"},
			{ID: 2, Content: "three four five"},
		},
	}
	parts, err := gaictx.RAG(store, 2, func(ctx context.Context, view gaictx.PromptView) (string, error) {
		return "query", nil
	}).BuildParts(context.Background(), testPromptView{}, gaictx.SourceBudget{
		Tokenizer:             whitespaceTokenizer{},
		MaxTokens:             4,
		RemainingPromptTokens: 4,
	})
	if err != nil {
		t.Fatalf("BuildParts failed: %v", err)
	}
	if len(parts) != 1 {
		t.Fatalf("expected one grouped RAG part, got %d", len(parts))
	}
	if got := len(parts[0].Children); got != 1 {
		t.Fatalf("expected one fitting document child, got %d", got)
	}
	if parts[0].Children[0].ID != "rag-doc-0-1" {
		t.Fatalf("unexpected child id: %+v", parts[0].Children[0])
	}
}

func TestXMLRendererRendersPartGroupAsSinglePart(t *testing.T) {
	t.Parallel()

	group := gaictx.NewPartGroup("rag", []gaictx.Part{
		gaictx.NewPart("doc-1", "document one"),
		gaictx.NewPart("doc-2", "document two"),
	})
	rendered := gaictx.XMLRenderer{}.Render(gaictx.SectionContext, []gaictx.Part{group})
	if strings.Count(rendered, "<part ") != 1 {
		t.Fatalf("expected one outer part, got %q", rendered)
	}
	assertContainsAll(t, rendered, "rendered group", `<item id="doc-1">`, `<item id="doc-2">`)
}

func TestRAGSourceRequiresRAGStore(t *testing.T) {
	t.Parallel()

	_, err := gaictx.RAG(nil, 1, func(ctx context.Context, view gaictx.PromptView) (string, error) {
		return "query", nil
	}).BuildParts(context.Background(), testPromptView{}, gaictx.SourceBudget{
		Tokenizer:             whitespaceTokenizer{},
		MaxTokens:             4,
		RemainingPromptTokens: 4,
	})
	if !errors.Is(err, gaictx.ErrRAGStoreNotFound) {
		t.Fatalf("expected ErrRAGStoreNotFound, got %v", err)
	}
}

func TestRAGSourceReportsMinimumDocumentTokensWhenRequiredDocsDoNotFit(t *testing.T) {
	t.Parallel()

	store := &fakeRAGStore{
		docs: []gaictx.Document{
			{ID: 1, Content: "one two three"},
			{ID: 2, Content: "four five"},
		},
	}
	_, err := gaictx.RAG(store, 2, func(ctx context.Context, view gaictx.PromptView) (string, error) {
		return "query", nil
	}).BuildParts(context.Background(), testPromptView{}, gaictx.SourceBudget{
		Tokenizer:             whitespaceTokenizer{},
		MaxTokens:             1,
		RemainingPromptTokens: 1,
		Required:              true,
	})
	if !errors.Is(err, gaictx.ErrPromptBudget) {
		t.Fatalf("expected ErrPromptBudget, got %v", err)
	}
	if !strings.Contains(err.Error(), "would use 2 tokens") {
		t.Fatalf("expected minimum document token count in error, got %v", err)
	}
}

type fakeRAGStore struct {
	docs []gaictx.Document
}

func (s *fakeRAGStore) GetRelevantDocuments(ctx context.Context, query string, limit int) ([]gaictx.Document, error) {
	if limit > 0 && limit < len(s.docs) {
		return s.docs[:limit], nil
	}
	return s.docs, nil
}

func (s *fakeRAGStore) AddDocument(ctx context.Context, content string) (int, error) {
	s.docs = append(s.docs, gaictx.Document{ID: len(s.docs) + 1, Content: content})
	return len(s.docs), nil
}

func (s *fakeRAGStore) UpdateDocumentTokens(ctx context.Context, documentID int, tokenizer string, tokens int) error {
	return nil
}
