package context

import (
	stdcontext "context"
	"fmt"
	"strconv"
	"strings"
)

type RAGQueryFunc func(ctx stdcontext.Context, view PromptView) (string, error)

type RAGSource struct {
	store RAGStore
	limit int
	query RAGQueryFunc
}

type Document struct {
	ID         int
	Content    string
	TokenCount map[string]int
}

func RAG(store RAGStore, limit int, query RAGQueryFunc) Source {
	return &RAGSource{
		store: store,
		limit: limit,
		query: query,
	}
}

func (s *RAGSource) BuildParts(ctx stdcontext.Context, view PromptView, budget SourceBudget) ([]Part, error) {
	if s == nil || s.store == nil {
		return nil, ErrRAGStoreNotFound
	}
	if s.query == nil {
		return nil, fmt.Errorf("%w: rag query function is nil", ErrPromptSource)
	}
	if budget.Tokenizer == nil {
		return nil, ErrTokenizerNotFound
	}

	query, err := s.query(ctx, view)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(query) == "" {
		return nil, nil
	}

	docs, err := s.store.GetRelevantDocuments(query, s.limit)
	if err != nil {
		return nil, err
	}

	limit := budget.ContentLimit()
	if limit == unlimitedTokens {
		limit = budget.RemainingPromptTokens
	}
	if limit < 0 {
		limit = 0
	}

	tokens := 0
	minOverflowTokens := 0
	children := []Part{}
	overflow := []string{}
	for i, doc := range docs {
		var docTokens int
		if count, exist := doc.TokenCount[budget.Tokenizer.ID()]; exist && count >= 0 {
			docTokens = count
		} else {
			docTokens, err = budget.Tokenizer.CountTokens(ctx, doc.Content)
			if err != nil {
				return nil, err
			}
			s.store.UpdateDocumentTokens(doc.ID, budget.Tokenizer.ID(), docTokens)
		}
		if tokens+docTokens > limit {
			if minOverflowTokens == 0 || docTokens < minOverflowTokens {
				minOverflowTokens = docTokens
			}
			overflow = append(overflow, doc.Content)
			continue
		}
		tokens += docTokens
		children = append(children, NewPart("rag-doc-"+strconv.Itoa(i)+"-"+strconv.Itoa(doc.ID), doc.Content, Tokens(docTokens), Meta("document_id", doc.ID)))
	}

	if len(overflow) > 0 && budget.Summarizer != nil && tokens < limit {
		summaryLimit := limit - tokens
		summary, err := budget.Summarizer.Summarize(ctx, SummaryRequest{
			ID:        "rag-overflow",
			Text:      strings.Join(overflow, "\n\n"),
			MaxTokens: summaryLimit,
			Required:  budget.Required,
			Meta: map[string]any{
				"source": "rag",
			},
		})
		if err != nil {
			if budget.Required {
				return nil, err
			}
		} else {
			summaryTokens, err := budget.Tokenizer.CountTokens(ctx, summary)
			if err != nil {
				return nil, err
			}
			if tokens+summaryTokens <= limit {
				tokens += summaryTokens
				children = append(children, NewPart("rag-summary", summary, Tokens(summaryTokens), Meta("summarized", true)))
			} else if budget.Required {
				return nil, promptBudgetError("rag-summary", tokens+summaryTokens, limit)
			}
		}
	}

	if len(children) == 0 {
		if budget.Required && len(docs) > 0 {
			return nil, promptBudgetError("rag", minOverflowTokens, limit)
		}
		return nil, nil
	}

	return []Part{
		NewPartGroup("rag", children, Tokens(tokens)),
	}, nil
}
