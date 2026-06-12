package context

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/lace-ai/gai"
	"go.opentelemetry.io/otel/attribute"
)

type RAGQueryFunc func(ctx context.Context, view PromptView) (string, error)

type RAGSource struct {
	store RAGStore
	limit int
	query RAGQueryFunc
	debug gai.DebugSink
}

type Document struct {
	ID         string
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

func (s *RAGSource) DebugSink(debug gai.DebugSink) {
	s.debug = debug
}

func (s *RAGSource) BuildParts(ctx context.Context, view PromptView, budget SourceBudget) (parts []Part, err error) {
	sourceLimit := 0
	if s != nil {
		sourceLimit = s.limit
	}
	ctx, span := gai.StartOperationSpan(ctx, contextTracerName, "context", "context.operation", "source.rag.build_parts",
		attribute.Int("source.document_limit", sourceLimit),
		attribute.Int("source.max_tokens", budget.MaxTokens),
		attribute.Int("source.remaining_prompt_tokens", budget.RemainingPromptTokens),
		attribute.Bool("source.required", budget.Required),
	)
	queryLength := 0
	documentCount := 0
	includedDocumentCount := 0
	overflowDocumentCount := 0
	totalTokens := 0
	summarizedOverflow := false
	defer func() {
		span.SetAttributes(
			attribute.Int("source.query_length", queryLength),
			attribute.Int("source.document_count", documentCount),
			attribute.Int("source.included_document_count", includedDocumentCount),
			attribute.Int("source.overflow_document_count", overflowDocumentCount),
			attribute.Int("source.tokens", totalTokens),
			attribute.Bool("source.summarized_overflow", summarizedOverflow),
			attribute.Int("source.part_count", len(parts)),
		)
		gai.EndSpan(span, err)
	}()

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
	queryLength = len(query)
	if strings.TrimSpace(query) == "" {
		return nil, nil
	}

	docs, err := s.store.GetRelevantDocuments(ctx, query, s.limit)
	if err != nil {
		return nil, err
	}
	documentCount = len(docs)

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

			go func(docID string, tokens int) {
				detachedCtx := context.WithoutCancel(ctx)
				innerCtx, cancel := context.WithTimeout(detachedCtx, 5*time.Second)
				defer cancel()
				err := s.store.UpdateDocumentTokens(innerCtx, docID, budget.Tokenizer.ID(), tokens)
				if err != nil {
					if s.debug != nil {
						s.debug.Emit(innerCtx, gai.DebugEvent{
							Name:   "Rag-Source",
							Source: "token_count_update_error",
							Fields: map[string]any{
								"doc.id":       docID,
								"tokenizer_id": budget.Tokenizer.ID(),
								"error":        err.Error(),
							},
							Err: err,
						})
					}
				}
			}(doc.ID, docTokens)
		}
		if tokens+docTokens > limit {
			if minOverflowTokens == 0 || docTokens < minOverflowTokens {
				minOverflowTokens = docTokens
			}
			overflow = append(overflow, doc.Content)
			overflowDocumentCount++
			continue
		}
		tokens += docTokens
		totalTokens = tokens
		includedDocumentCount++
		children = append(children, NewPart("rag-doc-"+strconv.Itoa(i)+"-"+doc.ID, doc.Content, Tokens(docTokens), Meta("document_id", doc.ID)))
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
				totalTokens = tokens
				summarizedOverflow = true
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

	parts = []Part{
		NewPartGroup("rag", children, Tokens(tokens)),
	}
	return parts, nil
}
