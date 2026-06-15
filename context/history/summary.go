package history

import (
	"context"
	"strings"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/agent/summary"
	gaictx "github.com/lace-ai/gai/context"
	"go.opentelemetry.io/otel/attribute"
)

// Summary is the compact representation of older conversation turns.
type Summary struct {
	ID             string
	StartTurnID    string
	EndTurnID      string
	StartTurnCount int
	EndTurnCount   int
	Content        gaictx.TextContent
	TokenCount     map[string]int
}

func (s *HistorySource) summarizeState(ctx context.Context, state *HistoryState, maxTokens int) (*HistoryState, error) {
	if s == nil {
		return nil, ErrHistorySourceNil
	}
	ctx, span := gai.StartOperationSpan(ctx, contextTracerName, "context.history", "context.operation", "summarize",
		attribute.String("context.session_id", s.sessionID),
		attribute.Int("context.token_budget", maxTokens),
		attribute.Float64("context.history.summary_amount", float64(s.summaryAmount)),
	)
	var err error
	defer func() {
		gai.EndSpan(span, err)
	}()

	if state == nil {
		err = ErrHistoryStateRequired
		return nil, err
	}
	if s.summarizer == nil {
		err = ErrSummarizerMissing
		return nil, err
	}
	span.SetAttributes(
		attribute.Int("context.history.turn_count", len(state.Turns)),
		attribute.Bool("context.history.existing_summary", state.Summary != nil),
	)
	if len(state.Turns) == 0 {
		span.SetAttributes(attribute.String("context.history.summary_skip_reason", "no_turns"))
		s.emit(ctx, "history_source_summary_skipped", map[string]any{
			"session_id":   s.sessionID,
			"token_budget": maxTokens,
			"reason":       "no_turns",
		}, nil)
		return state, nil
	}

	var builder strings.Builder

	if state.Summary != nil {
		builder.WriteString(state.Summary.Content.String())
		builder.WriteString("\n")
	}

	summarizedTurnCount := s.summarizedTurnCount(len(state.Turns))
	if summarizedTurnCount == 0 {
		span.SetAttributes(attribute.String("context.history.summary_skip_reason", "amount_zero"))
		s.emit(ctx, "history_source_summary_skipped", map[string]any{
			"session_id":     s.sessionID,
			"token_budget":   maxTokens,
			"summary_amount": float64(s.summaryAmount),
			"reason":         "amount_zero",
		}, nil)
		return state, nil
	}
	span.SetAttributes(
		attribute.Int("context.history.summarized_turn_count", summarizedTurnCount),
		attribute.Int("context.history.remaining_turn_count", len(state.Turns)-summarizedTurnCount),
	)
	summarizedTurns := state.Turns[:summarizedTurnCount]
	for i := range summarizedTurns {
		writeTurn(&builder, &summarizedTurns[i])
	}

	req := summary.Request{
		ID:        "history",
		Text:      builder.String(),
		MaxTokens: s.summaryMaxTokens,
	}

	res, err := s.summarizer.Summarize(ctx, req)
	if err != nil {
		return nil, err
	}

	firstTurn := summarizedTurns[0]
	lastTurn := summarizedTurns[len(summarizedTurns)-1]
	nextSummary := &Summary{
		StartTurnID:    firstTurn.ID,
		EndTurnID:      lastTurn.ID,
		StartTurnCount: firstTurn.Count,
		EndTurnCount:   lastTurn.Count,
		Content:        gaictx.NewTextContent(res),
		TokenCount:     map[string]int{},
	}
	if state.Summary != nil {
		nextSummary.StartTurnID = state.Summary.StartTurnID
		nextSummary.StartTurnCount = state.Summary.StartTurnCount
	}
	tokenCount, err := s.tokenizer.CountTokens(ctx, nextSummary.Content.String())
	if err != nil {
		return nil, err
	}
	nextSummary.TokenCount[s.tokenizer.ID()] = tokenCount
	span.SetAttributes(attribute.Int("context.history.summary_tokens", tokenCount))
	summaryFields := map[string]any{
		"session_id":             s.sessionID,
		"tokenizer_id":           s.tokenizer.ID(),
		"token_budget":           maxTokens,
		"summary_tokens":         tokenCount,
		"summarized_turn_count":  summarizedTurnCount,
		"remaining_turn_count":   len(state.Turns) - summarizedTurnCount,
		"summary_start_turn":     nextSummary.StartTurnID,
		"summary_end_turn":       nextSummary.EndTurnID,
		"summary_start_count":    nextSummary.StartTurnCount,
		"summary_end_count":      nextSummary.EndTurnCount,
		"previous_summary_found": state.Summary != nil,
	}
	if s.debug != nil && s.debug.IncludeSensitiveData() {
		summaryFields["summary_content"] = nextSummary.Content.String()
	}
	s.emit(ctx, "history_source_summary_generated", summaryFields, nil)

	nextState := &HistoryState{
		Summary: nextSummary,
		Turns:   append([]gaictx.Turn(nil), state.Turns[summarizedTurnCount:]...),
	}
	return nextState, nil
}

func (s *HistorySource) summarizedTurnCount(turnCount int) int {
	if turnCount == 0 || s.summaryAmount <= 0 {
		return 0
	}
	count := int(float32(turnCount) * s.summaryAmount)
	if count == 0 {
		return 1
	}
	if count > turnCount {
		return turnCount
	}
	return count
}

func writeTurn(builder *strings.Builder, turn *gaictx.Turn) {
	if turn.UserMessage != nil {
		builder.WriteString("user: ")
		builder.WriteString(turn.UserMessage.Content.String())
		builder.WriteString("\n")
	}
	for _, message := range turn.Messages {
		builder.WriteString(string(message.Role))
		builder.WriteString(": ")
		builder.WriteString(message.Content.String())
		builder.WriteString("\n")
	}
}
