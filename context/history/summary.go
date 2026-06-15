package history

import (
	"context"
	"strings"

	"github.com/lace-ai/gai/agent/summary"
	gaictx "github.com/lace-ai/gai/context"
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
	ctx, obs := newHistorySummaryObserver(ctx, s.debug, s.sessionID, maxTokens, s.summaryAmount)
	var err error
	defer func() {
		obs.Finish(err)
	}()

	if state == nil {
		err = ErrHistoryStateRequired
		return nil, err
	}
	if s.summarizer == nil {
		err = ErrSummarizerMissing
		return nil, err
	}
	obs.ObserveState(len(state.Turns), state.Summary != nil)
	obs.SetTokenizerID(s.tokenizer.ID())
	if len(state.Turns) == 0 {
		obs.SummarySkippedNoTurns(ctx)
		return state, nil
	}

	var builder strings.Builder

	if state.Summary != nil {
		builder.WriteString(state.Summary.Content.String())
		builder.WriteString("\n")
	}

	summarizedTurnCount := s.summarizedTurnCount(len(state.Turns))
	if summarizedTurnCount == 0 {
		obs.SummarySkippedAmountZero(ctx)
		return state, nil
	}
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
	obs.SummaryGenerated(ctx, nextSummary, summarizedTurnCount, len(state.Turns)-summarizedTurnCount, state.Summary != nil)

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
