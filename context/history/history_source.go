package history

import (
	"context"
	"fmt"
	"strings"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/agent/summary"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"go.opentelemetry.io/otel/attribute"
)

const contextTracerName = "github.com/lace-ai/gai/context"

type Summary struct {
	ID             string
	StartTurnID    string
	EndTurnID      string
	StartTurnCount int
	EndTurnCount   int
	Content        gaictx.TextContent
	TokenCount     map[string]int
}

type HistoryState struct {
	Turns   []gaictx.Turn
	Summary *Summary
}

type HistoryStore interface {
	GetLastHistoryState(ctx context.Context, sessionID string) (*HistoryState, error)
	SaveHistoryState(ctx context.Context, sessionID string, state *HistoryState) error

	gaictx.TurnTokenStore
}

type HistorySource struct {
	historyStateStore HistoryStore
	sessionID         string

	debug     gai.DebugSink
	tokenizer ai.Tokenizer

	summarizer    *summary.Summarizer
	summarize     bool
	summaryAmount float32
}

// SummarizerDefinition defines the configuration for the summarization of the history.
// If enabled is true, the history source will attempt to summarize the history when building the context.
// The amount defines the percentage (0 - 1) of the turns that get summarized,
// starting from the oldest turn.
//
// If summarization is enabled, a summarizer or model must be provided.
type SummarizerDefinition struct {
	Model      ai.Model
	Summarizer *summary.Summarizer
	Enabled    bool
	Amount     float32
}

// SummarizerDefinitoin is kept for compatibility with the original misspelled name.
type SummarizerDefinitoin = SummarizerDefinition

func NewHistory(sessionId string, historyStateStore HistoryStore) *HistorySource {
	return &HistorySource{
		historyStateStore: historyStateStore,
		sessionID:         sessionId,
	}
}

func New(sessionId string, historyStateStore HistoryStore, summaryDef *SummarizerDefinition) (*HistorySource, error) {
	if summaryDef == nil {
		return NewHistory(sessionId, historyStateStore), nil
	}
	config := *summaryDef
	if config.Amount < 0 || config.Amount > 1 {
		return nil, ErrInvalidSummaryAmount
	}
	if config.Amount == 0 {
		config.Amount = 0.7
	}
	if config.Enabled && config.Summarizer == nil {
		if config.Model != nil {
			summarizer := summary.New(config.Model)
			config.Summarizer = &summarizer
		} else {
			return nil, ErrSummarizerRequired
		}
	}

	return &HistorySource{
		historyStateStore: historyStateStore,
		sessionID:         sessionId,
		summarizer:        config.Summarizer,
		summarize:         config.Enabled,
		summaryAmount:     config.Amount,
	}, nil
}

func (p *HistorySource) Name() string {
	return "history"
}

func (s *HistorySource) SetTokenizer(tokenizer ai.Tokenizer) {
	s.tokenizer = tokenizer
}

func (s *HistorySource) DebugSink(debug gai.DebugSink, conv gaictx.Conversation) {
	s.debug = debug
}

type HistoryPart struct {
	Contents   []gaictx.Content
	TokenCount map[string]int
}

func (p *HistoryPart) Name() string {
	return "history"
}

func (p *HistoryPart) Marshal(ctx context.Context) ([]byte, error) {
	if p == nil || len(p.Contents) == 0 {
		return []byte{}, nil
	}

	var builder strings.Builder
	for i, content := range p.Contents {
		if i > 0 {
			builder.WriteString("\n")
		}
		builder.WriteString(content.String())
	}
	return []byte(builder.String()), nil
}

func (p *HistoryPart) Tokens(ctx context.Context, tokenizer ai.Tokenizer) (int, error) {
	if tokenizer == nil {
		return 0, gaictx.ErrTokenizerNotFound
	}
	tokenizerID := tokenizer.ID()
	if count, ok := p.TokenCount[tokenizerID]; ok {
		return count, nil
	}

	count := 0
	for _, content := range p.Contents {
		tokens, err := tokenizer.CountTokens(ctx, content.String())
		if err != nil {
			return 0, err
		}
		count += tokens
	}
	p.saveTokens(tokenizerID, count)
	return count, nil
}

func (p *HistoryPart) saveTokens(tokenizerID string, tokens int) {
	if p.TokenCount == nil {
		p.TokenCount = make(map[string]int)
	}
	p.TokenCount[tokenizerID] = tokens
}

func (s *HistorySource) Function(ctx context.Context, tokenBudget int) (result gaictx.Part, err error) {
	ctx, span := gai.StartOperationSpan(ctx, contextTracerName, "context.history", "context.operation", "build",
		attribute.String("context.source", s.Name()),
		attribute.String("context.session_id", s.sessionID),
		attribute.Int("context.token_budget", tokenBudget),
	)
	statePresent := false
	summaryIncluded := false
	budgetReached := false
	stateSaved := false
	tokenCount := 0
	turnCount := 0
	messageCount := 0
	includedTurnCount := 0
	defer func() {
		span.SetAttributes(
			attribute.Bool("context.history.state_present", statePresent),
			attribute.Bool("context.history.summary_included", summaryIncluded),
			attribute.Bool("context.history.budget_reached", budgetReached),
			attribute.Bool("context.history.state_saved", stateSaved),
			attribute.Int("context.history.total_tokens", tokenCount),
			attribute.Int("context.history.turn_count", turnCount),
			attribute.Int("context.history.included_turn_count", includedTurnCount),
			attribute.Int("context.history.message_count", messageCount),
		)
		gai.EndSpan(span, err)
	}()

	if s.historyStateStore == nil {
		s.emit(ctx, "history_source_store_missing", map[string]any{
			"session_id": s.sessionID,
		}, gaictx.ErrSessionStoreNotFound)
		return nil, gaictx.ErrSessionStoreNotFound
	}
	if s.tokenizer == nil {
		s.emit(ctx, "history_source_tokenizer_missing", map[string]any{
			"session_id": s.sessionID,
		}, gaictx.ErrTokenizerNotFound)
		return nil, gaictx.ErrTokenizerNotFound
	}
	tokenizerID := s.tokenizer.ID()
	span.SetAttributes(attribute.String("context.tokenizer_id", tokenizerID))
	lastHistoryState, err := s.historyStateStore.GetLastHistoryState(ctx, s.sessionID)
	if err != nil {
		s.emit(ctx, "history_source_state_load_failed", map[string]any{
			"session_id":   s.sessionID,
			"tokenizer_id": tokenizerID,
		}, err)
		return nil, err
	}
	var part HistoryPart
	if lastHistoryState == nil {
		s.emit(ctx, "history_source_state_missing", map[string]any{
			"session_id":   s.sessionID,
			"tokenizer_id": tokenizerID,
		}, nil)
	} else {
		statePresent = true
		state := lastHistoryState
		summarized := false
		for {
			part = HistoryPart{}
			summaryIncluded = false
			budgetReached = false
			tokenCount = 0
			turnCount = 0
			messageCount = 0
			includedTurnCount = 0

			builtState, buildBudgetReached, err := s.buildPart(ctx, state, tokenBudget, tokenizerID, &part, &tokenCount, &turnCount, &messageCount, &includedTurnCount, &summaryIncluded)
			if err != nil {
				return nil, err
			}
			budgetReached = buildBudgetReached

			if budgetReached && s.summarize && !summarized {
				state, err = s.summarizeState(ctx, lastHistoryState, tokenBudget)
				if err != nil {
					return nil, err
				}
				summarized = true
				continue
			}

			if builtState.Summary != nil || len(builtState.Turns) > 0 {
				if err := s.historyStateStore.SaveHistoryState(ctx, s.sessionID, builtState); err != nil {
					s.emit(ctx, "history_source_state_save_failed", map[string]any{
						"session_id":   s.sessionID,
						"tokenizer_id": tokenizerID,
					}, err)
					return nil, err
				}
				stateSaved = true
			}
			break
		}
	}
	part.saveTokens(tokenizerID, tokenCount)
	s.emit(ctx, "history_source_build_finished", map[string]any{
		"session_id":    s.sessionID,
		"tokenizer_id":  tokenizerID,
		"token_budget":  tokenBudget,
		"total_tokens":  tokenCount,
		"turn_count":    turnCount,
		"message_count": messageCount,
		"content_count": len(part.Contents),
	}, nil)
	result = &part
	return result, nil
}

func (s *HistorySource) buildPart(ctx context.Context, state *HistoryState, tokenBudget int, tokenizerID string, part *HistoryPart, tokenCount, turnCount, messageCount, includedTurnCount *int, summaryIncluded *bool) (*HistoryState, bool, error) {
	builtState := &HistoryState{}
	if state.Summary != nil {
		*summaryIncluded = true
		part.Contents = append(part.Contents, state.Summary.Content)
		*tokenCount += state.Summary.TokenCount[tokenizerID]
		summary := *state.Summary
		builtState.Summary = &summary
		summaryFields := map[string]any{
			"session_id":          s.sessionID,
			"tokenizer_id":        tokenizerID,
			"summary_tokens":      state.Summary.TokenCount[tokenizerID],
			"summary_start_turn":  state.Summary.StartTurnID,
			"summary_end_turn":    state.Summary.EndTurnID,
			"summary_start_count": state.Summary.StartTurnCount,
			"summary_end_count":   state.Summary.EndTurnCount,
		}
		if s.debug != nil && s.debug.IncludeSensitiveData() {
			summaryFields["summary_content"] = state.Summary.Content.String()
		}
		s.emit(ctx, "history_source_summary_included", summaryFields, nil)
	} else {
		s.emit(ctx, "history_source_summary_missing", map[string]any{
			"session_id":   s.sessionID,
			"tokenizer_id": tokenizerID,
		}, nil)
	}

	includedTurns := make([]gaictx.Turn, 0, len(state.Turns))
	for _, turn := range state.Turns {
		*turnCount++
		var contents []gaictx.Content
		if turn.UserMessage != nil {
			contents = append(contents, turn.UserMessage.Content)
			*messageCount++
		}
		for _, message := range turn.Messages {
			contents = append(contents, message.Content)
			*messageCount++
		}
		tokens, err := turn.Tokenize(ctx, s.tokenizer, s.historyStateStore)
		if err != nil {
			s.emit(ctx, "history_source_turn_tokenize_failed", map[string]any{
				"session_id":   s.sessionID,
				"tokenizer_id": tokenizerID,
				"turn_id":      turn.ID,
				"turn_count":   turn.Count,
			}, err)
			return nil, false, err
		}
		if *tokenCount+tokens > tokenBudget {
			s.emit(ctx, "history_source_token_budget_reached", map[string]any{
				"session_id":    s.sessionID,
				"tokenizer_id":  tokenizerID,
				"token_budget":  tokenBudget,
				"total_tokens":  *tokenCount,
				"last_turn_id":  turn.ID,
				"last_turn_cnt": turn.Count,
			}, nil)
			builtState.Turns = includedTurns
			return builtState, true, nil
		}
		*tokenCount += tokens
		part.Contents = append(part.Contents, contents...)
		includedTurns = append(includedTurns, turn)
		*includedTurnCount++
	}

	builtState.Turns = includedTurns
	return builtState, false, nil
}

func (s *HistorySource) summarizeState(ctx context.Context, state *HistoryState, maxTokens int) (*HistoryState, error) {
	if state == nil {
		return nil, fmt.Errorf("history state is nil")
	}
	if s == nil {
		return nil, fmt.Errorf("history source is nil")
	}
	if s.summarizer == nil {
		return nil, fmt.Errorf("summarizer not configured for history source")
	}
	if len(state.Turns) == 0 {
		return state, nil
	}

	var builder strings.Builder

	if state.Summary != nil {
		builder.WriteString(state.Summary.Content.String())
		builder.WriteString("\n")
	}

	summarizedTurnCount := s.summarizedTurnCount(len(state.Turns))
	if summarizedTurnCount == 0 {
		return state, nil
	}
	summarizedTurns := state.Turns[:summarizedTurnCount]
	for i := range summarizedTurns {
		writeTurn(&builder, &summarizedTurns[i])
	}

	req := summary.Request{
		ID:        "history",
		Text:      builder.String(),
		MaxTokens: maxTokens,
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

func (s *HistorySource) emit(ctx context.Context, name string, fields map[string]any, err error) {
	if s == nil || s.debug == nil {
		return
	}
	s.debug.Emit(ctx, gai.DebugEvent{
		Name:   name,
		Source: "context:HistorySource",
		Fields: fields,
		Err:    err,
	})
}
