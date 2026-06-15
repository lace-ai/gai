package history

import (
	"context"
	"sort"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/agent/summary"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"go.opentelemetry.io/otel/attribute"
)

const contextTracerName = "github.com/lace-ai/gai/context"

// HistoryState is the persisted conversation state consumed by HistorySource.
// Turns contains the unsummarized tail of the conversation; Summary contains
// older turns that have already been compacted.
type HistoryState struct {
	Turns   []gaictx.Turn
	Summary *Summary
}

// HistoryStore loads and saves history state for a session.
// Implementations also store cached per-turn token counts through TurnTokenStore.
type HistoryStore interface {
	GetLastHistoryState(ctx context.Context, sessionID string) (*HistoryState, error)
	SaveHistoryState(ctx context.Context, sessionID string, state *HistoryState) error

	gaictx.TurnTokenStore
}

// HistorySource renders persisted conversation history as prompt context.
type HistorySource struct {
	historyStateStore HistoryStore
	sessionID         string

	debug     gai.DebugSink
	tokenizer ai.Tokenizer

	summarizer       *summary.Summarizer
	summarize        bool
	summaryAmount    float32
	summaryMaxTokens int
}

// SummarizerDefinition configures history summarization.
// When Enabled is true, HistorySource first tries to fit history normally. If
// the token budget is reached, it summarizes the oldest Amount of unsummarized
// turns. Amount is a fraction from 0 to 1 and defaults to 0.7 when left unset.
// Provide either Summarizer or Model when Enabled is true.
type SummarizerDefinition struct {
	Model            ai.Model
	Summarizer       *summary.Summarizer
	Enabled          bool
	SummaryMaxTokens int
	Amount           float32
}

// NewHistory creates a HistorySource without summarization.
func NewHistory(sessionId string, historyStateStore HistoryStore) *HistorySource {
	return &HistorySource{
		historyStateStore: historyStateStore,
		sessionID:         sessionId,
	}
}

// New creates a HistorySource for sessionId.
// Pass nil summaryDef to disable summarization. When summarization is enabled,
// New validates the configuration and builds a default summary agent from Model
// if Summarizer is not provided.
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
		summaryMaxTokens:  config.SummaryMaxTokens,
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
	summaryAttempted := false
	summaryGenerated := false
	tokenCount := 0
	turnCount := 0
	messageCount := 0
	includedTurnCount := 0
	defer func() {
		span.SetAttributes(
			attribute.Bool("context.history.state_present", statePresent),
			attribute.Bool("context.history.summary_included", summaryIncluded),
			attribute.Bool("context.history.summary_configured", s.summarize),
			attribute.Bool("context.history.summary_attempted", summaryAttempted),
			attribute.Bool("context.history.summary_generated", summaryGenerated),
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
	var part Part
	if lastHistoryState == nil {
		s.emit(ctx, "history_source_state_missing", map[string]any{
			"session_id":   s.sessionID,
			"tokenizer_id": tokenizerID,
		}, nil)
	} else {
		lastHistoryState.Turns = sortTurnsByCount(lastHistoryState.Turns)
		statePresent = true
		state := lastHistoryState
		summarized := false
		for {
			part = Part{}
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
				summaryAttempted = true
				s.emit(ctx, "history_source_summary_attempted", map[string]any{
					"session_id":     s.sessionID,
					"tokenizer_id":   tokenizerID,
					"token_budget":   tokenBudget,
					"turn_count":     len(lastHistoryState.Turns),
					"summary_amount": float64(s.summaryAmount),
				}, nil)
				state, err = s.summarizeState(ctx, lastHistoryState, tokenBudget)
				if err != nil {
					s.emit(ctx, "history_source_summary_failed", map[string]any{
						"session_id":   s.sessionID,
						"tokenizer_id": tokenizerID,
						"token_budget": tokenBudget,
					}, err)
					return nil, err
				}
				summaryGenerated = state != lastHistoryState && state.Summary != nil
				summarized = true
				continue
			}
			if budgetReached && !s.summarize {
				s.emit(ctx, "history_source_summary_skipped", map[string]any{
					"session_id":   s.sessionID,
					"tokenizer_id": tokenizerID,
					"reason":       "disabled",
				}, nil)
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
	buildFields := map[string]any{
		"session_id":    s.sessionID,
		"tokenizer_id":  tokenizerID,
		"token_budget":  tokenBudget,
		"total_tokens":  tokenCount,
		"turn_count":    turnCount,
		"message_count": messageCount,
		"content_count": len(part.Contents),
	}
	if s.debug != nil && s.debug.IncludeSensitiveData() {
		if raw, marshalErr := part.Marshal(ctx); marshalErr == nil {
			buildFields["history_content"] = string(raw)
		}
	}
	s.emit(ctx, "history_source_build_finished", buildFields, nil)
	result = &part
	return result, nil
}

func (s *HistorySource) buildPart(ctx context.Context, state *HistoryState, tokenBudget int, tokenizerID string, part *Part, tokenCount, turnCount, messageCount, includedTurnCount *int, summaryIncluded *bool) (*HistoryState, bool, error) {
	builtState := &HistoryState{}
	if state.Summary != nil {
		*summaryIncluded = true
		summaryContent := Content{
			Text: state.Summary.Content.String(),
			Role: "summary",
		}
		part.Contents = append(part.Contents, summaryContent)
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
		var contents []Content
		if turn.UserMessage != nil {
			contents = append(contents, MapMessageToContent(*turn.UserMessage))
			*messageCount++
		}
		for _, message := range turn.Messages {
			contents = append(contents, MapMessageToContent(message))
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

// sortTurnsByCount() Sort turns by Count in ascending order (oldest first)
func sortTurnsByCount(turns []gaictx.Turn) []gaictx.Turn {
	sort.SliceStable(turns, func(i, j int) bool {
		return turns[i].Count < turns[j].Count
	})
	return turns
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
