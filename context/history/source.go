package history

import (
	"context"
	"sort"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/agent/summary"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

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
	ctx, obs := newHistoryBuildObserver(ctx, s.debug, s.sessionID, tokenBudget, s.summarize)
	defer func() {
		obs.Finish(err)
	}()

	if s.historyStateStore == nil {
		obs.StoreMissing(ctx)
		return nil, gaictx.ErrSessionStoreNotFound
	}
	if s.tokenizer == nil {
		obs.TokenizerMissing(ctx)
		return nil, gaictx.ErrTokenizerNotFound
	}
	tokenizerID := s.tokenizer.ID()
	obs.SetTokenizerID(tokenizerID)
	lastHistoryState, err := s.historyStateStore.GetLastHistoryState(ctx, s.sessionID)
	if err != nil {
		obs.StateLoadFailed(ctx, err)
		return nil, err
	}
	var part Part
	summaryIncluded := false
	budgetReached := false
	tokenCount := 0
	turnCount := 0
	messageCount := 0
	includedTurnCount := 0
	if lastHistoryState == nil {
		obs.StateMissing(ctx)
	} else {
		lastHistoryState.Turns = sortTurnsByCount(lastHistoryState.Turns)
		obs.MarkStatePresent()
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

			builtState, buildBudgetReached, err := s.buildPart(ctx, state, tokenBudget, s.tokenizer, &part, obs, &tokenCount, &turnCount, &messageCount, &includedTurnCount, &summaryIncluded)
			if err != nil {
				return nil, err
			}
			budgetReached = buildBudgetReached

			if budgetReached && s.summarize && !summarized {
				obs.SummaryAttempted(ctx, len(lastHistoryState.Turns))
				state, err = s.summarizeState(ctx, lastHistoryState, tokenBudget)
				if err != nil {
					obs.SummaryFailed(ctx, err)
					return nil, err
				}
				if state != lastHistoryState && state.Summary != nil {
					obs.MarkSummaryGenerated()
				}
				summarized = true
				continue
			}
			if budgetReached && !s.summarize {
				obs.SummarySkippedDisabled(ctx)
			}

			if builtState.Summary != nil || len(builtState.Turns) > 0 {
				if err := s.historyStateStore.SaveHistoryState(ctx, s.sessionID, builtState); err != nil {
					obs.StateSaveFailed(ctx, err)
					return nil, err
				}
				obs.MarkStateSaved()
			}
			break
		}
	}
	part.saveTokens(tokenizerID, tokenCount)
	obs.BuildFinished(ctx, &part, tokenCount, turnCount, includedTurnCount, messageCount)
	result = &part
	return result, nil
}

func (s *HistorySource) buildPart(
	ctx context.Context,
	state *HistoryState,
	tokenBudget int,
	tokenizer ai.Tokenizer,
	part *Part,
	obs *historyObserver,
	tokenCount,
	turnCount,
	messageCount,
	includedTurnCount *int,
	summaryIncluded *bool,
) (*HistoryState, bool, error) {
	builtState := &HistoryState{}
	if state.Summary != nil {
		*summaryIncluded = true
		summaryContent := Content{
			Text:  state.Summary.Content.String(),
			Role:  "summary",
			Value: state.Summary.Content,
		}
		part.Contents = append(part.Contents, summaryContent)
		summaryTokenCount, err := state.Summary.TokenCount(tokenizer)
		if err != nil {
			obs.SummaryTokenCountFailed(ctx, state.Summary, err)
			return nil, false, err
		}
		if *tokenCount+summaryTokenCount > tokenBudget {
			obs.BudgetReached(ctx, *tokenCount, nil)
			builtState.Turns = []gaictx.Turn{}
			return builtState, true, nil
		}
		*tokenCount += summaryTokenCount
		summary := *state.Summary
		builtState.Summary = &summary
		obs.SummaryIncluded(ctx, state.Summary)
	} else {
		obs.SummaryMissing(ctx)
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
			turnCopy := turn
			obs.TurnTokenizeFailed(ctx, &turnCopy, err)
			return nil, false, err
		}
		if *tokenCount+tokens > tokenBudget {
			turnCopy := turn
			obs.BudgetReached(ctx, *tokenCount, &turnCopy)
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
