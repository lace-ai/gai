package history_test

import (
	"context"
	"testing"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/context/history"
	"github.com/lace-ai/gai/testutil/mocks"
)

func TestHistoryPartMarshalJoinsContentStrings(t *testing.T) {
	t.Parallel()

	part := &history.HistoryPart{
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

	var part *history.HistoryPart
	got, err := part.Marshal(context.Background())
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	if string(got) != "" {
		t.Fatalf("expected empty history output, got %q", string(got))
	}
}

type historyStore struct {
	state *history.HistoryState
	saved *history.HistoryState
}

func (s *historyStore) GetLastHistoryState(ctx context.Context, sessionID string) (*history.HistoryState, error) {
	return s.state, nil
}

func (s *historyStore) SaveHistoryState(ctx context.Context, sessionID string, state *history.HistoryState) error {
	s.saved = state
	return nil
}

func (s *historyStore) UpdateTurnTokens(ctx context.Context, turnID string, tokenizer string, tokens int) error {
	return nil
}

func TestHistorySourceSavesBuiltState(t *testing.T) {
	t.Parallel()

	store := &historyStore{
		state: &history.HistoryState{
			Summary: &history.Summary{
				ID:             "summary-1",
				StartTurnID:    "turn-1",
				EndTurnID:      "turn-2",
				StartTurnCount: 1,
				EndTurnCount:   2,
				Content:        gaictx.NewTextContent("summary"),
				TokenCount:     map[string]int{"mock.tokenizer": 1},
			},
			Turns: []gaictx.Turn{
				{
					ID:    "turn-3",
					Count: 3,
					UserMessage: &gaictx.Message{
						Content:    gaictx.NewTextContent("hello"),
						TokenCount: map[string]int{"mock.tokenizer": 1},
					},
				},
				{
					ID:    "turn-4",
					Count: 4,
					UserMessage: &gaictx.Message{
						Content:    gaictx.NewTextContent("this message is too long"),
						TokenCount: map[string]int{"mock.tokenizer": 100},
					},
				},
			},
		},
	}

	source := history.NewHistory("session-1", store)
	source.SetTokenizer(&mocks.MockTokenizer{})

	part, err := source.Function(context.Background(), 10)
	if err != nil {
		t.Fatalf("Function failed: %v", err)
	}
	if part == nil {
		t.Fatal("expected history part")
	}
	if store.saved == nil {
		t.Fatal("expected history state to be saved")
	}
	if store.saved.Summary == nil || store.saved.Summary.ID != "summary-1" {
		t.Fatalf("expected summary to be preserved, got %+v", store.saved.Summary)
	}
	if len(store.saved.Turns) != 1 || store.saved.Turns[0].ID != "turn-3" {
		t.Fatalf("expected only the fitting turn to be saved, got %+v", store.saved.Turns)
	}
}

func TestNewHistoryWithSummarizerRequiresModelOrSummarizer(t *testing.T) {
	t.Parallel()

	_, err := history.New("session-1", &historyStore{}, &history.SummarizerDefinition{
		Enabled: true,
	})
	if err != history.ErrSummarizerRequired {
		t.Fatalf("expected ErrSummarizerRequired, got %v", err)
	}
}

func TestHistorySourceDoesNotSummarizeWhenHistoryFitsBudget(t *testing.T) {
	t.Parallel()

	store := &historyStore{
		state: &history.HistoryState{
			Turns: []gaictx.Turn{
				{
					ID:    "turn-1",
					Count: 1,
					UserMessage: &gaictx.Message{
						Content:    gaictx.NewTextContent("first user"),
						TokenCount: map[string]int{"mock.tokenizer": 2},
					},
				},
				{
					ID:    "turn-2",
					Count: 2,
					UserMessage: &gaictx.Message{
						Content:    gaictx.NewTextContent("second user"),
						TokenCount: map[string]int{"mock.tokenizer": 2},
					},
				},
				{
					ID:    "turn-3",
					Count: 3,
					UserMessage: &gaictx.Message{
						Content:    gaictx.NewTextContent("third user"),
						TokenCount: map[string]int{"mock.tokenizer": 2},
					},
				},
			},
		},
	}
	model := &mocks.MockModel{
		Responses: []mocks.MockModelResponse{
			{Res: ai.AIResponse{Text: "summary text"}},
		},
	}
	source, err := history.New("session-1", store, &history.SummarizerDefinition{
		Enabled: true,
		Amount:  0.67,
		Model:   model,
	})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	var events []gai.DebugEvent
	source.DebugSink(gai.DebugSinkFunc(func(ctx context.Context, e gai.DebugEvent) {
		events = append(events, e)
	}), nil)
	source.SetTokenizer(&mocks.MockTokenizer{})

	part, err := source.Function(context.Background(), 100)
	if err != nil {
		t.Fatalf("Function failed: %v", err)
	}
	if part == nil {
		t.Fatal("expected history part")
	}
	if model.Count != 0 {
		t.Fatalf("expected summarizer not to run, got %d calls", model.Count)
	}
	if hasHistoryDebugEvent(events, "history_source_summary_attempted") {
		t.Fatalf("expected no summary attempt event when history fits budget, got %+v", events)
	}
	if store.saved == nil {
		t.Fatal("expected history state to be saved")
	}
	if store.saved.Summary != nil {
		t.Fatalf("expected no summary when history fits budget, got %+v", store.saved.Summary)
	}
	if len(store.saved.Turns) != 3 {
		t.Fatalf("expected all turns to remain unsummarized, got %+v", store.saved.Turns)
	}
}

func TestHistorySourceSummarizesOldestTurnsWhenBudgetReached(t *testing.T) {
	t.Parallel()

	store := &historyStore{
		state: &history.HistoryState{
			Turns: []gaictx.Turn{
				{
					ID:    "turn-1",
					Count: 1,
					UserMessage: &gaictx.Message{
						Content:    gaictx.NewTextContent("first user"),
						TokenCount: map[string]int{"mock.tokenizer": 2},
					},
				},
				{
					ID:    "turn-2",
					Count: 2,
					UserMessage: &gaictx.Message{
						Content:    gaictx.NewTextContent("second user"),
						TokenCount: map[string]int{"mock.tokenizer": 2},
					},
				},
				{
					ID:    "turn-3",
					Count: 3,
					UserMessage: &gaictx.Message{
						Content:    gaictx.NewTextContent("third user"),
						TokenCount: map[string]int{"mock.tokenizer": 2},
					},
				},
			},
		},
	}
	model := &mocks.MockModel{
		Responses: []mocks.MockModelResponse{
			{Res: ai.AIResponse{Text: "summary text"}},
		},
	}
	source, err := history.New("session-1", store, &history.SummarizerDefinition{
		Enabled: true,
		Amount:  0.67,
		Model:   model,
	})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	var events []gai.DebugEvent
	source.DebugSink(gai.DebugSinkFunc(func(ctx context.Context, e gai.DebugEvent) {
		events = append(events, e)
	}), nil)
	source.SetTokenizer(&mocks.MockTokenizer{})

	part, err := source.Function(context.Background(), 5)
	if err != nil {
		t.Fatalf("Function failed: %v", err)
	}
	if part == nil {
		t.Fatal("expected history part")
	}
	if model.Count != 1 {
		t.Fatalf("expected summarizer to run once, got %d calls", model.Count)
	}
	for _, name := range []string{
		"history_source_summary_attempted",
		"history_source_summary_generated",
	} {
		if !hasHistoryDebugEvent(events, name) {
			t.Fatalf("expected debug event %q in %+v", name, events)
		}
	}
	if store.saved == nil {
		t.Fatal("expected summarized state to be saved")
	}
	if store.saved.Summary == nil {
		t.Fatal("expected summary to be saved")
	}
	if got := store.saved.Summary.Content.String(); got != "summary text" {
		t.Fatalf("unexpected summary content: %q", got)
	}
	if store.saved.Summary.StartTurnID != "turn-1" || store.saved.Summary.EndTurnID != "turn-2" {
		t.Fatalf("expected summary to cover first two turns, got %+v", store.saved.Summary)
	}
	if len(store.saved.Turns) != 1 || store.saved.Turns[0].ID != "turn-3" {
		t.Fatalf("expected newest turn to remain unsummarized, got %+v", store.saved.Turns)
	}
}

func hasHistoryDebugEvent(events []gai.DebugEvent, name string) bool {
	for _, event := range events {
		if event.Name == name {
			return true
		}
	}
	return false
}
