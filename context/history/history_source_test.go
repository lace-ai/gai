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

func TestHistorySourceFunctionTable(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name                  string
		state                 *history.HistoryState
		tokenBudget           int
		summaryDef            *history.SummarizerDefinition
		wantPart              bool
		wantSaved             bool
		wantSavedSummary      bool
		wantSavedTurnIDs      []string
		wantSummaryStartTurn  string
		wantSummaryEndTurn    string
		wantSummaryContent    string
		wantModelCalls        int
		wantDebugEventNames   []string
		wantAbsentDebugEvents []string
	}{
		{
			name:        "missing state returns empty part",
			state:       nil,
			tokenBudget: 10,
			wantPart:    true,
			wantSaved:   false,
		},
		{
			name: "budget trims turns and preserves summary",
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
			tokenBudget:      10,
			wantPart:         true,
			wantSaved:        true,
			wantSavedSummary: true,
			wantSavedTurnIDs: []string{"turn-3"},
		},
		{
			name: "summary enabled but budget fits skips summarizer",
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
			tokenBudget: 100,
			summaryDef: &history.SummarizerDefinition{
				Enabled: true,
				Amount:  0.67,
			},
			wantPart:              true,
			wantSaved:             true,
			wantSavedSummary:      false,
			wantSavedTurnIDs:      []string{"turn-1", "turn-2", "turn-3"},
			wantModelCalls:        0,
			wantAbsentDebugEvents: []string{"history_source_summary_attempted", "history_source_summary_generated"},
		},
		{
			name: "summary enabled and budget reached summarizes oldest turns",
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
			tokenBudget: 5,
			summaryDef: &history.SummarizerDefinition{
				Enabled: true,
				Amount:  0.67,
			},
			wantPart:             true,
			wantSaved:            true,
			wantSavedSummary:     true,
			wantSavedTurnIDs:     []string{"turn-3"},
			wantSummaryStartTurn: "turn-1",
			wantSummaryEndTurn:   "turn-2",
			wantSummaryContent:   "summary text",
			wantModelCalls:       1,
			wantDebugEventNames: []string{
				"history_source_summary_attempted",
				"history_source_summary_generated",
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			store := &historyStore{state: tt.state}
			model := &mocks.MockModel{
				Responses: []mocks.MockModelResponse{
					{Res: ai.AIResponse{Text: "summary text"}},
				},
			}

			var source *history.HistorySource
			var err error
			if tt.summaryDef != nil {
				summaryDef := *tt.summaryDef
				summaryDef.Model = model
				source, err = history.New("session-1", store, &summaryDef)
			} else {
				source = history.NewHistory("session-1", store)
			}
			if err != nil {
				t.Fatalf("New failed: %v", err)
			}

			var events []gai.DebugEvent
			source.DebugSink(gai.DebugSinkFunc(func(ctx context.Context, e gai.DebugEvent) {
				events = append(events, e)
			}), nil)
			source.SetTokenizer(&mocks.MockTokenizer{})

			part, err := source.Function(context.Background(), tt.tokenBudget)
			if err != nil {
				t.Fatalf("Function failed: %v", err)
			}
			if got := part != nil; got != tt.wantPart {
				t.Fatalf("unexpected part presence: want %v got %v", tt.wantPart, got)
			}
			if got := store.saved != nil; got != tt.wantSaved {
				t.Fatalf("unexpected saved state presence: want %v got %v", tt.wantSaved, got)
			}
			if model.Count != tt.wantModelCalls {
				t.Fatalf("unexpected summarizer calls: want %d got %d", tt.wantModelCalls, model.Count)
			}

			if !tt.wantSaved {
				return
			}
			if got := store.saved.Summary != nil; got != tt.wantSavedSummary {
				t.Fatalf("unexpected saved summary presence: want %v got %v", tt.wantSavedSummary, got)
			}
			if tt.wantSavedSummary {
				if tt.wantSummaryStartTurn != "" && store.saved.Summary.StartTurnID != tt.wantSummaryStartTurn {
					t.Fatalf("unexpected summary start turn: want %q got %q", tt.wantSummaryStartTurn, store.saved.Summary.StartTurnID)
				}
				if tt.wantSummaryEndTurn != "" && store.saved.Summary.EndTurnID != tt.wantSummaryEndTurn {
					t.Fatalf("unexpected summary end turn: want %q got %q", tt.wantSummaryEndTurn, store.saved.Summary.EndTurnID)
				}
				if got := store.saved.Summary.Content.String(); tt.wantSummaryContent != "" && got != tt.wantSummaryContent {
					t.Fatalf("unexpected summary content: want %q got %q", tt.wantSummaryContent, got)
				}
			}

			gotTurnIDs := make([]string, 0, len(store.saved.Turns))
			for _, turn := range store.saved.Turns {
				gotTurnIDs = append(gotTurnIDs, turn.ID)
			}
			if len(gotTurnIDs) != len(tt.wantSavedTurnIDs) {
				t.Fatalf("unexpected saved turn count: want %d got %d", len(tt.wantSavedTurnIDs), len(gotTurnIDs))
			}
			for i := range gotTurnIDs {
				if gotTurnIDs[i] != tt.wantSavedTurnIDs[i] {
					t.Fatalf("unexpected saved turn id at %d: want %q got %q", i, tt.wantSavedTurnIDs[i], gotTurnIDs[i])
				}
			}

			for _, name := range tt.wantDebugEventNames {
				if !hasHistoryDebugEvent(events, name) {
					t.Fatalf("expected debug event %q in %+v", name, events)
				}
			}
			for _, name := range tt.wantAbsentDebugEvents {
				if hasHistoryDebugEvent(events, name) {
					t.Fatalf("unexpected debug event %q in %+v", name, events)
				}
			}
		})
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
