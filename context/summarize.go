package context

type Summary struct {
	StartTurnID    string
	EndTurnID      string
	StartTurnCount int
	EndTurnCount   int
	Content        TextContent
	TokenCount     map[string]int
}

type SummaryRequest struct {
	ID        string
	Text      string
	MaxTokens int
	Required  bool
	Meta      map[string]any
}
