package context

type Summary struct {
	StartTurnID    string
	EndTurnID      string
	StartTurnCount int
	EndTurnCount   int
	Content        TextContent
	TokenCount     map[string]int
}
