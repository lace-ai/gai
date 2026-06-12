package context

type Summary struct {
	StartTurnID    string
	EndTurnID      string
	StartTurnCount int
	EndTurnCount   int
	Content        string
	TokenCount     map[string]int
}
