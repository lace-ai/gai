package context

type Summary struct {
	StartTurnID    int
	EndTurnID      int
	StartTurnCount int
	EndTurnCount   int
	Content        string
	TokenCount     map[string]int
}
