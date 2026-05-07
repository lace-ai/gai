package ai

type Tokenizer interface {
	Tokenize(text string) []string
	CountTokens(text string) int
}
