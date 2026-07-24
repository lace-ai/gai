package anthropic

import "fmt"

// Error is an error returned by the Anthropic API.
type Error struct {
	StatusCode int
	Type       string
	Message    string
}

func (e *Error) Error() string {
	if e == nil {
		return "<nil>"
	}
	if e.Type != "" {
		return fmt.Sprintf("anthropic API error (status %d, %s): %s", e.StatusCode, e.Type, e.Message)
	}
	return fmt.Sprintf("anthropic API error (status %d): %s", e.StatusCode, e.Message)
}
