package mistral

import "errors"

var (
	ErrInvalidAPIKey = errors.New("invalid API key")
	ErrNoChoices     = errors.New("no choices returned")
)
