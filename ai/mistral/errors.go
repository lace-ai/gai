package mistral

import (
	"errors"
	"fmt"
)

var (
	ErrInvalidAPIKey = errors.New("invalid API key")
	ErrNoChoices     = errors.New("no choices returned")
)

// HTTPError reports a failed Mistral HTTP operation without including the
// response body in Error. ResponseBody provides a defensive copy for callers
// that explicitly need provider diagnostics.
type HTTPError struct {
	Operation  string
	StatusCode int
	body       []byte
}

func newHTTPError(operation string, statusCode int, body []byte) *HTTPError {
	return &HTTPError{Operation: operation, StatusCode: statusCode, body: append([]byte(nil), body...)}
}

func (e *HTTPError) Error() string {
	if e == nil {
		return "mistral request failed"
	}
	return fmt.Sprintf("mistral %s failed (status %d)", e.Operation, e.StatusCode)
}

// ResponseBody returns a copy of the bounded provider response body.
func (e *HTTPError) ResponseBody() []byte {
	if e == nil {
		return nil
	}
	return append([]byte(nil), e.body...)
}
