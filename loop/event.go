package loop

import (
	"context"

	"github.com/lace-ai/gai/ai"
)

// EventType identifies the kind of ordered loop event.
type EventType string

const (
	// EventToken carries a streamed model token.
	EventToken EventType = "token"
	// EventAttemptStart reports that a model generation attempt has started.
	EventAttemptStart EventType = "attempt_start"
	// EventRetry reports that a failed attempt will be retried.
	EventRetry EventType = "retry"
	// EventIterationDone carries a completed agent iteration.
	EventIterationDone EventType = "iteration_done"
	// EventDone reports successful loop completion.
	EventDone EventType = "done"
	// EventError reports terminal loop failure.
	EventError EventType = "error"
	// EventCanceled reports terminal cancellation without treating it as a failure.
	EventCanceled EventType = "canceled"
)

// Event is one item from the loop's ordered execution stream.
type Event struct {
	Type EventType

	IterationCount int
	AttemptID      int
	RetryCount     int
	// TokenCount is the number of stream tokens produced by this attempt.
	TokenCount int
	PartCount  int

	Token     *ai.Token
	Iteration *Iteration
	Err       error
}

func AttemptStartEvent(iteration, attempt, retry int) Event {
	return Event{
		Type:           EventAttemptStart,
		IterationCount: iteration,
		AttemptID:      attempt,
		RetryCount:     retry,
	}
}

func TokenEvent(iteration, attempt, retry int, token ai.Token) Event {
	return Event{
		Type:           EventToken,
		IterationCount: iteration,
		AttemptID:      attempt,
		RetryCount:     retry,
		Token:          &token,
	}
}

func RetryEvent(iteration, attempt, retry, tokenCount int, attemptIteration Iteration) Event {
	return Event{
		Type:           EventRetry,
		IterationCount: iteration,
		AttemptID:      attempt,
		RetryCount:     retry,
		TokenCount:     tokenCount,
		PartCount:      len(attemptIteration.Parts),
		Iteration:      &attemptIteration,
	}
}

func IterationDoneEvent(iteration Iteration, attempt, retry, tokenCount int) Event {
	return Event{
		Type:           EventIterationDone,
		IterationCount: iteration.Count,
		AttemptID:      attempt,
		RetryCount:     retry,
		TokenCount:     tokenCount,
		PartCount:      len(iteration.Parts),
		Iteration:      &iteration,
	}
}

func DoneEvent() Event {
	return Event{Type: EventDone}
}

func ErrorEvent(err error) Event {
	return Event{Type: EventError, Err: err}
}

// CanceledEvent reports terminal cancellation that occurred outside a specific
// model generation attempt.
func CanceledEvent(err error) Event {
	return Event{Type: EventCanceled, Err: err}
}

// AttemptErrorEvent reports a terminal error that occurred inside a specific
// model generation attempt.
func AttemptErrorEvent(iterationCount, attemptID, retryCount, tokenCount int, attemptIteration *Iteration, err error) Event {
	return attemptTerminalEvent(EventError, iterationCount, attemptID, retryCount, tokenCount, attemptIteration, err)
}

// AttemptCanceledEvent reports terminal cancellation that occurred inside a
// specific model generation attempt.
func AttemptCanceledEvent(iterationCount, attemptID, retryCount, tokenCount int, attemptIteration *Iteration, err error) Event {
	return attemptTerminalEvent(EventCanceled, iterationCount, attemptID, retryCount, tokenCount, attemptIteration, err)
}

func attemptTerminalEvent(eventType EventType, iterationCount, attemptID, retryCount, tokenCount int, attemptIteration *Iteration, err error) Event {
	event := Event{
		Type:           eventType,
		IterationCount: iterationCount,
		AttemptID:      attemptID,
		RetryCount:     retryCount,
		TokenCount:     tokenCount,
		Err:            err,
	}
	if attemptIteration != nil {
		iteration := *attemptIteration
		event.PartCount = len(iteration.Parts)
		event.Iteration = &iteration
	}
	return event
}

func sendEvent(ctx context.Context, ch chan<- Event, event Event) error {
	select {
	case ch <- event:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
