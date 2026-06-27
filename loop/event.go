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
)

// Event is one item from the loop's ordered execution stream.
type Event struct {
	Type EventType

	IterationCount int
	AttemptID      int
	RetryCount     int
	PartCount      int

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

func RetryEvent(iteration, attempt, retry int, attemptIteration Iteration) Event {
	return Event{
		Type:           EventRetry,
		IterationCount: iteration,
		AttemptID:      attempt,
		RetryCount:     retry,
		PartCount:      len(attemptIteration.Parts),
		Iteration:      &attemptIteration,
	}
}

func IterationDoneEvent(iteration Iteration, attempt, retry int) Event {
	return Event{
		Type:           EventIterationDone,
		IterationCount: iteration.Count,
		AttemptID:      attempt,
		RetryCount:     retry,
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

func sendEvent(ctx context.Context, ch chan<- Event, event Event) error {
	select {
	case ch <- event:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
