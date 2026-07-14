package loop

import (
	"context"
	"testing"
	"time"
)

func TestSendTerminalEventDoesNotBlockWhenCanceledAndBufferIsFull(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	events := make(chan Event, 1)
	events <- DoneEvent()

	sent := make(chan struct{})
	go func() {
		sendTerminalEvent(ctx, events, CanceledEvent(context.Canceled))
		close(sent)
	}()

	select {
	case <-sent:
	case <-time.After(time.Second):
		t.Fatal("cancellation send blocked while the buffer was full")
	}

	if event := <-events; event.Type != EventDone {
		t.Fatalf("unexpected buffered event: %#v", event)
	}
}
