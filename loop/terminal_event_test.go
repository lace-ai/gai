package loop

import (
	"context"
	"testing"
	"time"
)

func TestSendTerminalEventDeliversCanceledEventWhenBufferIsFull(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	events := make(chan Event, 1)
	events <- DoneEvent()

	entered := make(chan struct{})
	sent := make(chan struct{})
	go func() {
		close(entered)
		sendTerminalEvent(ctx, events, CanceledEvent(context.Canceled))
		close(sent)
	}()

	<-entered
	select {
	case <-sent:
		t.Fatal("canceled event was dropped while the buffer was full")
	case <-time.After(50 * time.Millisecond):
	}

	<-events // Free capacity for the blocked cancellation send.

	select {
	case event := <-events:
		if event.Type != EventCanceled {
			t.Fatalf("expected canceled event, got %#v", event)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for canceled event")
	}

	select {
	case <-sent:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for cancellation send to finish")
	}
}
