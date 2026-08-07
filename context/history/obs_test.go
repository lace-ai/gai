package history

import (
	"context"
	"testing"
)

func TestHistoryObserverBuildFinishedAcceptsNilPart(t *testing.T) {
	t.Parallel()

	observer := &historyObserver{}
	observer.BuildFinished(context.Background(), nil, 0, 0, 0, 0)

	if observer.contentCount != 0 {
		t.Fatalf("content count = %d, want 0", observer.contentCount)
	}
}
