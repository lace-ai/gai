package context

import "testing"

func TestRenderObserverUsesSharedOperation(t *testing.T) {
	t.Parallel()

	observer := newRenderObserver("xml", nil, 0)
	if observer.operation == nil {
		t.Fatal("render observer must use a shared operation")
	}
}
