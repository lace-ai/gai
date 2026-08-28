package agent

import (
	"context"
	"errors"
	"testing"
)

func TestNewRunReturnsRandomIDGenerationError(t *testing.T) {
	want := errors.New("random source unavailable")
	previous := readRunID
	readRunID = func([]byte) (int, error) { return 0, want }
	t.Cleanup(func() { readRunID = previous })

	workflow, err := New(Definition{}).NewRun(context.Background(), RunInput{})
	if !errors.Is(err, want) {
		t.Fatalf("NewRun error = %v, want random ID error %v", err, want)
	}
	if workflow != nil {
		t.Fatal("NewRun returned a workflow after random ID generation failed")
	}
}
