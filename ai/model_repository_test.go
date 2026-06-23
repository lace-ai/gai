package ai_test

import (
	"context"
	"errors"
	"testing"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/testutil/mocks"
)

func TestModelRepository(t *testing.T) {
	repo := ai.NewModelRepository(nil)

	// Test registering a provider
	provider := &mocks.MockProvider{ProviderName: "mock"}
	err := repo.RegisterProvider(context.Background(), provider)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	// Test registering the same provider again
	err = repo.RegisterProvider(context.Background(), provider)
	if err == nil {
		t.Fatalf("expected error when registering duplicate provider, got nil")
	}
}

func TestModelRepositoryRejectsNilProvider(t *testing.T) {
	repo := ai.NewModelRepository(nil)

	err := repo.RegisterProvider(context.Background(), nil)
	if !errors.Is(err, ai.ErrNilProvider) {
		t.Fatalf("expected ErrNilProvider, got %v", err)
	}
}
