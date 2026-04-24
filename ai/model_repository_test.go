package ai_test

import (
	"testing"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/testutil/mocks"
)

func TestModelRepository(t *testing.T) {
	repo := ai.NewModelRepository()

	// Test registering a provider
	provider := &mocks.MockProvider{ProviderName: "mock"}
	err := repo.RegisterProvider(provider)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	// Test registering the same provider again
	err = repo.RegisterProvider(provider)
	if err == nil {
		t.Fatalf("expected error when registering duplicate provider, got nil")
	}
}
