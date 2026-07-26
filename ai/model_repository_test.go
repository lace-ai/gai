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

func TestModelRepositoryListModelDescriptors(t *testing.T) {
	repo := ai.NewModelRepository(nil)
	provider := &mocks.MockProvider{ProviderName: "mock", Models: map[string]ai.Model{
		"legacy":    &mocks.MockModel{ModelName: "legacy"},
		"described": &describedModel{MockModel: mocks.MockModel{ModelName: "described"}, descriptor: ai.ModelDescriptor{NativeMessages: ai.FeatureSupportSupported, ToolChoiceModes: []ai.ToolChoiceMode{ai.ToolChoiceAuto}}},
	}}
	if err := repo.RegisterProvider(context.Background(), provider); err != nil {
		t.Fatal(err)
	}

	descriptors, err := repo.ListModelDescriptors(context.Background())
	if err != nil {
		t.Fatalf("ListModelDescriptors: %v", err)
	}
	if len(descriptors) != 1 || descriptors[0].Provider != "mock" || descriptors[0].Model != "described" || descriptors[0].NativeMessages != ai.FeatureSupportSupported {
		t.Fatalf("descriptors = %#v", descriptors)
	}
	descriptors[0].ToolChoiceModes[0] = ai.ToolChoiceNone
	again, err := repo.ListModelDescriptors(context.Background())
	if err != nil || again[0].ToolChoiceModes[0] != ai.ToolChoiceAuto {
		t.Fatalf("aggregation did not return independent descriptor copies: %#v, %v", again, err)
	}
}

type describedModel struct {
	mocks.MockModel
	descriptor ai.ModelDescriptor
}

func (m *describedModel) Descriptor() ai.ModelDescriptor { return m.descriptor }
