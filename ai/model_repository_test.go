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

func TestModelRepositoryPrefersContextAwareModelCatalog(t *testing.T) {
	repo := ai.NewModelRepository(nil)
	provider := &catalogProvider{
		MockProvider: mocks.MockProvider{ProviderName: "catalog"},
		descriptors: []ai.ModelDescriptor{{
			Model:           "dynamic",
			NativeMessages:  ai.FeatureSupportSupported,
			ToolChoiceModes: []ai.ToolChoiceMode{ai.ToolChoiceAuto},
		}},
	}
	if err := repo.RegisterProvider(context.Background(), provider); err != nil {
		t.Fatal(err)
	}

	type contextKey string
	ctx := context.WithValue(context.Background(), contextKey("key"), "value")
	descriptors, err := repo.ListModelDescriptors(ctx)
	if err != nil {
		t.Fatalf("ListModelDescriptors: %v", err)
	}
	if provider.ctx != ctx {
		t.Fatal("repository did not propagate the caller context")
	}
	if provider.legacyCalls != 0 {
		t.Fatalf("legacy provider path called %d times", provider.legacyCalls)
	}
	if len(descriptors) != 1 || descriptors[0].Provider != "catalog" || descriptors[0].Model != "dynamic" {
		t.Fatalf("descriptors = %#v", descriptors)
	}

	descriptors[0].ToolChoiceModes[0] = ai.ToolChoiceNone
	again, err := repo.ListModelDescriptors(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if again[0].ToolChoiceModes[0] != ai.ToolChoiceAuto {
		t.Fatalf("catalog result was not copy isolated: %#v", again)
	}
}

type describedModel struct {
	mocks.MockModel
	descriptor ai.ModelDescriptor
}

func (m *describedModel) Descriptor() ai.ModelDescriptor { return m.descriptor }

type catalogProvider struct {
	mocks.MockProvider
	ctx         context.Context
	descriptors []ai.ModelDescriptor
	legacyCalls int
}

func (p *catalogProvider) ListModelDescriptors(ctx context.Context) ([]ai.ModelDescriptor, error) {
	p.ctx = ctx
	out := make([]ai.ModelDescriptor, len(p.descriptors))
	for i := range p.descriptors {
		out[i] = p.descriptors[i].Copy()
	}
	return out, nil
}

func (p *catalogProvider) ListModels() ([]string, error) {
	p.legacyCalls++
	return nil, errors.New("legacy ListModels must not be called")
}

func (p *catalogProvider) Model(string) (ai.Model, error) {
	p.legacyCalls++
	return nil, errors.New("legacy Model must not be called")
}
