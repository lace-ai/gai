package ai

import (
	"context"
	"sort"

	"github.com/lace-ai/gai"
)

// ModelRepository stores providers and resolves their models.
//
// A repository is safe to use only after construction with
// NewModelRepository. It is not safe for concurrent mutation.
type ModelRepository struct {
	providers map[string]Provider
	debug     gai.DebugSink
}

// NewModelRepository creates an empty provider registry.
//
// When debug is non-nil, repository operations emit diagnostic events.
func NewModelRepository(debug gai.DebugSink) *ModelRepository {
	return &ModelRepository{
		providers: make(map[string]Provider),
		debug:     debug,
	}
}

// Validate checks whether the repository can be used.
func (r *ModelRepository) Validate() error {
	if r == nil {
		return ErrNilModelRepository
	}
	return nil
}

// RegisterProvider validates and registers provider under Provider.Name.
// It returns ErrProviderAlreadyExists when that name is already registered.
func (r *ModelRepository) RegisterProvider(ctx context.Context, provider Provider) error {
	if err := r.Validate(); err != nil {
		return err
	}
	if err := provider.Validate(); err != nil {
		if r.debug != nil {
			r.debug.Emit(ctx, gai.DebugEvent{
				Name:   "provider_validation_failed",
				Source: "ai:ModelRepository.RegisterProvider",
				Fields: map[string]any{
					"provider_name": provider.Name(),
					"error":         err.Error(),
				},
				Err: err,
			})
		}
		return err
	}

	_, exists := r.providers[provider.Name()]
	if exists {
		if r.debug != nil {
			r.debug.Emit(ctx, gai.DebugEvent{
				Name:   "provider_already_registered",
				Source: "ai:ModelRepository.RegisterProvider",
				Fields: map[string]any{
					"provider_name": provider.Name(),
				},
			})
		}
		return ErrProviderAlreadyExists
	}
	r.providers[provider.Name()] = provider
	if r.debug != nil {
		r.debug.Emit(ctx, gai.DebugEvent{
			Name:   "provider_registered",
			Source: "ai:ModelRepository.RegisterProvider",
			Fields: map[string]any{
				"provider_name": provider.Name(),
			},
		})
	}
	return nil
}

// UnregisterProvider removes the named provider.
// It returns ErrProviderNotFound when no such provider is registered.
func (r *ModelRepository) UnregisterProvider(ctx context.Context, providerName string) error {
	if err := r.Validate(); err != nil {
		return err
	}

	_, exists := r.providers[providerName]
	if !exists {
		if r.debug != nil {
			r.debug.Emit(ctx, gai.DebugEvent{
				Name:   "provider_not_found_for_unregister",
				Source: "ai:ModelRepository.UnregisterProvider",
				Fields: map[string]any{
					"provider_name": providerName,
				},
			})
		}
		return ErrProviderNotFound
	}
	delete(r.providers, providerName)
	if r.debug != nil {
		r.debug.Emit(ctx, gai.DebugEvent{
			Name:   "provider_unregistered",
			Source: "ai:ModelRepository.UnregisterProvider",
			Fields: map[string]any{
				"provider_name": providerName,
			},
		})
	}
	return nil
}

// GetModel resolves modelName through the named provider.
func (r *ModelRepository) GetModel(ctx context.Context, providerName, modelName string) (Model, error) {
	if err := r.Validate(); err != nil {
		return nil, err
	}

	provider, ok := r.providers[providerName]
	if !ok {
		if r.debug != nil {
			r.debug.Emit(ctx, gai.DebugEvent{
				Name:   "provider_not_found_for_model",
				Source: "ai:ModelRepository.GetModel",
				Fields: map[string]any{
					"provider_name": providerName,
					"model_name":    modelName,
				},
			})
		}
		return nil, ErrProviderNotFound
	}
	if r.debug != nil {
		r.debug.Emit(ctx, gai.DebugEvent{
			Name:   "getting_model",
			Source: "ai:ModelRepository.GetModel",
			Fields: map[string]any{
				"provider_name": providerName,
				"model_name":    modelName,
			},
		})
	}
	return provider.Model(modelName)
}

// ListModels returns all registered models as sorted "provider:model" names.
func (r *ModelRepository) ListModels(ctx context.Context) ([]string, error) {
	if err := r.Validate(); err != nil {
		return nil, err
	}

	var models []string
	for _, provider := range r.providers {
		providerModels, err := provider.ListModels()
		if err != nil {
			if r.debug != nil {
				r.debug.Emit(ctx, gai.DebugEvent{
					Name:   "list_provider_models_failed",
					Source: "ai:ModelRepository.ListModels",
					Fields: map[string]any{
						"provider_name": provider.Name(),
						"error":         err.Error(),
					},
					Err: err,
				})
			}
			return nil, err
		}
		for _, model := range providerModels {
			models = append(models, provider.Name()+":"+model)
		}
	}
	sort.Strings(models)
	if r.debug != nil {
		r.debug.Emit(ctx, gai.DebugEvent{
			Name:   "models_listed",
			Source: "ai:ModelRepository.ListModels",
			Fields: map[string]any{
				"model_count": len(models),
			},
		})
	}
	return models, nil
}
