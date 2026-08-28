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
	debug     gai.ObservationSink
}

// NewModelRepository creates an empty provider registry.
//
// When debug is non-nil, repository operations emit diagnostic events.
func NewModelRepository(debug gai.ObservationSink) *ModelRepository {
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
	if provider == nil {
		return ErrNilProvider
	}
	if err := provider.Validate(); err != nil {
		if r.debug != nil {
			gai.EmitObservation(ctx, r.debug, gai.Observation{
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
			gai.EmitObservation(ctx, r.debug, gai.Observation{
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
		gai.EmitObservation(ctx, r.debug, gai.Observation{
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
			gai.EmitObservation(ctx, r.debug, gai.Observation{
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
		gai.EmitObservation(ctx, r.debug, gai.Observation{
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
			gai.EmitObservation(ctx, r.debug, gai.Observation{
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
		gai.EmitObservation(ctx, r.debug, gai.Observation{
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
				gai.EmitObservation(ctx, r.debug, gai.Observation{
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
		gai.EmitObservation(ctx, r.debug, gai.Observation{
			Name:   "models_listed",
			Source: "ai:ModelRepository.ListModels",
			Fields: map[string]any{
				"model_count": len(models),
			},
		})
	}
	return models, nil
}

// ListModelDescriptors returns descriptors for registered models that expose
// the optional ModelDescriber interface. Models without descriptors are
// skipped, preserving compatibility with existing Model implementations.
func (r *ModelRepository) ListModelDescriptors(ctx context.Context) ([]ModelDescriptor, error) {
	if err := r.Validate(); err != nil {
		return nil, err
	}

	var descriptors []ModelDescriptor
	for _, provider := range r.providers {
		if catalog, ok := provider.(ModelCatalogProvider); ok {
			providerDescriptors, err := catalog.ListModelDescriptors(ctx)
			if err != nil {
				return nil, err
			}
			for _, providerDescriptor := range providerDescriptors {
				descriptor := providerDescriptor.Copy()
				if descriptor.Model == "" {
					continue
				}
				descriptor.Provider = provider.Name()
				descriptors = append(descriptors, descriptor)
			}
			continue
		}
		models, err := provider.ListModels()
		if err != nil {
			return nil, err
		}
		for _, name := range models {
			model, err := provider.Model(name)
			if err != nil {
				return nil, err
			}
			describer, ok := model.(ModelDescriber)
			if !ok {
				continue
			}
			descriptor := describer.Descriptor().Copy()
			descriptor.Provider = provider.Name()
			if descriptor.Model == "" {
				descriptor.Model = name
			}
			descriptors = append(descriptors, descriptor)
		}
	}
	sort.Slice(descriptors, func(i, j int) bool {
		if descriptors[i].Provider == descriptors[j].Provider {
			return descriptors[i].Model < descriptors[j].Model
		}
		return descriptors[i].Provider < descriptors[j].Provider
	})
	return descriptors, nil
}
