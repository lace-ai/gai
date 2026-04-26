package ai

import (
	"sort"

	"github.com/lace-ai/gai"
)

type ModelRepository struct {
	providers map[string]Provider
	debug     gai.DebugSink
}

func NewModelRepository(debug gai.DebugSink) *ModelRepository {
	return &ModelRepository{
		providers: make(map[string]Provider),
		debug:     debug,
	}
}

func (r *ModelRepository) Validate() error {
	if r == nil {
		return ErrNilModelRepository
	}
	return nil
}

func (r *ModelRepository) RegisterProvider(provider Provider) error {
	if err := r.Validate(); err != nil {
		return err
	}
	if err := provider.Validate(); err != nil {
		if r.debug != nil {
			r.debug.Emit(nil, gai.DebugEvent{
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
			r.debug.Emit(nil, gai.DebugEvent{
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
		r.debug.Emit(nil, gai.DebugEvent{
			Name:   "provider_registered",
			Source: "ai:ModelRepository.RegisterProvider",
			Fields: map[string]any{
				"provider_name": provider.Name(),
			},
		})
	}
	return nil
}

func (r *ModelRepository) UnregisterProvider(providerName string) error {
	if err := r.Validate(); err != nil {
		return err
	}

	_, exists := r.providers[providerName]
	if !exists {
		if r.debug != nil {
			r.debug.Emit(nil, gai.DebugEvent{
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
		r.debug.Emit(nil, gai.DebugEvent{
			Name:   "provider_unregistered",
			Source: "ai:ModelRepository.UnregisterProvider",
			Fields: map[string]any{
				"provider_name": providerName,
			},
		})
	}
	return nil
}

func (r *ModelRepository) GetModel(providerName, modelName string) (Model, error) {
	if err := r.Validate(); err != nil {
		return nil, err
	}

	provider, ok := r.providers[providerName]
	if !ok {
		if r.debug != nil {
			r.debug.Emit(nil, gai.DebugEvent{
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
		r.debug.Emit(nil, gai.DebugEvent{
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

func (r *ModelRepository) ListModels() ([]string, error) {
	if err := r.Validate(); err != nil {
		return nil, err
	}

	var models []string
	for _, provider := range r.providers {
		providerModels, err := provider.ListModels()
		if err != nil {
			if r.debug != nil {
				r.debug.Emit(nil, gai.DebugEvent{
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
		r.debug.Emit(nil, gai.DebugEvent{
			Name:   "models_listed",
			Source: "ai:ModelRepository.ListModels",
			Fields: map[string]any{
				"model_count": len(models),
			},
		})
	}
	return models, nil
}
