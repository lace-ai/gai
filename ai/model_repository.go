package ai

import (
	"sort"
)

type ModelRepository struct {
	providers map[string]Provider
}

func NewModelRepository() *ModelRepository {
	return &ModelRepository{
		providers: make(map[string]Provider),
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
		return err
	}

	_, exists := r.providers[provider.Name()]
	if exists {
		return ErrProviderAlreadyExists
	}
	r.providers[provider.Name()] = provider
	return nil
}

func (r *ModelRepository) UnregisterProvider(providerName string) error {
	if err := r.Validate(); err != nil {
		return err
	}

	_, exists := r.providers[providerName]
	if !exists {
		return ErrProviderNotFound
	}
	delete(r.providers, providerName)
	return nil
}

func (r *ModelRepository) GetModel(providerName, modelName string) (Model, error) {
	if err := r.Validate(); err != nil {
		return nil, err
	}

	provider, ok := r.providers[providerName]
	if !ok {
		return nil, ErrProviderNotFound
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
			return nil, err
		}
		for _, model := range providerModels {
			models = append(models, provider.Name()+":"+model)
		}
	}
	sort.Strings(models)
	return models, nil
}
