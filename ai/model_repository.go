package ai

type ModelRepository struct {
	Providers map[string]Provider
}

func NewModelRepository() *ModelRepository {
	return &ModelRepository{
		Providers: make(map[string]Provider),
	}
}

func (r *ModelRepository) RegisterProvider(provider Provider) error {
	_, exists := r.Providers[provider.Name()]
	if exists {
		return ErrProviderAlreadyExists
	}
	r.Providers[provider.Name()] = provider
	return nil
}

func (r *ModelRepository) UnregisterProvider(providerName string) error {
	_, exists := r.Providers[providerName]
	if !exists {
		return ErrProviderNotFound
	}
	delete(r.Providers, providerName)
	return nil
}

func (r *ModelRepository) GetModel(providerName, modelName string) (Model, error) {
	provider, ok := r.Providers[providerName]
	if !ok {
		return nil, ErrProviderNotFound
	}
	return provider.Model(modelName)
}

func (r *ModelRepository) ListModels() ([]string, error) {
	var models []string
	for _, provider := range r.Providers {
		providerModels, err := provider.ListModels()
		if err != nil {
			return nil, err
		}
		for _, model := range providerModels {
			models = append(models, provider.Name()+":"+model)
		}
	}
	return models, nil
}
