package mocks

import "agent-backend/gai/ai"

type MockProvider struct {
	ProviderName string
	Models       map[string]ai.Model
}

func (p *MockProvider) Name() string {
	return p.ProviderName
}

func (p *MockProvider) Model(name string) (ai.Model, error) {
	model, exists := p.Models[name]
	if !exists {
		return nil, ai.ErrModelNotFound
	}
	return model, nil
}

func (p *MockProvider) ListModels() ([]string, error) {
	var modelNames []string
	for name := range p.Models {
		modelNames = append(modelNames, name)
	}
	return modelNames, nil
}
