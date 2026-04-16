package gemini

import (
	"strings"

	"agent-backend/gai/ai"
)

type Provider struct {
	apiKey string
}

func New(apiKey string) *Provider {
	return &Provider{apiKey: apiKey}
}

func (p *Provider) Validate() error {
	if p == nil {
		return ai.ErrNilProvider
	}
	if strings.TrimSpace(p.apiKey) == "" {
		return ErrInvalidAPIKey
	}
	return nil
}

func (p *Provider) Name() string {
	return "gemini"
}

func (p *Provider) Model(name string) (ai.Model, error) {
	if strings.TrimSpace(name) == "" {
		return nil, ai.ErrModelNotFound
	}

	return &Model{
		name:   name,
		client: p,
	}, nil
}

func (p *Provider) ListModels() ([]string, error) {
	return models, nil
}
