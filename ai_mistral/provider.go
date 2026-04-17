package mistral

import (
	"net/http"
	"strings"
	"time"

	"github.com/HecoAI/gai/ai"
)

type Provider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

func New(apiKey string) *Provider {
	return &Provider{
		apiKey:  apiKey,
		baseURL: "https://api.mistral.ai",
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
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
	return "mistral"
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
