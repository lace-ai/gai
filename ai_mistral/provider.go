package mistral

import (
	"net/http"
	"strings"
	"time"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
)

type Provider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
	debug      gai.DebugSink
}

var _ ai.Provider = (*Provider)(nil)

func New(apiKey string, debug gai.DebugSink) *Provider {
	return &Provider{
		apiKey:  apiKey,
		baseURL: "https://api.mistral.ai",
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
		debug: debug,
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
	if err := p.Validate(); err != nil {
		return nil, err
	}

	modelName := strings.TrimSpace(name)
	if modelName == "" || !isKnownModel(modelName) {
		return nil, ai.ErrModelNotFound
	}

	return &Model{
		name:   modelName,
		client: p,
		debug:  p.debug,
	}, nil
}

func (p *Provider) ListModels() ([]string, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	out := make([]string, len(models))
	copy(out, models)
	return out, nil
}

func isKnownModel(name string) bool {
	for _, modelName := range models {
		if modelName == name {
			return true
		}
	}
	return false
}
