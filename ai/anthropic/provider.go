// Package anthropic implements GAI models using Anthropic's Messages API.
package anthropic

import (
	"errors"
	"net/http"
	"strings"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
)

const apiVersion = "2023-06-01"

var ErrInvalidAPIKey = errors.New("invalid API key")

// Provider is an Anthropic Messages API provider.
type Provider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
	debug      gai.DebugSink
}

var _ ai.Provider = (*Provider)(nil)

// New creates an Anthropic provider using apiKey.
func New(apiKey string, debug gai.DebugSink) *Provider {
	return &Provider{apiKey: apiKey, baseURL: "https://api.anthropic.com", httpClient: &http.Client{}, debug: debug}
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

func (p *Provider) Name() string { return "anthropic" }

func (p *Provider) Model(name string) (ai.Model, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	name = strings.TrimSpace(name)
	if name == "" || !isKnownModel(name) {
		return nil, ai.ErrModelNotFound
	}
	return &Model{name: name, client: p, debug: p.debug}, nil
}

func (p *Provider) ListModels() ([]string, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	return append([]string(nil), models...), nil
}

func isKnownModel(name string) bool {
	for _, model := range models {
		if model == name {
			return true
		}
	}
	return false
}
