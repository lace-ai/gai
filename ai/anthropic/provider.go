// Package anthropic implements GAI models using Anthropic's Messages API.
package anthropic

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"time"

	antropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
)

const (
	apiVersion            = "2023-06-01"
	modelDiscoveryTimeout = 10 * time.Second
)

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
	return &Provider{apiKey: strings.TrimSpace(apiKey), baseURL: "https://api.anthropic.com", httpClient: &http.Client{}, debug: debug}
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
	if name == "" {
		return nil, ai.ErrModelNotFound
	}
	return &Model{name: name, client: p, debug: p.debug}, nil
}

func (p *Provider) ListModels() ([]string, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), modelDiscoveryTimeout)
	defer cancel()
	discovered, err := p.listModels(ctx)
	if err == nil {
		return discovered, nil
	}
	return fallbackModels(), nil
}

func (p *Provider) listModels(ctx context.Context) ([]string, error) {
	client := p.sdkClient()
	pager := client.Models.ListAutoPaging(ctx, antropic.ModelListParams{})
	names := make([]string, 0)
	for pager.Next() {
		if name := strings.TrimSpace(pager.Current().ID); name != "" {
			names = append(names, name)
		}
	}
	if err := pager.Err(); err != nil {
		return nil, err
	}
	return names, nil
}

func fallbackModels() []string {
	return append([]string(nil), models...)
}
