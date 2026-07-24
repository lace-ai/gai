package gemini

import (
	"context"
	"net/http"
	"strings"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"google.golang.org/genai"
)

type Provider struct {
	apiKey     string
	httpClient *http.Client
	baseURL    string
	newClient  func(context.Context, *genai.ClientConfig) (*genai.Client, error)
	debug      gai.DebugSink
}

var _ ai.Provider = (*Provider)(nil)

func New(apiKey string, debug gai.DebugSink) *Provider {
	return &Provider{
		apiKey: apiKey,
		debug:  debug,
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
	return "gemini"
}

func (p *Provider) Model(name string) (ai.Model, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	modelName := strings.TrimSpace(name)
	if modelName == "" {
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

	discovered, err := p.listModels(context.Background())
	if err == nil {
		return discovered, nil
	}
	return fallbackModels(), nil
}

func (p *Provider) listModels(ctx context.Context) ([]string, error) {
	client, err := p.getClient(ctx)
	if err != nil {
		return nil, err
	}

	page, err := client.Models.List(ctx, nil)
	if err != nil {
		return nil, err
	}

	var names []string
	for {
		for _, model := range page.Items {
			name := strings.TrimSpace(strings.TrimPrefix(model.Name, "models/"))
			if name != "" {
				names = append(names, name)
			}
		}
		page, err = page.Next(ctx)
		if err == genai.ErrPageDone {
			return names, nil
		}
		if err != nil {
			return nil, err
		}
	}
}

func (p *Provider) getClient(ctx context.Context) (*genai.Client, error) {
	newClient := p.newClient
	if newClient == nil {
		newClient = genai.NewClient
	}
	return newClient(ctx, &genai.ClientConfig{
		APIKey:     p.apiKey,
		HTTPClient: p.httpClient,
		HTTPOptions: genai.HTTPOptions{
			BaseURL: p.baseURL,
		},
	})
}

func fallbackModels() []string {
	out := make([]string, len(models))
	copy(out, models)
	return out
}

func containsModel(models []string, name string) bool {
	for _, modelName := range models {
		if modelName == name {
			return true
		}
	}
	return false
}
