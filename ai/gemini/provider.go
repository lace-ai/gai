package gemini

import (
	"context"
	"net/http"
	"strings"
	"time"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"google.golang.org/genai"
)

const modelDiscoveryTimeout = 10 * time.Second

type Provider struct {
	apiKey     string
	httpClient *http.Client
	baseURL    string
	newClient  func(context.Context, *genai.ClientConfig) (*genai.Client, error)
	debug      gai.DebugSink
	catalog    ai.ModelCatalogCache
	catalogMu  ai.ContextMutex
}

var _ ai.Provider = (*Provider)(nil)
var _ ai.ModelCatalogProvider = (*Provider)(nil)

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
	descriptors, err := p.ListModelDescriptors(context.Background())
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(descriptors))
	for _, descriptor := range descriptors {
		names = append(names, descriptor.Model)
	}
	return names, nil
}

// ListModelDescriptors returns a cached point-in-time model catalog.
func (p *Provider) ListModelDescriptors(ctx context.Context) ([]ai.ModelDescriptor, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if err := p.catalogMu.Lock(ctx); err != nil {
		return nil, err
	}
	defer p.catalogMu.Unlock()
	if cached, ok := p.catalog.Load(); ok {
		return p.effectiveDescriptors(cached), nil
	}
	ctx, cancel := context.WithTimeout(ctx, modelDiscoveryTimeout)
	defer cancel()
	discovered, err := p.listModelCatalog(ctx)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return p.fallbackDescriptors(), nil
	}
	p.catalog.Replace(discovered)
	return p.effectiveDescriptors(discovered), nil
}

func (p *Provider) listModelCatalog(ctx context.Context) ([]ai.ModelDescriptor, error) {
	client, err := p.getClient(ctx)
	if err != nil {
		return nil, err
	}

	page, err := client.Models.List(ctx, nil)
	if err != nil {
		return nil, err
	}

	var descriptors []ai.ModelDescriptor
	for {
		for _, model := range page.Items {
			if len(model.SupportedActions) > 0 && !containsModel(model.SupportedActions, "generateContent") {
				continue
			}
			name := strings.TrimSpace(strings.TrimPrefix(model.Name, "models/"))
			if name != "" {
				facts := ai.ModelDescriptor{Model: name}
				if model.Thinking {
					facts.Reasoning = ai.FeatureSupportSupported
				}
				descriptors = append(descriptors, facts)
			}
		}
		page, err = page.Next(ctx)
		if err == genai.ErrPageDone {
			return descriptors, nil
		}
		if err != nil {
			return nil, err
		}
	}
}

func (p *Provider) fallbackDescriptors() []ai.ModelDescriptor {
	descriptors := make([]ai.ModelDescriptor, 0, len(models))
	for _, model := range models {
		descriptors = append(descriptors, geminiAdapterDescriptor(model))
	}
	return descriptors
}

func (p *Provider) effectiveDescriptors(facts []ai.ModelDescriptor) []ai.ModelDescriptor {
	descriptors := make([]ai.ModelDescriptor, 0, len(facts))
	for _, fact := range facts {
		descriptors = append(descriptors, effectiveGeminiDescriptor(fact.Model, fact))
	}
	return descriptors
}

func effectiveGeminiDescriptor(model string, catalog ai.ModelDescriptor) ai.ModelDescriptor {
	adapter := geminiAdapterDescriptor(model)
	return ai.IntersectModelDescriptors(adapter, ai.OverrideModelDescriptor(adapter, catalog))
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
