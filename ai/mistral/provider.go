package mistral

import (
	"context"
	"encoding/json"
	"fmt"
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
	if p.httpClient == nil {
		return nil, fmt.Errorf("mistral model discovery: nil HTTP client")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(p.baseURL, "/")+"/v1/models", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+p.apiKey)

	res, err := p.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode < http.StatusOK || res.StatusCode >= http.StatusMultipleChoices {
		return nil, fmt.Errorf("mistral model discovery: unexpected status %s", res.Status)
	}

	var payload struct {
		Data []struct {
			ID           string `json:"id"`
			Capabilities struct {
				CompletionChat bool `json:"completion_chat"`
			} `json:"capabilities"`
		} `json:"data"`
	}
	if err := json.NewDecoder(res.Body).Decode(&payload); err != nil {
		return nil, err
	}

	names := make([]string, 0, len(payload.Data))
	for _, model := range payload.Data {
		if name := strings.TrimSpace(model.ID); name != "" && model.Capabilities.CompletionChat {
			names = append(names, name)
		}
	}
	return names, nil
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
