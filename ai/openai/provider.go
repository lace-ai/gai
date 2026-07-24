package openai

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

const modelDiscoveryTimeout = 10 * time.Second

// Provider resolves models served by OpenAI.
type Provider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
	debug      gai.DebugSink
}

var _ ai.Provider = (*Provider)(nil)

func New(apiKey string, debug gai.DebugSink) *Provider {
	return &Provider{
		apiKey:     apiKey,
		baseURL:    "https://api.openai.com/v1",
		httpClient: &http.Client{},
		debug:      debug,
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

func (p *Provider) Name() string { return "openai" }

// streamingHTTPClient returns a client whose request lifetime is controlled by
// the caller's context, rather than the provider's non-streaming timeout.
func (p *Provider) streamingHTTPClient() *http.Client {
	client := *p.httpClient
	client.Timeout = 0
	return &client
}

func (p *Provider) Model(name string) (ai.Model, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	name = strings.TrimSpace(name)
	if name == "" {
		return nil, ai.ErrModelNotFound
	}
	return &Model{name: name, provider: p}, nil
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
	if p.httpClient == nil {
		return nil, fmt.Errorf("openai model discovery: nil HTTP client")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(p.baseURL, "/")+"/models", nil)
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
		return nil, fmt.Errorf("openai model discovery: unexpected status %s", res.Status)
	}

	var payload struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(res.Body).Decode(&payload); err != nil {
		return nil, err
	}

	names := make([]string, 0, len(payload.Data))
	for _, model := range payload.Data {
		if name := strings.TrimSpace(model.ID); isChatCapableModel(name) {
			names = append(names, name)
		}
	}
	return names, nil
}

// isChatCapableModel excludes model families that the Models endpoint lists but
// that cannot be used with this provider's Chat Completions adapter. The
// endpoint does not expose per-endpoint capabilities, so unknown IDs remain
// discoverable for compatible custom and future chat models.
func isChatCapableModel(name string) bool {
	name = strings.ToLower(strings.TrimSpace(name))
	if name == "" {
		return false
	}
	for _, prefix := range []string{
		"dall-e-",
		"gpt-image-",
		"text-embedding-",
		"text-moderation-",
		"omni-moderation-",
		"whisper-",
		"tts-",
		"sora-",
		"computer-use-",
	} {
		if strings.HasPrefix(name, prefix) {
			return false
		}
	}
	return name != "gpt-5.6-auto"
}

func fallbackModels() []string {
	out := make([]string, len(models))
	copy(out, models)
	return out
}

func isKnownModel(name string) bool {
	for _, model := range models {
		if name == model {
			return true
		}
	}
	return false
}
