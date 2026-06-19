// Package exa provides prebuilt tools backed by the Exa API.
package exa

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/loop"
)

const (
	defaultEndpoint    = "https://api.exa.ai/search"
	defaultSearchType  = "auto"
	defaultNumResults  = 10
	maxResponseBytes   = 4 << 20
	maxErrorBodyBytes  = 4 << 10
	apiKeyEnvironment  = "EXA_API_KEY"
	defaultHTTPTimeout = 60 * time.Second
)

var (
	ErrAPIKeyMissing    = errors.New("exa API key is missing")
	ErrInvalidOption    = errors.New("invalid Exa search option")
	ErrResponseTooLarge = errors.New("Exa response is too large")
	ErrInvalidResponse  = errors.New("Exa returned invalid JSON")
)

// SearchTool searches the web with Exa and returns the raw JSON response to the
// agent. Its default request uses type=auto and query-relevant highlights.
type SearchTool struct {
	apiKey     string
	endpoint   string
	client     *http.Client
	searchType string
	numResults int
}

var _ loop.Tool = (*SearchTool)(nil)

// Option configures a SearchTool.
type Option func(*SearchTool) error

// WithSearchType sets the Exa search type.
func WithSearchType(searchType string) Option {
	return func(tool *SearchTool) error {
		switch searchType {
		case "auto", "fast", "instant", "deep-lite", "deep", "deep-reasoning":
			tool.searchType = searchType
			return nil
		default:
			return fmt.Errorf("%w: unsupported search type %q", ErrInvalidOption, searchType)
		}
	}
}

// WithNumResults sets the number of search results requested from Exa.
func WithNumResults(numResults int) Option {
	return func(tool *SearchTool) error {
		if numResults < 1 || numResults > 100 {
			return fmt.Errorf("%w: num results must be between 1 and 100", ErrInvalidOption)
		}
		tool.numResults = numResults
		return nil
	}
}

// WithHTTPClient replaces the HTTP client. It is useful for setting transport,
// proxy, and timeout policy at the application boundary.
func WithHTTPClient(client *http.Client) Option {
	return func(tool *SearchTool) error {
		if client == nil {
			return fmt.Errorf("%w: HTTP client is nil", ErrInvalidOption)
		}
		tool.client = client
		return nil
	}
}

// WithEndpoint replaces the Exa endpoint. Most applications should use the
// default; this option also supports compatible gateways and local tests.
func WithEndpoint(endpoint string) Option {
	return func(tool *SearchTool) error {
		if strings.TrimSpace(endpoint) == "" {
			return fmt.Errorf("%w: endpoint is empty", ErrInvalidOption)
		}
		tool.endpoint = endpoint
		return nil
	}
}

// NewSearchTool constructs an Exa web search tool.
func NewSearchTool(apiKey string, options ...Option) (*SearchTool, error) {
	apiKey = strings.TrimSpace(apiKey)
	if apiKey == "" {
		return nil, ErrAPIKeyMissing
	}

	tool := &SearchTool{
		apiKey:     apiKey,
		endpoint:   defaultEndpoint,
		client:     &http.Client{Timeout: defaultHTTPTimeout},
		searchType: defaultSearchType,
		numResults: defaultNumResults,
	}
	for _, option := range options {
		if option == nil {
			return nil, fmt.Errorf("%w: option is nil", ErrInvalidOption)
		}
		if err := option(tool); err != nil {
			return nil, err
		}
	}
	return tool, nil
}

// NewSearchToolFromEnv constructs a tool using EXA_API_KEY.
func NewSearchToolFromEnv(options ...Option) (*SearchTool, error) {
	return NewSearchTool(os.Getenv(apiKeyEnvironment), options...)
}

func (t *SearchTool) Name() string {
	return "web_search"
}

func (t *SearchTool) Description() string {
	return "Searches the web for current or factual information and returns relevant pages with excerpts."
}

func (t *SearchTool) Params() string {
	return `{"type":"object","required":["query"],"properties":{"query":{"type":"string","description":"A specific natural-language web search query"}}}`
}

type searchArgs struct {
	Query string `json:"query"`
}

type searchRequest struct {
	Query      string         `json:"query"`
	Type       string         `json:"type"`
	NumResults int            `json:"numResults"`
	Contents   searchContents `json:"contents"`
}

type searchContents struct {
	Highlights bool `json:"highlights"`
}

func (t *SearchTool) Function(ctx context.Context, call *ai.ToolCall) *loop.ToolResponse {
	var args searchArgs
	if err := loop.DecodeToolArgs(call, &args); err != nil {
		return &loop.ToolResponse{Err: err}
	}
	args.Query = strings.TrimSpace(args.Query)
	if args.Query == "" {
		return &loop.ToolResponse{Err: fmt.Errorf("%w: query is empty", loop.ErrToolReqValidation)}
	}

	payload, err := json.Marshal(searchRequest{
		Query:      args.Query,
		Type:       t.searchType,
		NumResults: t.numResults,
		Contents:   searchContents{Highlights: true},
	})
	if err != nil {
		return &loop.ToolResponse{Err: fmt.Errorf("marshal Exa search request: %w", err)}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.endpoint, bytes.NewReader(payload))
	if err != nil {
		return &loop.ToolResponse{Err: fmt.Errorf("create Exa search request: %w", err)}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", t.apiKey)

	res, err := t.client.Do(req)
	if err != nil {
		return &loop.ToolResponse{Err: fmt.Errorf("execute Exa search request: %w", err)}
	}
	defer res.Body.Close()

	body, err := io.ReadAll(io.LimitReader(res.Body, maxResponseBytes+1))
	if err != nil {
		return &loop.ToolResponse{Err: fmt.Errorf("read Exa search response: %w", err)}
	}
	if len(body) > maxResponseBytes {
		return &loop.ToolResponse{Err: ErrResponseTooLarge}
	}
	if res.StatusCode < http.StatusOK || res.StatusCode >= http.StatusMultipleChoices {
		message := strings.TrimSpace(string(body))
		if len(message) > maxErrorBodyBytes {
			message = message[:maxErrorBodyBytes]
		}
		return &loop.ToolResponse{Err: fmt.Errorf("Exa search failed with status %d: %s", res.StatusCode, message)}
	}
	if !json.Valid(body) {
		return &loop.ToolResponse{Err: ErrInvalidResponse}
	}

	return &loop.ToolResponse{Text: string(body)}
}
