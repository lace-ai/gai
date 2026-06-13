package summary

import (
	"context"
	_ "embed"
	"fmt"
	"strings"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
)

//go:embed system.md
var DefaultSystemPrompt string

type Config struct {
	SystemPrompt      string
	Tools             []loop.Tool
	MaxLoopIterations int
	RetryCount        int
	MaxTokens         int
	Tokenizer         ai.Tokenizer
}

type Option func(*Config)

func WithSystemPrompt(prompt string) Option {
	return func(config *Config) {
		config.SystemPrompt = prompt
	}
}

func WithTools(tools ...loop.Tool) Option {
	return func(config *Config) {
		config.Tools = tools
	}
}

func WithMaxLoopIterations(maxLoopIterations int) Option {
	return func(config *Config) {
		config.MaxLoopIterations = maxLoopIterations
	}
}

func WithRetryCount(retryCount int) Option {
	return func(config *Config) {
		config.RetryCount = retryCount
	}
}

func WithMaxTokens(maxTokens int) Option {
	return func(config *Config) {
		config.MaxTokens = maxTokens
	}
}

func WithTokenizer(tokenizer ai.Tokenizer) Option {
	return func(config *Config) {
		config.Tokenizer = tokenizer
	}
}

func Definition(model ai.Model, opts ...Option) agent.Definition {
	config := Config{
		SystemPrompt:      DefaultSystemPrompt,
		MaxLoopIterations: 1,
		Tokenizer:         model.Tokenizer(),
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&config)
		}
	}
	systemPrompt := strings.TrimSpace(config.SystemPrompt)
	return agent.Definition{
		Name:  "summary",
		Model: model,
		Tools: config.Tools,
		Prompt: func(ctx context.Context, input agent.RunInput) gaictx.PromptBuilder {
			return gaictx.New(gaictx.Definition{
				Renderer:           gaictx.XMLRenderer{},
				SystemInstructions: []gaictx.Part{gaictx.NewTextPart(systemPrompt)},
				UserPrompt:         input.Text,
				TokenBudget:        -1,
				Tokenizer:          config.Tokenizer,
			})
		},
		Tokenizer: config.Tokenizer,
		Limits: agent.Limits{
			MaxLoopIterations: config.MaxLoopIterations,
			RetryCount:        config.RetryCount,
			MaxTokens:         config.MaxTokens,
		},
	}
}

func New(model ai.Model, opts ...Option) Summarizer {
	return Summarizer{
		Definition: Definition(model, opts...),
	}
}

type Summarizer struct {
	Definition agent.Definition
}

type activeKey struct{}

func (s Summarizer) Summarize(ctx context.Context, req gaictx.SummaryRequest) (string, error) {
	if ctx.Value(activeKey{}) == true {
		return "", fmt.Errorf("%w: recursive summary agent call", gaictx.ErrPromptSource)
	}
	def := s.Definition
	if def.Limits.MaxLoopIterations == 0 {
		def.Limits.MaxLoopIterations = 1
	}
	input := agent.RunInput{
		ID:        req.ID,
		Text:      req.Text,
		MaxTokens: req.MaxTokens,
		Meta:      req.Meta,
	}
	l, err := agent.New(def).NewRun(ctx, input)
	if err != nil {
		return "", err
	}

	runCtx := context.WithValue(ctx, activeKey{}, true)
	tokenCh, statusCh, errCh := l.Loop(runCtx)
	statusDone := make(chan struct{})
	go func() {
		defer close(statusDone)
		for range statusCh {
		}
	}()
	errDone := make(chan error, 1)
	go func() {
		var firstErr error
		for err := range errCh {
			if firstErr == nil && err != nil {
				firstErr = err
			}
		}
		errDone <- firstErr
	}()

	var summary []byte
	for token := range tokenCh {
		if token.Err != nil {
			<-statusDone
			<-errDone
			return "", token.Err
		}
		switch token.Type {
		case ai.TokenTypeText, ai.TokenTypeThought:
			if token.Text != "" {
				summary = append(summary, token.Text...)
			} else {
				summary = append(summary, token.Data...)
			}
		}
	}

	<-statusDone
	if err := <-errDone; err != nil {
		return "", err
	}
	if len(summary) == 0 {
		return "", fmt.Errorf("%w: summary agent produced no text", gaictx.ErrPromptSource)
	}
	return string(summary), nil
}
