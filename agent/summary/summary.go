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
	PreProcessToolRes loop.ToolResPreProcessor
	MaxLoopIterations int
	RetryCount        int
	MaxTokens         int
	PromptBudget      *gaictx.PromptBudget
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

func WithPreProcessor(preProcessor loop.ToolResPreProcessor) Option {
	return func(config *Config) {
		config.PreProcessToolRes = preProcessor
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

func WithPromptBudget(budget gaictx.PromptBudget) Option {
	return func(config *Config) {
		config.PromptBudget = &budget
	}
}

func Definition(model ai.Model, opts ...Option) agent.Definition {
	config := Config{
		SystemPrompt:      DefaultSystemPrompt,
		MaxLoopIterations: 1,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&config)
		}
	}
	systemPrompt := strings.TrimSpace(config.SystemPrompt)
	return agent.Definition{
		Model:             model,
		Tools:             config.Tools,
		PreProcessToolRes: config.PreProcessToolRes,
		MaxLoopIterations: config.MaxLoopIterations,
		RetryCount:        config.RetryCount,
		MaxTokens:         config.MaxTokens,
		PromptBuilderFactory: func(input agent.RunInput) gaictx.PromptBuilder {
			builder := gaictx.NewPromptBuilder()
			if config.PromptBudget != nil {
				builder.Budget(*config.PromptBudget)
			}
			return builder.
				System("summary-system", systemPrompt, gaictx.Required()).
				User("summary-request", input.Text, gaictx.Required())
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
	if def.MaxLoopIterations == 0 {
		def.MaxLoopIterations = 1
	}
	input := agent.RunInput{
		ID:        req.ID,
		Text:      req.Text,
		MaxTokens: req.MaxTokens,
		Meta:      req.Meta,
	}
	l, err := agent.NewLoop(def, input)
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
