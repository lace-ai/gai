package loop

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/HecoAI/gai/ai"
	aicontext "github.com/HecoAI/gai/context"
)

const (
	defaultMaxLoopIterations = 8
)

type ContextBuilder interface {
	BuildContext(conv aicontext.Conversation) (string, error)
}
type ToolResPreProcessor interface {
	Process(req ai.ToolCall, res *ToolResponse) error
}

type Loop struct {
	InitialPrompt     ai.Prompt
	Iterations        []Iteration
	Model             ai.Model
	Tools             []Tool
	MaxLoopIterations int
	ContextBuilder    ContextBuilder
	PreProcessToolRes ToolResPreProcessor
}

func (a *Loop) Validate() error {
	if a == nil {
		return ErrNilAgent
	}
	if a.MaxLoopIterations <= 0 {
		a.MaxLoopIterations = defaultMaxLoopIterations
	}
	if a.Model == nil {
		return ErrModelNotConfigured
	}
	return nil
}

func New(model ai.Model, tools []Tool, initialPrompt string, sysPrompt string, contextBuilder ContextBuilder, toolResPreProcessor ToolResPreProcessor) *Loop {
	prompt := ai.Prompt{
		Prompt: initialPrompt,
		System: sysPrompt,
	}
	agent := &Loop{
		InitialPrompt:     prompt,
		Model:             model,
		Tools:             tools,
		MaxLoopIterations: defaultMaxLoopIterations,
		ContextBuilder:    contextBuilder,
		PreProcessToolRes: toolResPreProcessor,
	}
	return agent
}

func (a *Loop) Loop(ctx context.Context) (<-chan ai.Token, <-chan error) {
	errCh := make(chan error, 1)
	tokenCh := make(chan ai.Token)
	if err := a.Validate(); err != nil {
		errCh <- err
		return tokenCh, errCh
	}

	go func() {
		defer close(errCh)
		defer close(tokenCh)

		var iteration Iteration
		for i := range a.MaxLoopIterations {
			iteration = Iteration{Count: i + 1}

			if a.ContextBuilder != nil {
				context, err := a.ContextBuilder.BuildContext(a)
				if err != nil {
					errCh <- err
					return
				}
				a.InitialPrompt.Context = context
			} else {
				var builder strings.Builder
				aicontext.RenderMessages(a.Messages(), &builder)
				a.InitialPrompt.Context = builder.String()
			}

			request := ai.AIRequest{
				Prompt: a.InitialPrompt,
			}
			iteration.request = &request

			tokens := a.Model.GenerateStream(ctx, request)
			wg := sync.WaitGroup{}
			for t := range tokens {
				iteration.AppendToken(t)

				switch t.Type {
				case ai.TokenTypeErr:
					errCh <- t.Err
				case ai.TokenTypeText:
					tokenCh <- t
				case ai.TokenTypeTought:
					tokenCh <- t
				case ai.TokenTypeToolCall:
					part := iteration.CurrentPart()
					tokenCh <- t
					wg.Go(func() {
						res, err := CallTool(t.ToolCall, a.Tools)
						if err != nil {
							errCh <- err
							return
						}
						if a.PreProcessToolRes != nil {
							if err := a.PreProcessToolRes.Process(*t.ToolCall, res); err != nil {
								errCh <- err
								return
							}
						}
						part.ToolResp = res
					})
				}
			}
			wg.Wait()
			a.Iterations = append(a.Iterations, iteration)
		}

		errCh <- fmt.Errorf("%w: limit=%d", ErrMaxIterations, a.MaxLoopIterations)
	}()

	return nil, nil
}

func (a *Loop) Messages() []aicontext.Message {
	var msgs []aicontext.Message

	for _, i := range a.Iterations {
		msgs = append(msgs, i.Messages()...)
	}

	return msgs
}
