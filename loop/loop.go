package loop

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/lace-ai/gai/ai"
	aicontext "github.com/lace-ai/gai/context"
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
		close(errCh)
		close(tokenCh)
		return tokenCh, errCh
	}

	go func() {
		defer close(errCh)
		defer close(tokenCh)

		var iteration Iteration
		for i := range a.MaxLoopIterations {
			iteration = Iteration{Count: i + 1}
			toolCalls := 0

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
				tokenCh <- t

				if t.Type == ai.TokenTypeToolCall {
					toolReq := t.ToolCall
					partIdx := len(iteration.Parts) - 1

					toolCalls++

					wg.Go(func() {
						res := CallTool(toolReq, a.Tools)
						if a.PreProcessToolRes != nil {
							if err := a.PreProcessToolRes.Process(*toolReq, res); err != nil {
								res.Err = err
							}
						}

						iteration.Parts[partIdx].ToolResp = res
					})
				}
			}

			wg.Wait()

			a.Iterations = append(a.Iterations, iteration)
			if toolCalls == 0 {
				return
			}
		}

		errCh <- fmt.Errorf("%w: limit=%d", ErrMaxIterations, a.MaxLoopIterations)
	}()

	return tokenCh, errCh
}

func (a *Loop) Messages() []aicontext.Message {
	var msgs []aicontext.Message

	for _, i := range a.Iterations {
		msgs = append(msgs, i.Messages()...)
	}

	return msgs
}
