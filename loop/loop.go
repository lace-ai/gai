package loop

import (
	"context"
	"fmt"
	"strings"

	"github.com/lace-ai/gai/ai"
	aicontext "github.com/lace-ai/gai/context"
)

const (
	defaultMaxLoopIterations = 8
	defaultMaxMessages       = 100
)

type ContextBuilder interface {
	BuildContext(conv aicontext.Conversation) (string, error)
}
type ToolResPreProcessor interface {
	Process(req ToolRequest, res *ToolResponse) error
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

func (a *Loop) Loop(ctx context.Context) error {
	if err := a.Validate(); err != nil {
		return err
	}

	var iteration Iteration
	for i := range a.MaxLoopIterations {
		iteration = Iteration{Count: i + 1}

		if a.ContextBuilder != nil {
			context, err := a.ContextBuilder.BuildContext(a)
			if err != nil {
				return fmt.Errorf("%w: %w", ErrBuildContext, err)
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

		res, err := a.Model.Generate(ctx, request)
		if err != nil {
			return err
		}

		iteration.response = res

		toolReq, tCall := DetectToolCall(res.Text)
		if !tCall {
			iteration.Type = IterationTypeResponse
			a.Iterations = append(a.Iterations, iteration)
			return nil
		}

		iteration.Type = IterationTypeToolCall

		if toolReq == nil {
			return ErrToolCallMalformed
		}

		toolRes, err := CallTool(toolReq, a.Tools)
		if err != nil {
			return err
		}

		if a.PreProcessToolRes != nil {
			if err := a.PreProcessToolRes.Process(*toolReq, toolRes); err != nil {
				return fmt.Errorf("%w: %w", ErrPreProcessToolRes, err)
			}
		}

		iteration.ToolResp = toolRes
		iteration.ToolReq = toolReq

		a.Iterations = append(a.Iterations, iteration)
	}

	return fmt.Errorf("%w: limit=%d", ErrMaxIterations, a.MaxLoopIterations)
}

func (a *Loop) Messages() []aicontext.Message {
	var msgs []aicontext.Message

	for _, i := range a.Iterations {
		msgs = append(msgs, i.Messages()...)
	}

	return msgs
}
