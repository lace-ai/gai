package loop

import (
	"context"
	"fmt"

	"agent-backend/gai/ai"
)

const (
	defaultMaxLoopIterations = 8
	defaultMaxMessages       = 100
)

type ContextBuilder interface {
	BuildContext(iterations []Iteration) string
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

func (a *Loop) Loop(ctx context.Context, sysPrompt string) error {
	if err := a.Validate(); err != nil {
		return err
	}

	var iteration Iteration
	for i := range a.MaxLoopIterations {
		iteration = Iteration{Count: i + 1}

		a.InitialPrompt.Context = a.ContextBuilder.BuildContext(a.Iterations)
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
			iteration.IterType = IterationTypeResponse
			a.Iterations = append(a.Iterations, iteration)
			return nil
		}

		iteration.IterType = IterationTypeToolCall

		if toolReq == nil {
			return ErrToolCallMalformed
		}

		toolRes, err := CallTool(toolReq, a.Tools)
		if err != nil {
			return err
		}

		if err := a.PreProcessToolRes.Process(*toolReq, toolRes); err != nil {
			return fmt.Errorf("%w: %v", ErrPreProcessToolRes, err)
		}

		iteration.ToolResp = toolRes
		iteration.ToolReq = toolReq

		a.Iterations = append(a.Iterations, iteration)
	}

	return fmt.Errorf("%w: limit=%d", ErrMaxIterations, a.MaxLoopIterations)
}
