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

type Loop struct {
	InitialPrompt     string
	Iterations        []Iteration
	Model             ai.Model
	Tools             []Tool
	MaxLoopIterations int
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

func New(model ai.Model, tools []Tool, initialPrompt string) *Loop {
	agent := &Loop{
		InitialPrompt:     initialPrompt,
		Model:             model,
		Tools:             tools,
		MaxLoopIterations: defaultMaxLoopIterations,
	}
	return agent
}

func (a *Loop) Loop(ctx context.Context, sysPrompt string, buildContext func([]Iteration) string, preProcessToolRes func(req ToolRequest, res *ToolResponse) error) error {
	if err := a.Validate(); err != nil {
		return err
	}

	var iteration Iteration
	for i := range a.MaxLoopIterations {
		iteration = Iteration{Count: i + 1}
		request := ai.AIRequest{
			SystemPrompt: sysPrompt,
			Prompt:       a.InitialPrompt,
			Context:      buildContext(a.Iterations),
		}

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

		if err := preProcessToolRes(*toolReq, toolRes); err != nil {
			return fmt.Errorf("%w: %v", ErrPreProcessToolRes, err)
		}

		iteration.ToolResp = toolRes
		iteration.ToolReq = toolReq

		a.Iterations = append(a.Iterations, iteration)
	}

	return fmt.Errorf("%w: limit=%d", ErrMaxIterations, a.MaxLoopIterations)
}
