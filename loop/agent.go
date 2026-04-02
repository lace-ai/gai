package loop

import (
	"context"
	"fmt"
	"strings"

	"agent-backend/gai/ai"
	"agent-backend/gai/memory"
)

const (
	defaultMaxLoopIterations = 8
	defaultMaxMessages       = 100
)

type Agent struct {
	Model             ai.Model
	Tools             []Tool
	BaseSystemPrompt  string
	ToolSystemPrompt  string
	MaxLoopIterations int
	MaxMessages       int
	MemorySystem      memory.Memory
}

func NewAgent(model ai.Model, tools []Tool, systemPrompt string, sessionID int, toolPrompt string) (*Agent, error) {
	m, err := memory.NewMemory(sessionID)
	if err != nil {
		return nil, err
	}
	agent := &Agent{
		Model:             model,
		Tools:             tools,
		BaseSystemPrompt:  systemPrompt,
		ToolSystemPrompt:  toolPrompt,
		MaxLoopIterations: defaultMaxLoopIterations,
		MaxMessages:       defaultMaxMessages,
		MemorySystem:      m,
	}
	return agent, nil
}

func NewAgentFromPromptFiles(model ai.Model, tools []Tool, systemPromptPath string, toolPromptPath string, sessionID int) (*Agent, error) {
	basePrompt, err := LoadPromptFromFile(systemPromptPath)
	if err != nil {
		return nil, err
	}
	toolPrompt, err := LoadOptionalPromptFromFile(toolPromptPath, toolPromptPath)
	if err != nil {
		return nil, err
	}
	return NewAgent(model, tools, basePrompt, sessionID, toolPrompt)
}

func (a *Agent) FollowUp(ctx context.Context, prompt string) (string, error) {
	if a == nil {
		return "", ErrNilAgent
	}
	if a.Model == nil {
		return "", ErrModelNotConfigured
	}
	if a.MemorySystem == nil {
		return "", ErrMemoryNotConfigured
	}
	if strings.TrimSpace(prompt) == "" {
		return "", ErrEmptyPrompt
	}

	userMessage := memory.Message{Content: prompt, Role: memory.RoleUser}
	var response strings.Builder

	err := a.Loop(ctx, userMessage, &response)
	if err != nil {
		return response.String(), err
	}

	return response.String(), nil
}

func (a *Agent) Loop(ctx context.Context, message memory.Message, response *strings.Builder) error {
	if response == nil {
		return ErrNilResponseBuilder
	}
	if a.MemorySystem == nil {
		return ErrMemoryNotConfigured
	}

	maxIterations := a.MaxLoopIterations

	if maxIterations <= 0 {
		maxIterations = defaultMaxLoopIterations
	}

	_, err := a.MemorySystem.AddMessage(message.Content, message.Role)
	if err != nil {
		return err
	}

	for i := 0; i < maxIterations; i++ {
		request := ai.AIRequest{
			SystemPrompt: buildSystemPrompt(a.BaseSystemPrompt, a.ToolSystemPrompt, a.Tools),
		}
		contextPrompt, err := a.MemorySystem.EnrichPrompt("")
		if err != nil {
			return err
		}
		request.Context = contextPrompt
		res, err := a.Model.Generate(ctx, request)
		if err != nil {
			return err
		}

		message, err := a.MemorySystem.AddMessage(res.Text, memory.RoleAssistant)
		if err != nil {
			return err
		}

		response.WriteString("\n\n")
		response.WriteString("Agent: \n")
		response.WriteString("\t")
		response.WriteString(message.Content)

		toolReq, tCall := detectToolCall(res.Text)
		if !tCall {
			return nil
		}

		var toolRes *ToolResponse
		if toolReq == nil {
			err = ErrToolCallMalformed
		} else {
			toolRes, err = callTool(toolReq, a.Tools)
		}
		toolResultText := ""
		if err != nil {
			toolResultText = err.Error()
		} else if toolRes != nil {
			toolResultText = toolRes.Text
		}

		toolID := ""
		toolArgs := ""
		if toolReq != nil {
			toolID = toolReq.ID
			toolArgs = toolReq.ArgsString()
		}

		response.WriteString("\n\n")
		response.WriteString("Tool ")
		response.WriteString(toolID)
		response.WriteString(" ")
		response.WriteString(toolArgs)
		response.WriteString(":\n")
		response.WriteString("\t")
		response.WriteString(toolResultText)

		_, addErr := a.MemorySystem.AddMessage(toolResultText, memory.RoleTool)
		if addErr != nil {
			return addErr
		}

		if err != nil {
			return err
		}
	}

	return fmt.Errorf("%w: limit=%d", ErrMaxIterations, maxIterations)
}

func buildSystemPrompt(baseSystemPrompt, toolSystemPrompt string, tools []Tool) string {
	var builder strings.Builder

	if strings.TrimSpace(baseSystemPrompt) != "" {
		builder.WriteString(baseSystemPrompt)
		builder.WriteString("\n\n")
	}

	if len(tools) > 0 {
		prompt := strings.TrimSpace(toolSystemPrompt)
		builder.WriteString(prompt)
		builder.WriteString("<tools>\n")
		builder.WriteString(RenderToolSignatures(tools))
		builder.WriteString("\n</tools>")
	}

	return builder.String()
}
