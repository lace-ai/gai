package loop

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

type ToolArg struct {
	ArgType     string `json:"type"`
	Description string `json:"description"`
}

type ToolArgs struct {
	Args map[string]ToolArg `json:"arguments"`
}

type ToolResponse struct {
	Text string
}

type ToolRequest struct {
	ID   string          `json:"id"`
	Type string          `json:"type"`
	Args json.RawMessage `json:"arguments"`
}

type Tool interface {
	Name() string
	Description() string
	Params() string
	Function(req *ToolRequest) (*ToolResponse, error)
}

func detectToolCall(s string) (*ToolRequest, bool) {
	payload := strings.TrimSpace(s)
	if payload == "" {
		return nil, false
	}
	if !strings.HasPrefix(payload, "{") {
		return nil, false
	}

	var tr ToolRequest
	if err := json.Unmarshal([]byte(payload), &tr); err != nil {
		return nil, true
	}
	return &tr, true
}

func (r *ToolRequest) ArgsString() string {
	if r == nil {
		return ""
	}
	if len(r.Args) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(r.Args, &s); err == nil {
		return s
	}
	return string(r.Args)
}

func callTool(req *ToolRequest, tools []Tool) (*ToolResponse, error) {
	if req == nil {
		return nil, ErrInvalidToolRequest
	}
	if strings.TrimSpace(req.Type) == "" {
		return nil, ErrToolCallType
	}
	if req.Type != "function" {
		return nil, fmt.Errorf("%w: got=%s", ErrToolCallType, req.Type)
	}
	if strings.TrimSpace(req.ID) == "" {
		return nil, ErrToolCallID
	}
	if len(req.Args) == 0 {
		return nil, ErrToolArgsMissing
	}

	for _, tool := range tools {
		if tool.Name() == req.ID {
			return tool.Function(req)
		}
	}

	return nil, fmt.Errorf("%w: %s", ErrToolNotFound, req.ID)
}

func DecodeToolArgs[T any](req *ToolRequest, target *T) error {
	if req == nil {
		return ErrInvalidToolRequest
	}
	if target == nil {
		return ErrArgsDecodeTarget
	}
	if len(req.Args) == 0 {
		return ErrToolArgsMissing
	}
	if err := json.Unmarshal(req.Args, target); err != nil {
		return fmt.Errorf("%w: %v", ErrToolCallMalformed, err)
	}
	return nil
}

func RenderToolSignatures(tools []Tool) string {
	if len(tools) == 0 {
		return ""
	}

	sorted := make([]Tool, 0, len(tools))
	for _, tool := range tools {
		if tool != nil {
			sorted = append(sorted, tool)
		}
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Name() < sorted[j].Name()
	})

	var builder strings.Builder
	for _, t := range sorted {
		builder.WriteString("\n<tool name=\"")
		builder.WriteString(t.Name())
		builder.WriteString("\">")
		builder.WriteString("\n<description>")
		builder.WriteString(t.Description())
		builder.WriteString("</description>")
		builder.WriteString("\n<signature>")
		builder.WriteString(t.Params())
		builder.WriteString("</signature>")
		builder.WriteString("\n</tool>")
	}
	return builder.String()
}
