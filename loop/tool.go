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
	Err  error
}

type ToolRequest struct {
	ID   string          `json:"id"`
	Type string          `json:"type"`
	Args json.RawMessage `json:"arguments"`
}

func (r *ToolRequest) Validate() error {
	if r == nil {
		return fmt.Errorf("%w: request is nil", ErrToolReqValidation)
	}
	if strings.TrimSpace(r.ID) == "" {
		return fmt.Errorf("%w: ID is empty", ErrToolReqValidation)
	}
	if strings.TrimSpace(r.Type) == "" {
		return fmt.Errorf("%w: Type is empty", ErrToolReqValidation)
	}
	if strings.TrimSpace(r.Type) != "function" {
		return fmt.Errorf(`%w: Type has to be "function", got=%v`, ErrToolReqValidation, r.Type)
	}
	return nil
}

type Tool interface {
	Name() string
	Description() string
	Params() string
	Function(req *ToolRequest) (*ToolResponse, error)
}

func DetectToolCall(s string) (*ToolRequest, bool) {
	payload := strings.TrimSpace(s)
	if payload == "" {
		return nil, false
	}
	if !strings.HasPrefix(payload, "{") {
		return nil, false
	}

	var tr ToolRequest
	if err := json.Unmarshal([]byte(payload), &tr); err != nil {
		return nil, false
	}
	if err := tr.Validate(); err != nil {
		return nil, false
	}
	return &tr, true
}

func CallTool(req *ToolRequest, tools []Tool) (*ToolResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	for _, tool := range tools {
		if tool.Name() == req.ID {
			return tool.Function(req)
		}
	}

	return nil, fmt.Errorf("%w: %s", ErrToolNotFound, req.ID)
}

func DecodeToolArgs[T any](req *ToolRequest, target *T) error {
	if err := req.Validate(); err != nil {
		return err
	}
	if target == nil {
		return ErrArgsDecodeTarget
	}
	if err := json.Unmarshal(req.Args, target); err != nil {
		return fmt.Errorf("%w: %w", ErrToolCallMalformed, err)
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

func (r *ToolRequest) String() string {
	var builder strings.Builder
	builder.WriteString("id: ")
	builder.WriteString(r.ID)
	builder.WriteString(",type: ")
	builder.WriteString(r.Type)
	builder.WriteString(",arguments: ")
	builder.Write(r.Args)
	return builder.String()
}

func (r *ToolResponse) String() string {
	var builder strings.Builder
	builder.WriteString(r.Text)
	return builder.String()
}
