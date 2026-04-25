package loop

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/lace-ai/gai/ai"
)

type ToolArg struct {
	ArgType     string `json:"type"`
	Description string `json:"description"`
}

type ToolResponse struct {
	Text string
	Err  error
}

type Tool interface {
	Name() string
	Description() string
	Params() string
	Function(req *ai.ToolCall) *ToolResponse
}

func CallTool(req *ai.ToolCall, tools []Tool) *ToolResponse {
	if err := req.Validate(); err != nil {
		return &ToolResponse{Err: err}
	}

	for _, tool := range tools {
		if tool.Name() == req.ID {
			return tool.Function(req)
		}
	}

	return &ToolResponse{Err: fmt.Errorf("%w: %s", ErrToolNotFound, req.ID)}
}

func DecodeToolArgs[T any](req *ai.ToolCall, target *T) error {
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

func ToolCallToString(tc ai.ToolCall) string {
	var builder strings.Builder
	builder.WriteString("id: ")
	builder.WriteString(tc.ID)
	builder.WriteString(",type: ")
	builder.WriteString(tc.Name)
	builder.WriteString(",arguments: ")
	builder.Write(tc.Args)
	return builder.String()
}

func (r *ToolResponse) String() string {
	var builder strings.Builder
	builder.WriteString(r.Text)
	return builder.String()
}
