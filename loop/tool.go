package loop

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/lace-ai/gai/ai"
)

// ToolArg describes one tool argument in a provider-independent form.
type ToolArg struct {
	// ArgType is the argument's JSON Schema type.
	ArgType string `json:"type"`
	// Description explains the argument to the model.
	Description string `json:"description"`
}

// ToolResponse is the result of a tool invocation.
type ToolResponse struct {
	// Text contains successful tool output to return to the model.
	Text string
	// Err contains an invocation or tool error.
	Err error
}

// Tool defines a function that a model may request during a loop run.
type Tool interface {
	// Name returns the function name exposed to the model.
	Name() string
	// Description explains when and how the model should use the tool.
	Description() string
	// Params returns the tool parameters as JSON Schema.
	Params() string
	// Function invokes the tool for req.
	Function(ctx context.Context, req *ai.ToolCall) *ToolResponse
}

// CallTool validates req and invokes the matching tool by name.
// It returns an error response when validation fails, no tool matches, or a
// tool returns nil.
func CallTool(ctx context.Context, req *ai.ToolCall, tools []Tool) *ToolResponse {
	if err := req.Validate(); err != nil {
		return &ToolResponse{Err: err}
	}

	for _, tool := range tools {
		if tool.Name() == req.Name {
			res := tool.Function(ctx, req)
			if res == nil {
				return &ToolResponse{Err: fmt.Errorf("tool %s returned nil response", req.Name)}
			}
			return res
		}
	}

	return &ToolResponse{Err: fmt.Errorf("%w: %s", ErrToolNotFound, req.Name)}
}

// DecodeToolArgs validates req and decodes its JSON arguments into target.
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

// RenderToolSignatures renders non-nil tools as deterministic XML fragments,
// sorted by tool name.
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

// ToolCallToString returns a diagnostic representation of tc.
func ToolCallToString(tc ai.ToolCall) string {
	var builder strings.Builder
	builder.WriteString("id: ")
	builder.WriteString(tc.ID)
	builder.WriteString(",type: ")
	builder.WriteString(tc.Type)
	builder.WriteString(",name: ")
	builder.WriteString(tc.Name)
	builder.WriteString(",arguments: ")
	builder.Write(tc.Args)
	return builder.String()
}

func toolCallSignature(tc ai.ToolCall) string {
	args := strings.TrimSpace(string(tc.Args))
	if len(tc.Args) > 0 {
		var compact bytes.Buffer
		if err := json.Compact(&compact, tc.Args); err == nil {
			args = compact.String()
		}
	}
	return tc.Name + "\x00" + args
}

// String returns the response text.
func (r *ToolResponse) String() string {
	var builder strings.Builder
	builder.WriteString(r.Text)
	return builder.String()
}
