package loop

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/lace-ai/gai/ai"
)

type ToolArg struct {
	ArgType     string `json:"type"`
	Description string `json:"description"`
}

type ToolResponse struct {
	Status string // "success" or "error"
	Text   *string
	Err    *error
}

func NewToolSuccess(text string) *ToolResponse {
	return &ToolResponse{
		Status: "success",
		Text:   &text,
	}
}

func NewToolError(err error) *ToolResponse {
	return &ToolResponse{
		Status: "error",
		Err:    &err,
	}
}

func (r *ToolResponse) TextValue() string {
	if r == nil || r.Text == nil {
		return ""
	}
	return *r.Text
}

func (r *ToolResponse) ErrorValue() error {
	if r == nil || r.Err == nil {
		return nil
	}
	return *r.Err
}

type Tool interface {
	Name() string
	Description() string
	Params() string
	Function(ctx context.Context, req *ai.ToolCall) *ToolResponse
}

func CallTool(ctx context.Context, req *ai.ToolCall, tools []Tool) *ToolResponse {
	if err := req.Validate(); err != nil {
		return NewToolError(err)
	}

	for _, tool := range tools {
		if tool.Name() == req.Name {
			res := tool.Function(ctx, req)
			if res == nil {
				return NewToolError(fmt.Errorf("tool %s returned nil response", req.Name))
			}
			return res
		}
	}

	return NewToolError(fmt.Errorf("%w: %s", ErrToolNotFound, req.Name))
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

func (r *ToolResponse) String() string {
	if r == nil {
		return ""
	}
	if r.Text != nil {
		return *r.Text
	}
	if r.Err != nil {
		return (*r.Err).Error()
	}
	return ""
}
