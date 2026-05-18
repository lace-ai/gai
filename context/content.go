package context

import (
	"encoding/json"
	"fmt"
)

// Content is the atomic unit of a message
type Content interface {
	String() string
	Type() string
	Marshal() ([]byte, error)
}

const (
	ContentTypeText          = "text"
	ContentTypeToolCall      = "tool_call"
	ContentTypeToolResult    = "tool_result"
	ContentTypeToolResultErr = "tool_result_err"
)

// TextContent is a generic content type
type TextContent struct {
	Text string
}

func (c TextContent) String() string {
	return c.Text
}

func (c TextContent) Type() string {
	return ContentTypeText
}

func (c TextContent) Marshal() ([]byte, error) {
	return json.Marshal(c)
}

func NewTextContent(text string) TextContent {
	return TextContent{Text: text}
}

// ToolCallContent represents a tool call
type ToolCallContent struct {
	ToolName string
	Args     string
}

func (c ToolCallContent) String() string {
	return c.ToolName + "(" + c.Args + ")"
}

func (c ToolCallContent) Type() string {
	return ContentTypeToolCall
}

func (c ToolCallContent) Marshal() ([]byte, error) {
	return json.Marshal(c)
}

func NewToolCallContent(toolName, args string) ToolCallContent {
	return ToolCallContent{
		ToolName: toolName,
		Args:     args,
	}
}

// ToolResultContent represents a tool result
type ToolResultContent struct {
	ToolName          string
	Result            string
	Precomputed       bool
	PrecomputedResult string
}

func (c ToolResultContent) String() string {
	return c.ToolName + " result: " + c.Result
}

func (c ToolResultContent) Type() string {
	return ContentTypeToolResult
}

func (c ToolResultContent) Marshal() ([]byte, error) {
	return json.Marshal(c)
}

func NewToolResultContent(toolName, result string, precomputed bool, precomputedResult string) ToolResultContent {
	return ToolResultContent{
		ToolName:          toolName,
		Result:            result,
		Precomputed:       precomputed,
		PrecomputedResult: precomputedResult,
	}
}

// ToolResultErrContent represents a tool error result
type ToolResultErrContent struct {
	ToolName string
	Err      string
}

func (c ToolResultErrContent) String() string {
	return c.ToolName + " error: " + c.Err
}

func (c ToolResultErrContent) Type() string {
	return ContentTypeToolResultErr
}

func (c ToolResultErrContent) Marshal() ([]byte, error) {
	return json.Marshal(c)
}

func NewToolResultErrContent(toolName, err string) ToolResultErrContent {
	return ToolResultErrContent{
		ToolName: toolName,
		Err:      err,
	}
}

func NewContentFromType(contentType string, data []byte) (Content, error) {
	switch contentType {
	case ContentTypeText:
		var textContent TextContent
		err := json.Unmarshal(data, &textContent)
		if err != nil {
			return nil, fmt.Errorf("failed to unmarshal content type %q: %w", contentType, err)
		}
		return textContent, nil
	case ContentTypeToolCall:
		var toolCallContent ToolCallContent
		err := json.Unmarshal(data, &toolCallContent)
		if err != nil {
			return nil, fmt.Errorf("failed to unmarshal content type %q: %w", contentType, err)
		}
		return toolCallContent, nil
	case ContentTypeToolResult:
		var toolResultContent ToolResultContent
		err := json.Unmarshal(data, &toolResultContent)
		if err != nil {
			return nil, fmt.Errorf("failed to unmarshal content type %q: %w", contentType, err)
		}
		return toolResultContent, nil
	case ContentTypeToolResultErr:
		var toolResultErrContent ToolResultErrContent
		err := json.Unmarshal(data, &toolResultErrContent)
		if err != nil {
			return nil, fmt.Errorf("failed to unmarshal content type %q: %w", contentType, err)
		}
		return toolResultErrContent, nil
	default:
		return nil, fmt.Errorf("unknown content type: %s", contentType)
	}
}
