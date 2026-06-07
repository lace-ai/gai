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
	ContentTypeTool          = "tool"
	ContentTypeAgent         = "agent"
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

// ToolContent represents the content of a tool call, including the call itself, whether it was successful, and the result or error
type ToolContent struct {
	Call    ToolCallContent
	Success bool
	Result  *ToolResultContent
	Err     *ToolResultErrContent
}

func (c ToolContent) String() string {
	if c.Success {
		return c.Call.String() + " -> " + c.Result.String()
	}
	return c.Call.String() + " -> " + c.Err.String()
}

func (c ToolContent) Type() string {
	return ContentTypeTool
}

func (c ToolContent) Marshal() ([]byte, error) {
	return json.Marshal(c)
}

func NewToolContent(call ToolCallContent, success bool, result *ToolResultContent, err *ToolResultErrContent) ToolContent {
	return ToolContent{
		Call:    call,
		Success: success,
		Result:  result,
		Err:     err,
	}
}

// AgentContent represents the content of an agent response, including the response text and any tool calls made by the agent
type AgentContent struct {
	Response  string
	ToolCalls []ToolCallContent
}

func (c AgentContent) String() string {
	return "Agent response: " + c.Response
}

func (c AgentContent) Type() string {
	return ContentTypeAgent
}

func (c AgentContent) Marshal() ([]byte, error) {
	return json.Marshal(c)
}

func NewAgentContent(response string, toolCalls []ToolCallContent) AgentContent {
	return AgentContent{
		Response:  response,
		ToolCalls: toolCalls,
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
