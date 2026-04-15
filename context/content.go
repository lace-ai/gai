package context

// Content is the atomic unit of a message
type Content interface {
	String() string
	Type() string
}

// TextContent is a generic content type
type TextContent struct {
	Text string
}

func (c TextContent) String() string {
	return c.Text
}

func (c TextContent) Type() string {
	return "text"
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
	return "tool_call"
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
	return "tool_result"
}

func NewToolResultContent(toolName, result string, precomputed bool, precomputedResult string) ToolResultContent {
	return ToolResultContent{
		ToolName:          toolName,
		Result:            result,
		Precomputed:       precomputed,
		PrecomputedResult: precomputedResult,
	}
}
