package loop

import (
	"strconv"
	"strings"

	"agent-backend/gai/ai"
)

type IterationType string

const (
	// TODO: add thinking, ...
	IterationTypeToolCall  IterationType = "tool_call"
	IterationTypeResponse  IterationType = "response"
	IterationTypeToolError IterationType = "tool_error"
)

type Iteration struct {
	Count    int
	Type     IterationType
	response *ai.AIResponse
	ToolReq  *ToolRequest
	ToolResp *ToolResponse
}

func (i *Iteration) String() string {
	var builder strings.Builder
	builder.WriteString("<I c=")
	builder.WriteString(strconv.Itoa(i.Count))
	builder.WriteString(" t=")
	builder.WriteString(string(i.Type))
	builder.WriteString(">")
	switch i.Type {
	case IterationTypeToolCall:
		builder.WriteString("<Req u=assistant>")
		builder.WriteString(i.ToolReq.String())
		builder.WriteString("</Req>")
		builder.WriteString("<Resp u=tool>")
		builder.WriteString(i.ToolResp.String())
		builder.WriteString("</Resp>")
	case IterationTypeResponse:
		builder.WriteString("<Resp u=assistant>")
		builder.WriteString(i.response.Text)
		builder.WriteString("</Resp>")
	case IterationTypeToolError:
		builder.WriteString("<Error u=tool>")
		builder.WriteString(i.ToolResp.Err.Error())
		builder.WriteString("</Error>")
	}
	builder.WriteString("</I>")
	return builder.String()
}

func BuildIterationsString(builder *strings.Builder, iterations []Iteration) {
	for _, i := range iterations {
		builder.WriteString(i.String())
	}
}
