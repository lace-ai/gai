package loop

import (
	"strconv"
	"strings"

	"github.com/lace-ai/gai/ai"
	aicontext "github.com/lace-ai/gai/context"
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
	request  *ai.AIRequest
	response *ai.AIResponse
	ToolReq  *ToolRequest
	ToolResp *ToolResponse
}

func (i *Iteration) String() string {
	if i == nil {
		return "<I>nil</I>"
	}
	var builder strings.Builder
	builder.WriteString("<I c=")
	builder.WriteString(strconv.Itoa(i.Count))
	builder.WriteString(" t=")
	builder.WriteString(string(i.Type))
	builder.WriteString(">")
	switch i.Type {
	case IterationTypeToolCall:
		if i.ToolReq == nil || i.ToolResp == nil {
			builder.WriteString("<Error>missing tool request/response</Error>")
			break
		}
		builder.WriteString("<Req u=assistant>")
		builder.WriteString(i.ToolReq.String())
		builder.WriteString("</Req>")
		builder.WriteString("<Resp u=tool>")
		builder.WriteString(i.ToolResp.String())
		builder.WriteString("</Resp>")
	case IterationTypeResponse:
		if i.response == nil {
			builder.WriteString("<Resp u=assistant></Resp>")
			break
		}
		builder.WriteString("<Resp u=assistant>")
		builder.WriteString(i.response.Text)
		builder.WriteString("</Resp>")
	case IterationTypeToolError:
		if i.ToolResp == nil || i.ToolResp.Err == nil {
			builder.WriteString("<Error u=tool>unknown error</Error>")
			break
		}
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

func (i *Iteration) Messages() []aicontext.Message {
	if i == nil {
		return nil
	}
	var msgs []aicontext.Message

	if i.Count == 1 {
		if i.request == nil {
			msgs = append(msgs, aicontext.Message{
				Role:    aicontext.RoleUser,
				Content: aicontext.NewTextContent(""),
			})
		}
		msgs = append(msgs, aicontext.Message{
			Role:    aicontext.RoleUser,
			Content: aicontext.NewTextContent(i.request.Prompt.Prompt),
		})
	}

	switch i.Type {
	case IterationTypeToolCall, IterationTypeToolError:
		if i.ToolReq != nil {
			msgs = append(msgs, aicontext.Message{
				Role:    aicontext.RoleAssistant,
				Content: aicontext.NewToolCallContent(i.ToolReq.ID, string(i.ToolReq.Args)),
			})
		}
		if i.ToolResp != nil && i.ToolReq != nil {
			if i.ToolResp.Err != nil {
				msgs = append(msgs, aicontext.Message{
					Role:    aicontext.RoleTool,
					Content: aicontext.NewTextContent("Error: " + i.ToolResp.Err.Error()),
				})
			} else {
				msgs = append(msgs, aicontext.Message{
					Role:    aicontext.RoleTool,
					Content: aicontext.NewToolResultContent(i.ToolReq.ID, i.ToolResp.Text, false, ""),
				})
			}
		}
	case IterationTypeResponse:
		if i.response != nil {
			msgs = append(msgs, aicontext.Message{
				Role:    aicontext.RoleAssistant,
				Content: aicontext.NewTextContent(i.response.Text),
			})
		}
	}

	return msgs
}
