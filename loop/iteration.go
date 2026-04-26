package loop

import (
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

type IterationInformation struct {
	Iteration      Iteration
	IterationCount int
	PartCount      int
	RetryCount     int
}

type Iteration struct {
	Count   int
	Parts   []IterationPart
	Request *ai.AIRequest
}

type IterationPart struct {
	Type     IterationType
	Response *ai.AIResponse
	ToolReq  *ai.ToolCall
	ToolResp *ToolResponse
}

func (i *IterationPart) String() string {
	if i == nil {
		return "<I>nil</I>"
	}
	var builder strings.Builder
	builder.WriteString("<I")
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
		if i.Response == nil {
			builder.WriteString("<Resp u=assistant></Resp>")
			break
		}
		builder.WriteString("<Resp u=assistant>")
		builder.WriteString(i.Response.Text)
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
		for _, part := range i.Parts {
			builder.WriteString(part.String())
		}
	}
}

func (i *Iteration) Messages() []aicontext.Message {
	if i == nil {
		return nil
	}
	var msgs []aicontext.Message

	if i.Count == 1 {
		if i.Request != nil {
			msgs = append(msgs, aicontext.Message{
				Role:    aicontext.RoleUser,
				Content: aicontext.NewTextContent(i.Request.Prompt.Prompt),
			})
		}
	}

	for _, part := range i.Parts {
		switch part.Type {
		case IterationTypeToolCall, IterationTypeToolError:
			if part.ToolReq != nil {
				msgs = append(msgs, aicontext.Message{
					Role:    aicontext.RoleAssistant,
					Content: aicontext.NewToolCallContent(part.ToolReq.ID, string(part.ToolReq.Args)),
				})
				if part.ToolResp != nil {
					if part.ToolResp.Err != nil {
						msgs = append(msgs, aicontext.Message{
							Role:    aicontext.RoleTool,
							Content: aicontext.NewToolResultErrContent(part.ToolReq.ID, part.ToolResp.Err.Error()),
						})
					} else {
						msgs = append(msgs, aicontext.Message{
							Role:    aicontext.RoleTool,
							Content: aicontext.NewToolResultContent(part.ToolReq.ID, part.ToolResp.Text, false, ""),
						})
					}
				}
			}
		case IterationTypeResponse:
			if part.Response != nil {
				msgs = append(msgs, aicontext.Message{
					Role:    aicontext.RoleAssistant,
					Content: aicontext.NewTextContent(part.Response.Text),
				})
			}
		}
	}

	return msgs
}

func (i *Iteration) CurrentPart() *IterationPart {
	if len(i.Parts) == 0 {
		return nil
	}
	return &i.Parts[len(i.Parts)-1]
}

func (i *Iteration) AppendToken(t ai.Token) {
	text := t.Text
	if text == "" && len(t.Data) > 0 {
		text = string(t.Data)
	}

	var last *IterationPart
	if len(i.Parts) > 0 {
		last = &i.Parts[len(i.Parts)-1]
	}

	switch t.Type {
	case ai.TokenTypeText:
		if last != nil && last.Type == IterationTypeResponse {
			last.Response.AppendToken(t)
		} else {
			i.Parts = append(i.Parts, IterationPart{
				Type:     IterationTypeResponse,
				Response: &ai.AIResponse{Text: text, OutputTokens: t.TokenUsage},
			})
		}
	case ai.TokenTypeErr:
		i.Parts = append(i.Parts, IterationPart{
			Type: IterationTypeToolError,
			ToolResp: &ToolResponse{
				Err: t.Err,
			},
		})
	case ai.TokenTypeThought:
		if last != nil && last.Type == IterationTypeResponse {
			last.Response.AppendToken(t)
		} else {
			i.Parts = append(i.Parts, IterationPart{
				Type:     IterationTypeResponse,
				Response: &ai.AIResponse{Text: text, OutputTokens: t.TokenUsage},
			})
		}
	case ai.TokenTypeToolCall:
		i.Parts = append(i.Parts, IterationPart{
			Type:    IterationTypeToolCall,
			ToolReq: t.ToolCall,
		})
	}
}
