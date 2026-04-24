package loop

import (
	"strings"

	"github.com/HecoAI/gai/ai"
)

type EchoTool struct{}

func NewEchoTool() *EchoTool {
	return &EchoTool{}
}

func (t *EchoTool) Name() string {
	return "echo"
}

func (t *EchoTool) Description() string {
	return "Returns the same text passed in arguments."
}

func (t *EchoTool) Params() string {
	return `{"type":"object","required":["text"],"properties":{"text":{"type":"string","description":"Text to echo back"}}}`
}

type echoArgs struct {
	Text string `json:"text"`
}

func (t *EchoTool) Function(req *ai.ToolCall) *ToolResponse {
	var args echoArgs
	if err := DecodeToolArgs(req, &args); err != nil {
		return &ToolResponse{Err: err}
	}

	text := strings.TrimSpace(args.Text)
	if text == "" {
		text = "(empty)"
	}
	return &ToolResponse{Text: text}
}
