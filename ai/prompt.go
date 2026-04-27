package ai

import (
	"context"
	"strings"

	"github.com/lace-ai/gai"
)

type Prompt struct {
	Prompt  string
	System  string
	Context string
}

func (p Prompt) CombinedPrompt() string {
	var res strings.Builder
	systemPrompt := p.System
	prompt := p.Prompt
	context := p.Context

	if systemPrompt != "" {
		res.WriteString(systemPrompt)
		res.WriteString("\n\n")
	}
	if context != "" {
		res.WriteString(context)
	}
	if prompt != "" {
		res.WriteString(prompt)
		res.WriteString("\n\n")
	}

	return res.String()
}

// CombinedPromptWithDebug combines the prompt with debug logging
func (p Prompt) CombinedPromptWithDebug(ctx context.Context, debug gai.DebugSink) string {
	combined := p.CombinedPrompt()
	if debug != nil && debug.IncludeSensitiveData() {
		debug.Emit(ctx, gai.DebugEvent{
			Name:   "prompt_combined",
			Source: "ai:Prompt.CombinedPromptWithDebug",
			Fields: map[string]any{
				"combined_prompt": combined,
				"system_length":   len(p.System),
				"context_length":  len(p.Context),
				"prompt_length":   len(p.Prompt),
			},
		})
	}
	return combined
}
