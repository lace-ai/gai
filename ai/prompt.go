package ai

import "strings"

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
