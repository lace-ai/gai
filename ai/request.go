package ai

import "strings"

type AIRequest struct {
	Prompt       string
	SystemPrompt string
	Context      string
	MaxTokens    int
}

func (r AIRequest) CombinedPrompt() string {
	var res strings.Builder
	systemPrompt := r.SystemPrompt
	prompt := r.Prompt
	context := r.Context

	if systemPrompt != "" {
		res.WriteString(systemPrompt)
		res.WriteString("\n\n")
	}
	if prompt != "" {
		res.WriteString(prompt)
		res.WriteString("\n\n")
	}
	if context != "" {
		res.WriteString(context)
	}

	return res.String()
}
