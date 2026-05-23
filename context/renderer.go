package context

import (
	"encoding/xml"
	"strings"

	"github.com/lace-ai/gai/ai"
)

type Renderer interface {
	Render(section Section, parts []Part) string
}

type XMLRenderer struct{}

func (r XMLRenderer) Render(section Section, parts []Part) string {
	if len(parts) == 0 {
		return ""
	}

	var builder strings.Builder
	builder.WriteString("<")
	builder.WriteString(string(section))
	builder.WriteString(">\n")
	for _, part := range parts {
		writeXMLPart(&builder, part, "part")
	}
	builder.WriteString("</")
	builder.WriteString(string(section))
	builder.WriteString(">\n")

	return builder.String()
}

func writeXMLPart(builder *strings.Builder, part Part, tag string) {
	builder.WriteString("<")
	builder.WriteString(tag)
	builder.WriteString(` id="`)
	writeEscaped(builder, part.ID)
	builder.WriteString(`">`)
	if part.Text != "" {
		builder.WriteString("\n")
		writeEscaped(builder, part.Text)
		builder.WriteString("\n")
	}
	for _, child := range part.Children {
		writeXMLPart(builder, child, "item")
	}
	builder.WriteString("</")
	builder.WriteString(tag)
	builder.WriteString(">\n")
}

func writeEscaped(builder *strings.Builder, text string) {
	if text == "" {
		return
	}
	_ = xml.EscapeText(builder, []byte(text))
}

func renderPrompt(renderer Renderer, parts map[Section][]Part) ai.Prompt {
	return ai.Prompt{
		System:  renderer.Render(SectionSystem, parts[SectionSystem]),
		Context: renderer.Render(SectionContext, parts[SectionContext]),
		Prompt:  renderer.Render(SectionUser, parts[SectionUser]),
	}
}
