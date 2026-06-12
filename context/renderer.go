package context

import (
	"context"
	"encoding/xml"
	"strings"
)

type Renderer interface {
	Render(ctx context.Context, contextParts []Part) (string, error)
}

type XMLRenderer struct{}

func (r XMLRenderer) Render(ctx context.Context, parts []Part) (string, error) {
	if len(parts) == 0 {
		return "", nil
	}

	var builder strings.Builder
	for _, part := range parts {
		writeXMLPart(ctx, &builder, part)
	}

	return builder.String(), nil
}

func writeXMLPart(ctx context.Context, builder *strings.Builder, part Part) {
	content, err := part.Marshal(ctx)
	if err != nil {
		return
	}
	contentStr := string(content)

	builder.WriteString("<")
	writeEscaped(builder, part.Name())
	builder.WriteString(`>`)
	if contentStr != "" {
		builder.WriteString("\n")
		writeEscaped(builder, contentStr)
		builder.WriteString("\n")
	} else {
		builder.WriteString("\n")
	}
	builder.WriteString("</")
	writeEscaped(builder, part.Name())
	builder.WriteString(">\n")
}

func writeEscaped(builder *strings.Builder, text string) {
	if text == "" {
		return
	}
	_ = xml.EscapeText(builder, []byte(text))
}
