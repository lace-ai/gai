package context

import (
	"context"
	"encoding/xml"
	"fmt"
	"strings"
	"unicode"
)

type Renderer interface {
	Render(ctx context.Context, contextParts []Part) (string, error)
}

type RenderField struct {
	Key   string
	Value string
}

type RenderNode struct {
	Type     string
	Fields   []RenderField
	Value    string
	Children []RenderNode
}

type XMLRenderer struct{}

func (r XMLRenderer) Render(ctx context.Context, parts []Part) (string, error) {
	if len(parts) == 0 {
		return "", nil
	}

	var builder strings.Builder
	for _, part := range parts {
		err := writeXMLPart(ctx, &builder, part)
		if err != nil {
			return "", err
		}
	}

	return builder.String(), nil
}

func writeXMLPart(ctx context.Context, builder *strings.Builder, part Part) error {
	if part == nil {
		return nil
	}
	node, err := part.Render(ctx)
	if err != nil {
		return err
	}
	return writeXMLNode(builder, node, 0)
}

func writeXMLNode(builder *strings.Builder, node RenderNode, depth int) error {
	if !validXMLName(node.Type) {
		return fmt.Errorf("invalid xml tag name: %q", node.Type)
	}
	indent(builder, depth)
	builder.WriteString("<")
	builder.WriteString(node.Type)
	for _, field := range node.Fields {
		if !validXMLName(field.Key) {
			return fmt.Errorf("invalid xml attribute name: %q", field.Key)
		}
		builder.WriteString(" ")
		builder.WriteString(field.Key)
		builder.WriteString(`="`)
		writeEscaped(builder, field.Value)
		builder.WriteString(`"`)
	}
	builder.WriteString(">")
	if node.Value != "" {
		builder.WriteString("\n")
		indent(builder, depth+1)
		writeEscaped(builder, node.Value)
		builder.WriteString("\n")
	}
	if len(node.Children) > 0 {
		builder.WriteString("\n")
	}
	for _, child := range node.Children {
		if err := writeXMLNode(builder, child, depth+1); err != nil {
			return err
		}
	}
	if node.Value == "" && len(node.Children) == 0 {
		builder.WriteString("\n")
	}
	indent(builder, depth)
	builder.WriteString("</")
	builder.WriteString(node.Type)
	builder.WriteString(">\n")
	return nil
}

func indent(builder *strings.Builder, depth int) {
	for range depth {
		builder.WriteString("  ")
	}
}

func writeEscaped(builder *strings.Builder, text string) {
	if text == "" {
		return
	}
	_ = xml.EscapeText(builder, []byte(text))
}

func validXMLName(name string) bool {
	if name == "" {
		return false
	}
	for i, r := range name {
		if i == 0 {
			if r != '_' && !unicode.IsLetter(r) {
				return false
			}
			continue
		}
		if r != '_' && r != '-' && r != '.' && !unicode.IsLetter(r) && !unicode.IsDigit(r) {
			return false
		}
	}
	return true
}
