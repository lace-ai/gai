package context

import (
	"context"
	"encoding/xml"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
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

type (
	XMLRenderer    struct{}
	SimpleRenderer struct{}
)

var (
	_ Renderer = (*XMLRenderer)(nil)
	_ Renderer = (*SimpleRenderer)(nil)
)

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

func (r SimpleRenderer) Render(ctx context.Context, parts []Part) (string, error) {
	if len(parts) == 0 {
		return "", nil
	}

	blocks := make([]string, 0, len(parts))
	for _, part := range parts {
		block, err := renderSimplePart(ctx, part)
		if err != nil {
			return "", err
		}
		if block == "" {
			continue
		}
		blocks = append(blocks, block)
	}

	return strings.Join(blocks, "\n\n"), nil
}

func renderSimplePart(ctx context.Context, part Part) (string, error) {
	if part == nil {
		return "", nil
	}

	switch p := part.(type) {
	case SystemPart:
		return renderSimpleInstructions(ctx, p)
	case MessagePart:
		return renderSimpleMessage(ctx, p.Render)
	}

	node, err := part.Render(ctx)
	if err != nil {
		return "", err
	}
	return renderSimpleNode(node), nil
}

func renderSimpleInstructions(ctx context.Context, part SystemPart) (string, error) {
	var builder strings.Builder
	builder.WriteString("<Instructions>")

	wroteInstruction := false
	for _, instruction := range part.Instructions {
		if instruction == nil {
			continue
		}

		body, err := renderSimpleInstruction(ctx, instruction)
		if err != nil {
			return "", err
		}
		if body == "" {
			continue
		}

		builder.WriteString("\n\n")
		builder.WriteString(formatSimpleInstructionLabel(instruction.Name()))
		builder.WriteString(":\n")
		builder.WriteString(body)
		wroteInstruction = true
	}

	if wroteInstruction {
		builder.WriteString("\n\n")
	} else {
		builder.WriteString("\n")
	}
	builder.WriteString("</Instructions>")
	return builder.String(), nil
}

func renderSimpleInstruction(ctx context.Context, part Part) (string, error) {
	switch p := part.(type) {
	case TextPart:
		return p.Content, nil
	case MessagePart:
		node, err := p.Render(ctx)
		if err != nil {
			return "", err
		}
		return renderSimpleMessageNode(node), nil
	}

	node, err := part.Render(ctx)
	if err != nil {
		return "", err
	}
	return renderSimpleNodeBody(node), nil
}

func renderSimpleMessage(ctx context.Context, render func(context.Context) (RenderNode, error)) (string, error) {
	node, err := render(ctx)
	if err != nil {
		return "", err
	}
	return renderSimpleMessageNode(node), nil
}

func renderSimpleNode(node RenderNode) string {
	switch node.Type {
	case "history":
		return renderSimpleHistory(node)
	case string(RoleUser), string(RoleAssistant), string(RoleTool), string(RoleSystem), "summary", "message":
		return renderSimpleMessageNode(node)
	default:
		return renderSimpleNodeBody(node)
	}
}

func renderSimpleHistory(node RenderNode) string {
	var builder strings.Builder
	builder.WriteString("<history>")
	if len(node.Children) > 0 {
		builder.WriteString("\n")
		for i, child := range node.Children {
			if i > 0 {
				builder.WriteString("\n")
			}
			builder.WriteString(renderSimpleMessageNode(child))
		}
		builder.WriteString("\n")
	}
	builder.WriteString("</history>")
	return builder.String()
}

func renderSimpleMessageNode(node RenderNode) string {
	return formatSimpleLine(simpleMessageLabel(node.Type), renderSimpleNodeBody(node))
}

func simpleMessageLabel(nodeType string) string {
	switch nodeType {
	case string(RoleTool):
		return "tool res"
	default:
		return nodeType
	}
}

func formatSimpleLine(label, body string) string {
	if body == "" {
		return label + ":"
	}
	if strings.Contains(body, "\n") {
		return label + ":\n" + body
	}
	return label + ": " + body
}

func renderSimpleNodeBody(node RenderNode) string {
	switch node.Type {
	case ContentTypeText:
		return node.Value
	case ContentTypeToolCall:
		return simpleNodeChildValue(node, "arguments")
	case ContentTypeToolResult:
		return simpleNodeChildValue(node, "result")
	case ContentTypeToolResultErr:
		return simpleNodeChildValue(node, "error")
	}

	parts := make([]string, 0, len(node.Children)+1)
	if node.Value != "" {
		parts = append(parts, node.Value)
	}
	for _, child := range node.Children {
		body := renderSimpleNodeBody(child)
		if body != "" {
			parts = append(parts, body)
		}
	}
	return strings.Join(parts, "\n")
}

func simpleNodeChildValue(node RenderNode, childType string) string {
	for _, child := range node.Children {
		if child.Type == childType {
			return child.Value
		}
	}
	return node.Value
}

func formatSimpleInstructionLabel(name string) string {
	if name == "" {
		return "Instruction"
	}

	parts := strings.FieldsFunc(name, func(r rune) bool {
		return r == '_' || r == '-' || unicode.IsSpace(r)
	})
	if len(parts) == 0 {
		return "Instruction"
	}

	for i, part := range parts {
		r, size := utf8.DecodeRuneInString(part)
		if r == utf8.RuneError && size == 0 {
			continue
		}
		parts[i] = string(unicode.ToUpper(r)) + strings.ToLower(part[size:])
	}
	return strings.Join(parts, " ")
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
