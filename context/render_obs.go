package context

import (
	"context"
	"fmt"

	"github.com/lace-ai/gai"
)

const defaultRendererDebugPreviewChars = 500

type renderObserver struct {
	renderer    string
	debug       gai.DebugSink
	previewSize int
	parts       []map[string]any
}

func newRenderObserver(renderer string, debug gai.DebugSink, previewSize int) *renderObserver {
	if previewSize <= 0 {
		previewSize = defaultRendererDebugPreviewChars
	}
	return &renderObserver{renderer: renderer, debug: debug, previewSize: previewSize}
}

func (o *renderObserver) started(ctx context.Context, partCount int) {
	if !o.enabled() {
		return
	}
	o.emit(ctx, "renderer_render_started", map[string]any{
		"renderer":   o.renderer,
		"part_count": partCount,
	}, nil)
}

func (o *renderObserver) partRendered(ctx context.Context, index int, part Part, node *RenderNode, rendered string) {
	if !o.enabled() {
		return
	}
	fields := map[string]any{
		"renderer":       o.renderer,
		"part_index":     index,
		"part_name":      renderPartName(part),
		"rendered_chars": len(rendered),
	}
	if node != nil {
		fields["node"] = rendererNodeStructure(*node, o.previewSize, o.includeSensitiveData())
	}
	if o.includeSensitiveData() {
		addRendererPreview(fields, "rendered", rendered, o.previewSize)
	}
	o.parts = append(o.parts, fields)
	o.emit(ctx, "renderer_part_rendered", fields, nil)
}

func (o *renderObserver) partFailed(ctx context.Context, index int, part Part, err error) {
	if !o.enabled() {
		return
	}
	fields := map[string]any{
		"renderer":   o.renderer,
		"part_index": index,
		"part_name":  renderPartName(part),
	}
	o.emit(ctx, "renderer_part_failed", fields, err)
}

func (o *renderObserver) finished(ctx context.Context, err error, prompt string) {
	if !o.enabled() {
		return
	}
	fields := map[string]any{
		"renderer":     o.renderer,
		"part_count":   len(o.parts),
		"prompt_chars": len(prompt),
		"structure":    o.parts,
	}
	if o.includeSensitiveData() {
		addRendererPreview(fields, "prompt", prompt, o.previewSize)
	}
	o.emit(ctx, "renderer_render_finished", fields, err)
}

func (o *renderObserver) includeSensitiveData() bool {
	return o.enabled() && o.debug.IncludeSensitiveData()
}

func (o *renderObserver) enabled() bool {
	return o != nil && o.debug != nil
}

func (o *renderObserver) emit(ctx context.Context, name string, fields map[string]any, err error) {
	if o == nil || o.debug == nil {
		return
	}
	o.debug.Emit(ctx, gai.DebugEvent{
		Name:   name,
		Source: "context:" + rendererSourceName(o.renderer),
		Fields: fields,
		Err:    err,
	})
}

func rendererSourceName(renderer string) string {
	if renderer == "xml" {
		return "XMLRenderer"
	}
	return "SimpleRenderer"
}

func renderPartName(part Part) string {
	if part == nil {
		return "<nil>"
	}
	return part.Name()
}

func rendererNodeStructure(node RenderNode, previewSize int, includeContent bool) map[string]any {
	structure := map[string]any{
		"type":        node.Type,
		"value_chars": len(node.Value),
		"child_count": len(node.Children),
	}
	if len(node.Fields) > 0 {
		fields := make([]map[string]any, 0, len(node.Fields))
		for _, field := range node.Fields {
			entry := map[string]any{"key": field.Key, "value_chars": len(field.Value)}
			if includeContent {
				addRendererPreview(entry, "value", field.Value, previewSize)
			}
			fields = append(fields, entry)
		}
		structure["fields"] = fields
	}
	if includeContent && node.Value != "" {
		addRendererPreview(structure, "value", node.Value, previewSize)
	}
	if len(node.Children) > 0 {
		children := make([]map[string]any, 0, len(node.Children))
		for _, child := range node.Children {
			children = append(children, rendererNodeStructure(child, previewSize, includeContent))
		}
		structure["children"] = children
	}
	return structure
}

func addRendererPreview(fields map[string]any, key, content string, previewSize int) {
	runes := []rune(content)
	if len(runes) <= previewSize*2 {
		fields[key] = content
		fields[key+"_mode"] = "full"
		return
	}
	omitted := len(runes) - previewSize*2
	fields[key+"_mode"] = "truncated"
	fields[key+"_head"] = string(runes[:previewSize])
	fields[key+"_omitted"] = fmt.Sprintf("[%d chars omitted]", omitted)
	fields[key+"_tail"] = string(runes[len(runes)-previewSize:])
}
