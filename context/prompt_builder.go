package context

import (
	stdcontext "context"
	"encoding/xml"
	"fmt"
	"strings"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
)

type Section string

const (
	SectionSystem  Section = "system"
	SectionContext Section = "context"
	SectionUser    Section = "user"
)

type EntryKind string

const (
	EntryKindPart   EntryKind = "part"
	EntryKindSource EntryKind = "source"
)

type Part struct {
	ID       string
	Text     string
	Tokens   int
	Required bool
	Meta     map[string]any
}

func (p Part) WithTokens(tokens int) Part {
	p.Tokens = tokens
	return p
}

func (p Part) RequiredPart() Part {
	p.Required = true
	return p
}

func (p Part) OptionalPart() Part {
	p.Required = false
	return p
}

type Source interface {
	BuildParts(ctx stdcontext.Context, view PromptView) ([]Part, error)
}

type SourceFunc func(ctx stdcontext.Context, view PromptView) ([]Part, error)

func (f SourceFunc) BuildParts(ctx stdcontext.Context, view PromptView) ([]Part, error) {
	return f(ctx, view)
}

type PromptView interface {
	Conversation() Conversation
	Entries() []EntryView
	SectionEntries(section Section) []EntryView
	Entry(id string) (EntryView, bool)
}

type EntryView struct {
	ID       string
	Section  Section
	Kind     EntryKind
	Required bool
	Tokens   int
	Text     string
	Meta     map[string]any
}

type BuildTrace struct {
	Entries []BuildTraceEntry
	Parts   map[Section][]Part
}

type BuildTraceEntry struct {
	ID       string
	Section  Section
	Kind     EntryKind
	Status   string
	Required bool
	Parts    []Part
	Err      error
}

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
		builder.WriteString(`<part id="`)
		writeEscaped(&builder, part.ID)
		builder.WriteString(`">`)
		if part.Text != "" {
			builder.WriteString("\n")
			writeEscaped(&builder, part.Text)
			builder.WriteString("\n")
		}
		builder.WriteString("</part>\n")
	}
	builder.WriteString("</")
	builder.WriteString(string(section))
	builder.WriteString(">\n")

	return builder.String()
}

func writeEscaped(builder *strings.Builder, text string) {
	if text == "" {
		return
	}
	_ = xml.EscapeText(builder, []byte(text))
}

type PromptBuilder interface {
	BuildPrompt(ctx stdcontext.Context, conv Conversation) (ai.Prompt, error)
}

type Builder struct {
	renderer Renderer
	debug    gai.DebugSink
	entries  []builderEntry
	trace    BuildTrace
}

type builderEntry struct {
	id       string
	section  Section
	kind     EntryKind
	required bool
	tokens   int
	text     string
	meta     map[string]any
	source   Source
}

type EntryOption func(*builderEntry)

func Required() EntryOption {
	return func(entry *builderEntry) {
		entry.required = true
	}
}

// Optional is the default a no-op, but can be used to override a Required option.
func Optional() EntryOption {
	return func(entry *builderEntry) {
		entry.required = false
	}
}

func Tokens(tokens int) EntryOption {
	return func(entry *builderEntry) {
		entry.tokens = tokens
	}
}

func Meta(key string, value any) EntryOption {
	return func(entry *builderEntry) {
		if entry.meta == nil {
			entry.meta = map[string]any{}
		}
		entry.meta[key] = value
	}
}

func NewPromptBuilder() *Builder {
	return &Builder{
		renderer: XMLRenderer{},
	}
}

func NewPart(id, text string, opts ...EntryOption) Part {
	entry := builderEntry{
		id:   id,
		kind: EntryKindPart,
		text: text,
	}
	applyOptions(&entry, opts)
	return entry.part()
}

func (b *Builder) Renderer(renderer Renderer) *Builder {
	if renderer != nil {
		b.renderer = renderer
	}
	return b
}

func (b *Builder) Debug(debug gai.DebugSink) *Builder {
	b.debug = debug
	return b
}

func (b *Builder) System(id, text string, opts ...EntryOption) *Builder {
	return b.Part(SectionSystem, id, text, opts...)
}

func (b *Builder) Context(id, text string, opts ...EntryOption) *Builder {
	return b.Part(SectionContext, id, text, opts...)
}

func (b *Builder) User(id, text string, opts ...EntryOption) *Builder {
	return b.Part(SectionUser, id, text, opts...)
}

func (b *Builder) Part(section Section, id, text string, opts ...EntryOption) *Builder {
	entry := builderEntry{
		id:      id,
		section: section,
		kind:    EntryKindPart,
		text:    text,
	}
	applyOptions(&entry, opts)
	b.entries = append(b.entries, entry)
	return b
}

func (b *Builder) Source(section Section, id string, source Source, opts ...EntryOption) *Builder {
	entry := builderEntry{
		id:      id,
		section: section,
		kind:    EntryKindSource,
		source:  source,
	}
	applyOptions(&entry, opts)
	b.entries = append(b.entries, entry)
	return b
}

func (b *Builder) LastTrace() BuildTrace {
	return cloneTrace(b.trace)
}

func (b *Builder) BuildPrompt(ctx stdcontext.Context, conv Conversation) (ai.Prompt, error) {
	if b == nil {
		return ai.Prompt{}, ErrPromptBuilderNil
	}
	renderer := b.renderer
	if renderer == nil {
		renderer = XMLRenderer{}
	}
	if err := b.validate(); err != nil {
		b.trace = BuildTrace{}
		b.emit(ctx, "prompt_build_failed", map[string]any{"error": err.Error()}, err)
		return ai.Prompt{}, err
	}

	view := newBuilderView(conv, b.entries)
	trace := BuildTrace{Parts: map[Section][]Part{}}
	parts := map[Section][]Part{
		SectionSystem:  {},
		SectionContext: {},
		SectionUser:    {},
	}
	partIDs := map[string]Section{}

	b.emit(ctx, "prompt_build_started", map[string]any{"entries": len(b.entries)}, nil)
	for _, entry := range b.entries {
		traceEntry := BuildTraceEntry{
			ID:       entry.id,
			Section:  entry.section,
			Kind:     entry.kind,
			Required: entry.required,
		}

		switch entry.kind {
		case EntryKindPart:
			part := entry.part()
			if err := validatePartIDs(partIDs, entry.section, []Part{part}); err != nil {
				traceEntry.Err = err
				traceEntry.Status = "error"
				trace.Entries = append(trace.Entries, traceEntry)
				b.trace = finalizeTrace(trace, parts)
				b.emitEntry(ctx, "prompt_entry_error", traceEntry)
				return ai.Prompt{}, err
			}
			parts[entry.section] = append(parts[entry.section], part)
			traceEntry.Status = "emitted"
			traceEntry.Parts = []Part{part}
			b.emitEntry(ctx, "prompt_entry_emitted", traceEntry)
		case EntryKindSource:
			if entry.source == nil {
				err := fmt.Errorf("%w: section %s source %q is nil", ErrPromptSource, entry.section, entry.id)
				traceEntry.Err = err
				if entry.required {
					traceEntry.Status = "error"
					trace.Entries = append(trace.Entries, traceEntry)
					b.trace = finalizeTrace(trace, parts)
					b.emitEntry(ctx, "prompt_source_error", traceEntry)
					return ai.Prompt{}, err
				}
				traceEntry.Status = "skipped"
				trace.Entries = append(trace.Entries, traceEntry)
				b.emitEntry(ctx, "prompt_source_skipped", traceEntry)
				continue
			}

			sourceParts, err := entry.source.BuildParts(ctx, view)
			if err != nil {
				traceEntry.Err = err
				if entry.required {
					wrapped := fmt.Errorf("%w: section %s source %q: %w", ErrPromptSource, entry.section, entry.id, err)
					traceEntry.Err = wrapped
					traceEntry.Status = "error"
					trace.Entries = append(trace.Entries, traceEntry)
					b.trace = finalizeTrace(trace, parts)
					b.emitEntry(ctx, "prompt_source_error", traceEntry)
					return ai.Prompt{}, wrapped
				}
				traceEntry.Status = "skipped"
				trace.Entries = append(trace.Entries, traceEntry)
				b.emitEntry(ctx, "prompt_source_skipped", traceEntry)
				continue
			}
			if entry.required {
				for i := range sourceParts {
					sourceParts[i].Required = true
				}
			}
			if err := validatePartIDs(partIDs, entry.section, sourceParts); err != nil {
				traceEntry.Err = err
				if entry.required {
					traceEntry.Status = "error"
					trace.Entries = append(trace.Entries, traceEntry)
					b.trace = finalizeTrace(trace, parts)
					b.emitEntry(ctx, "prompt_source_error", traceEntry)
					return ai.Prompt{}, err
				}
				traceEntry.Status = "skipped"
				trace.Entries = append(trace.Entries, traceEntry)
				b.emitEntry(ctx, "prompt_source_skipped", traceEntry)
				continue
			}
			parts[entry.section] = append(parts[entry.section], sourceParts...)
			traceEntry.Status = "emitted"
			traceEntry.Parts = cloneParts(sourceParts)
			b.emitEntry(ctx, "prompt_source_emitted", traceEntry)
		}

		trace.Entries = append(trace.Entries, traceEntry)
	}

	trace = finalizeTrace(trace, parts)
	b.trace = trace
	prompt := ai.Prompt{
		System:  renderer.Render(SectionSystem, parts[SectionSystem]),
		Context: renderer.Render(SectionContext, parts[SectionContext]),
		Prompt:  renderer.Render(SectionUser, parts[SectionUser]),
	}
	b.emit(ctx, "prompt_build_completed", map[string]any{
		"system_parts":  len(parts[SectionSystem]),
		"context_parts": len(parts[SectionContext]),
		"user_parts":    len(parts[SectionUser]),
	}, nil)

	return prompt, nil
}

func (b *Builder) validate() error {
	seen := map[string]Section{}
	for _, entry := range b.entries {
		if strings.TrimSpace(entry.id) == "" {
			return fmt.Errorf("%w: entry ID is empty", ErrPromptEntryID)
		}
		if !validSection(entry.section) {
			return fmt.Errorf("%w: entry %q uses unknown section %q", ErrPromptEntryID, entry.id, entry.section)
		}
		if section, ok := seen[entry.id]; ok {
			return fmt.Errorf("%w: duplicate entry ID %q in sections %s and %s", ErrPromptEntryID, entry.id, section, entry.section)
		}
		seen[entry.id] = entry.section
	}
	return nil
}

func validatePartIDs(seen map[string]Section, section Section, parts []Part) error {
	pending := make([]string, 0, len(parts))
	local := map[string]struct{}{}
	for _, part := range parts {
		if strings.TrimSpace(part.ID) == "" {
			return fmt.Errorf("%w: emitted part ID is empty in section %s", ErrPromptEntryID, section)
		}
		if _, ok := local[part.ID]; ok {
			return fmt.Errorf("%w: duplicate emitted part ID %q in section %s", ErrPromptEntryID, part.ID, section)
		}
		if previousSection, ok := seen[part.ID]; ok {
			return fmt.Errorf("%w: duplicate emitted part ID %q in sections %s and %s", ErrPromptEntryID, part.ID, previousSection, section)
		}
		local[part.ID] = struct{}{}
		pending = append(pending, part.ID)
	}
	for _, id := range pending {
		seen[id] = section
	}
	return nil
}

func validSection(section Section) bool {
	switch section {
	case SectionSystem, SectionContext, SectionUser:
		return true
	default:
		return false
	}
}

func (b *Builder) emitEntry(ctx stdcontext.Context, name string, entry BuildTraceEntry) {
	fields := map[string]any{
		"id":       entry.ID,
		"section":  string(entry.Section),
		"kind":     string(entry.Kind),
		"status":   entry.Status,
		"required": entry.Required,
		"parts":    len(entry.Parts),
	}
	if entry.Err != nil {
		fields["error"] = entry.Err.Error()
	}
	if b.debug != nil && b.debug.IncludeSensitiveData() {
		fields["emitted_parts"] = entry.Parts
	}
	b.emit(ctx, name, fields, entry.Err)
}

func (b *Builder) emit(ctx stdcontext.Context, name string, fields map[string]any, err error) {
	if b.debug == nil {
		return
	}
	b.debug.Emit(ctx, gai.DebugEvent{
		Name:   name,
		Source: "context:PromptBuilder.BuildPrompt",
		Fields: fields,
		Err:    err,
	})
}

func applyOptions(entry *builderEntry, opts []EntryOption) {
	for _, opt := range opts {
		if opt != nil {
			opt(entry)
		}
	}
}

func (e builderEntry) part() Part {
	return Part{
		ID:       e.id,
		Text:     e.text,
		Tokens:   e.tokens,
		Required: e.required,
		Meta:     cloneMeta(e.meta),
	}
}

func (e builderEntry) view() EntryView {
	return EntryView{
		ID:       e.id,
		Section:  e.section,
		Kind:     e.kind,
		Required: e.required,
		Tokens:   e.tokens,
		Text:     e.text,
		Meta:     cloneMeta(e.meta),
	}
}

type builderView struct {
	conv    Conversation
	entries []EntryView
	byID    map[string]EntryView
}

func newBuilderView(conv Conversation, entries []builderEntry) builderView {
	view := builderView{
		conv:    conv,
		entries: make([]EntryView, 0, len(entries)),
		byID:    map[string]EntryView{},
	}
	for _, entry := range entries {
		entryView := entry.view()
		view.entries = append(view.entries, entryView)
		view.byID[entryView.ID] = entryView
	}
	return view
}

func (v builderView) Conversation() Conversation {
	return v.conv
}

func (v builderView) Entries() []EntryView {
	return cloneEntryViews(v.entries)
}

func (v builderView) SectionEntries(section Section) []EntryView {
	entries := make([]EntryView, 0)
	for _, entry := range v.entries {
		if entry.Section == section {
			entries = append(entries, entry.clone())
		}
	}
	return entries
}

func (v builderView) Entry(id string) (EntryView, bool) {
	entry, ok := v.byID[id]
	if !ok {
		return EntryView{}, false
	}
	return entry.clone(), true
}

func (e EntryView) clone() EntryView {
	e.Meta = cloneMeta(e.Meta)
	return e
}

func finalizeTrace(trace BuildTrace, parts map[Section][]Part) BuildTrace {
	trace.Parts = map[Section][]Part{
		SectionSystem:  cloneParts(parts[SectionSystem]),
		SectionContext: cloneParts(parts[SectionContext]),
		SectionUser:    cloneParts(parts[SectionUser]),
	}
	return trace
}

func cloneTrace(trace BuildTrace) BuildTrace {
	cloned := BuildTrace{
		Entries: make([]BuildTraceEntry, len(trace.Entries)),
		Parts:   map[Section][]Part{},
	}
	for i, entry := range trace.Entries {
		cloned.Entries[i] = entry
		cloned.Entries[i].Parts = cloneParts(entry.Parts)
	}
	for section, parts := range trace.Parts {
		cloned.Parts[section] = cloneParts(parts)
	}
	return cloned
}

func cloneEntryViews(entries []EntryView) []EntryView {
	cloned := make([]EntryView, len(entries))
	for i, entry := range entries {
		cloned[i] = entry.clone()
	}
	return cloned
}

func cloneParts(parts []Part) []Part {
	cloned := make([]Part, len(parts))
	for i, part := range parts {
		cloned[i] = part
		cloned[i].Meta = cloneMeta(part.Meta)
	}
	return cloned
}

func cloneMeta(meta map[string]any) map[string]any {
	if len(meta) == 0 {
		return nil
	}
	cloned := make(map[string]any, len(meta))
	for key, value := range meta {
		cloned[key] = value
	}
	return cloned
}
