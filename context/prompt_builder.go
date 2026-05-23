package context

import (
	stdcontext "context"
	"fmt"
	"maps"
	"math"
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

const unlimitedTokens = math.MaxInt

type Part struct {
	ID       string
	Text     string
	Tokens   int
	Required bool
	Meta     map[string]any
	Children []Part
}

func (p Part) tokenCount() int {
	if p.Tokens > 0 {
		return p.Tokens
	}
	tokens := 0
	for _, child := range p.Children {
		tokens += child.tokenCount()
	}
	return tokens
}

type PromptBudget struct {
	Tokenizer                  ai.Tokenizer
	ContextWindowTokens        int
	ReservedOutputTokens       int
	RenderOverheadReserveRatio float64
	Summarizer                 Summarizer
}

func (b PromptBudget) promptLimit() int {
	limit := b.ContextWindowTokens - b.ReservedOutputTokens
	if limit < 0 {
		return 0
	}
	return limit
}

type SourceBudget struct {
	Tokenizer                  ai.Tokenizer
	MaxTokens                  int
	RemainingPromptTokens      int
	Required                   bool
	RenderOverheadReserveRatio float64
	Summarizer                 Summarizer
}

func (b SourceBudget) ContentLimit() int {
	if b.MaxTokens == unlimitedTokens {
		return unlimitedTokens
	}
	limit := b.MaxTokens - renderOverheadTokens(b.MaxTokens, b.RenderOverheadReserveRatio)
	if limit < 0 {
		return 0
	}
	return limit
}

type Source interface {
	BuildParts(ctx stdcontext.Context, view PromptView, budget SourceBudget) ([]Part, error)
}

type SourceFunc func(ctx stdcontext.Context, view PromptView, budget SourceBudget) ([]Part, error)

func (f SourceFunc) BuildParts(ctx stdcontext.Context, view PromptView, budget SourceBudget) ([]Part, error) {
	return f(ctx, view, budget)
}

type PromptView interface {
	Conversation() Conversation
	Entries() []EntryView
	SectionEntries(section Section) []EntryView
	Entry(id string) (EntryView, bool)
}

type EntryView struct {
	ID        string
	Section   Section
	Kind      EntryKind
	Required  bool
	Tokens    int
	SourceCap int
	Text      string
	Meta      map[string]any
}

type BuildTrace struct {
	Entries []BuildTraceEntry
	Parts   map[Section][]Part
}

type BuildTraceEntry struct {
	ID              string
	Section         Section
	Kind            EntryKind
	Status          string
	Reason          string
	Required        bool
	Parts           []Part
	EntryTokens     int
	PromptTokens    int
	AvailableTokens int
	Err             error
}

type PromptBuilder interface {
	BuildPrompt(ctx stdcontext.Context, conv Conversation) (ai.Prompt, error)
}

type Builder struct {
	renderer Renderer
	debug    gai.DebugSink
	budget   *PromptBudget
	entries  []builderEntry
	trace    BuildTrace
}

type builderEntry struct {
	id           string
	section      Section
	kind         EntryKind
	required     bool
	tokens       int
	sourceCap    int
	hasSourceCap bool
	text         string
	meta         map[string]any
	source       Source
}

type EntryOption func(*builderEntry)

func Required() EntryOption {
	return func(entry *builderEntry) {
		entry.required = true
	}
}

// Optional is the default (a no-op), but can be used to override a previous Required option.
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

func SourceTokenCap(tokens int) EntryOption {
	return func(entry *builderEntry) {
		entry.sourceCap = tokens
		entry.hasSourceCap = true
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

func NewPartGroup(id string, children []Part, opts ...EntryOption) Part {
	entry := builderEntry{
		id:   id,
		kind: EntryKindPart,
	}
	applyOptions(&entry, opts)
	part := entry.part()
	part.Children = cloneParts(children)
	if part.Tokens == 0 {
		for _, child := range children {
			part.Tokens += child.tokenCount()
		}
	}
	return part
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

func (b *Builder) Budget(budget PromptBudget) *Builder {
	b.budget = &budget
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
	state := newPromptBuildState(renderer)

	b.emit(ctx, "prompt_build_started", map[string]any{"entries": len(b.entries)}, nil)
	for _, entry := range orderedEntries(b.entries) {
		traceEntry := BuildTraceEntry{
			ID:       entry.id,
			Section:  entry.section,
			Kind:     entry.kind,
			Required: entry.required,
		}

		switch entry.kind {
		case EntryKindPart:
			part := entry.part()
			if b.budget != nil && part.Tokens == 0 && part.Text != "" {
				tokens, err := b.budget.Tokenizer.CountTokens(ctx, part.Text)
				if err != nil {
					return ai.Prompt{}, b.failEntry(ctx, &state, "prompt_entry_error", traceEntry, err)
				}
				part.Tokens = tokens
			}
			if err := b.admitEntryParts(ctx, &state, entry, traceEntry, []Part{part}, entryAdmissionOptions{
				eventPrefix:       "prompt_entry",
				summarizeOptional: true,
			}); err != nil {
				return ai.Prompt{}, err
			}
		case EntryKindSource:
			if entry.source == nil {
				err := fmt.Errorf("%w: section %s source %q is nil", ErrPromptSource, entry.section, entry.id)
				traceEntry.Err = err
				if entry.required {
					return ai.Prompt{}, b.failEntry(ctx, &state, "prompt_source_error", traceEntry, err)
				}
				traceEntry.Status = "skipped"
				traceEntry.Reason = "nil_source"
				state.record(traceEntry)
				b.emitEntry(ctx, "prompt_source_skipped", traceEntry)
				continue
			}

			sourceBudget := b.sourceBudget(state.promptTokens, entry)
			traceEntry.AvailableTokens = sourceBudget.MaxTokens
			sourceParts, err := entry.source.BuildParts(ctx, view, sourceBudget)
			if err != nil {
				traceEntry.Err = err
				if entry.required {
					wrapped := fmt.Errorf("%w: section %s source %q: %w", ErrPromptSource, entry.section, entry.id, err)
					traceEntry.Err = wrapped
					return ai.Prompt{}, b.failEntry(ctx, &state, "prompt_source_error", traceEntry, wrapped)
				}
				traceEntry.Status = "skipped"
				traceEntry.Reason = "optional_source_error"
				state.record(traceEntry)
				b.emitEntry(ctx, "prompt_source_skipped", traceEntry)
				continue
			}
			if entry.required {
				markRequired(sourceParts)
			}
			if err := b.admitEntryParts(ctx, &state, entry, traceEntry, sourceParts, entryAdmissionOptions{
				eventPrefix:         "prompt_source",
				optionalIDConflict:  true,
				includeDroppedParts: true,
			}); err != nil {
				return ai.Prompt{}, err
			}
		}
	}

	trace := state.finalTrace()
	b.trace = trace
	prompt := renderPrompt(renderer, state.parts)
	if b.budget != nil {
		if state.promptTokens > b.budget.promptLimit() {
			err := promptBudgetError("prompt", state.promptTokens, b.budget.promptLimit())
			b.emit(ctx, "prompt_build_failed", map[string]any{
				"error":            err.Error(),
				"prompt_tokens":    state.promptTokens,
				"available_tokens": b.budget.promptLimit(),
			}, err)
			return ai.Prompt{}, err
		}
	}
	b.emit(ctx, "prompt_build_completed", map[string]any{
		"system_parts":  len(state.parts[SectionSystem]),
		"context_parts": len(state.parts[SectionContext]),
		"user_parts":    len(state.parts[SectionUser]),
	}, nil)

	return prompt, nil
}

type promptBuildState struct {
	renderer     Renderer
	trace        BuildTrace
	parts        map[Section][]Part
	partIDs      map[string]Section
	promptTokens int
}

func newPromptBuildState(renderer Renderer) promptBuildState {
	return promptBuildState{
		renderer: renderer,
		trace:    BuildTrace{Parts: map[Section][]Part{}},
		parts: map[Section][]Part{
			SectionSystem:  {},
			SectionContext: {},
			SectionUser:    {},
		},
		partIDs: map[string]Section{},
	}
}

func (s *promptBuildState) finalTrace() BuildTrace {
	return finalizeTrace(s.trace, s.parts)
}

func (s *promptBuildState) record(entry BuildTraceEntry) {
	s.trace.Entries = append(s.trace.Entries, entry)
}

func (s *promptBuildState) nextParts(section Section, parts []Part) []Part {
	return append(cloneParts(s.parts[section]), parts...)
}

func (s *promptBuildState) appendParts(section Section, parts []Part, partIDs map[string]Section, promptTokens int) {
	s.parts[section] = parts
	s.partIDs = partIDs
	s.promptTokens = promptTokens
}

func (s *promptBuildState) dropOptionalContextFor(parts map[Section][]Part, promptTokens int) {
	s.parts = parts
	s.partIDs = rebuildPartIDs(s.parts)
	s.promptTokens = promptTokens
	s.trace.Entries = markDroppedOptionalContextEntries(s.trace.Entries)
}

type entryAdmissionOptions struct {
	eventPrefix         string
	summarizeOptional   bool
	optionalIDConflict  bool
	includeDroppedParts bool
}

func (b *Builder) failEntry(ctx stdcontext.Context, state *promptBuildState, event string, entry BuildTraceEntry, err error) error {
	entry.Err = err
	entry.Status = "error"
	state.record(entry)
	b.trace = state.finalTrace()
	b.emitEntry(ctx, event, entry)
	return err
}

func (b *Builder) admitEntryParts(ctx stdcontext.Context, state *promptBuildState, entry builderEntry, traceEntry BuildTraceEntry, entryParts []Part, opts entryAdmissionOptions) error {
	nextPartIDs := clonePartIDMap(state.partIDs)
	if err := validatePartIDs(nextPartIDs, entry.section, entryParts); err != nil {
		traceEntry.Err = err
		if opts.optionalIDConflict && !entry.required {
			traceEntry.Status = "skipped"
			traceEntry.Reason = "duplicate_part_id"
			state.record(traceEntry)
			b.emitEntry(ctx, opts.eventPrefix+"_skipped", traceEntry)
			return nil
		}
		return b.failEntry(ctx, state, opts.eventPrefix+"_error", traceEntry, err)
	}

	next := state.nextParts(entry.section, entryParts)
	ok, count, available, err := b.partsFit(state.parts, entry.section, next)
	if err != nil {
		return b.failEntry(ctx, state, opts.eventPrefix+"_error", traceEntry, err)
	}
	if ok {
		state.appendParts(entry.section, next, nextPartIDs, count)
		traceEntry.Status = "emitted"
		traceEntry.Parts = cloneParts(entryParts)
		setTraceTokens(&traceEntry, partsTokenCount(entryParts), count)
		state.record(traceEntry)
		b.emitEntry(ctx, opts.eventPrefix+"_emitted", traceEntry)
		return nil
	}

	entryTokens := partsTokenCount(entryParts)
	setTraceTokens(&traceEntry, entryTokens, count)
	traceEntry.AvailableTokens = available
	if entry.required {
		cleanedParts, ok, retryCount, retryAvailable, err := b.partsFitAfterDroppingOptionalContext(state.parts, entry.section, next)
		if err != nil {
			return b.failEntry(ctx, state, opts.eventPrefix+"_error", traceEntry, err)
		}
		if ok {
			state.dropOptionalContextFor(cleanedParts, retryCount)
			traceEntry.Reason = "dropped_optional_context"
			traceEntry.Status = "emitted"
			traceEntry.Parts = cloneParts(entryParts)
			setTraceTokens(&traceEntry, entryTokens, retryCount)
			traceEntry.AvailableTokens = retryAvailable
			state.record(traceEntry)
			b.emitEntry(ctx, opts.eventPrefix+"_emitted", traceEntry)
			return nil
		}
		err = promptBudgetError(entry.id, count, available)
		traceEntry.Reason = "required_over_budget"
		return b.failEntry(ctx, state, opts.eventPrefix+"_error", traceEntry, err)
	}

	traceEntry.Status = "dropped"
	traceEntry.Reason = "optional_over_budget"
	if opts.summarizeOptional && len(entryParts) == 1 {
		summarizedPart, ok, summaryCount, summaryPromptCount, summaryAvailable, err := b.summarizeOptionalPart(ctx, state.parts, entry, entryParts[0], state.promptTokens)
		if err != nil {
			return b.failEntry(ctx, state, opts.eventPrefix+"_error", traceEntry, err)
		}
		if ok {
			nextSummaryPartIDs := clonePartIDMap(state.partIDs)
			if err := validatePartIDs(nextSummaryPartIDs, entry.section, []Part{summarizedPart}); err != nil {
				return b.failEntry(ctx, state, opts.eventPrefix+"_error", traceEntry, err)
			}
			state.appendParts(entry.section, state.nextParts(entry.section, []Part{summarizedPart}), nextSummaryPartIDs, summaryPromptCount)
			traceEntry.Status = "summarized"
			traceEntry.Reason = "optional_summarized"
			traceEntry.Parts = []Part{summarizedPart}
			setTraceTokens(&traceEntry, summaryCount, summaryPromptCount)
			traceEntry.AvailableTokens = summaryAvailable
			state.record(traceEntry)
			b.emitEntry(ctx, opts.eventPrefix+"_summarized", traceEntry)
			return nil
		}
	}
	if opts.includeDroppedParts {
		traceEntry.Parts = cloneParts(entryParts)
	}
	state.record(traceEntry)
	b.emitEntry(ctx, opts.eventPrefix+"_dropped", traceEntry)
	return nil
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
		if entry.hasSourceCap && entry.sourceCap < 0 {
			return fmt.Errorf("%w: source %q has negative token cap", ErrPromptBudget, entry.id)
		}
		seen[entry.id] = entry.section
	}
	if b.budget != nil {
		if b.budget.Tokenizer == nil {
			return ErrTokenizerNotFound
		}
		if b.budget.promptLimit() <= 0 {
			return fmt.Errorf("%w: prompt token limit must be positive", ErrPromptBudget)
		}
	}
	return nil
}

func orderedEntries(entries []builderEntry) []builderEntry {
	ordered := make([]builderEntry, 0, len(entries))
	for _, entry := range entries {
		if entry.required {
			ordered = append(ordered, entry)
		}
	}
	for _, entry := range entries {
		if !entry.required {
			ordered = append(ordered, entry)
		}
	}
	return ordered
}

func validatePartIDs(seen map[string]Section, section Section, parts []Part) error {
	pending := make([]string, 0, len(parts))
	local := map[string]struct{}{}
	var visit func(Part) error
	visit = func(part Part) error {
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
		for _, child := range part.Children {
			if err := visit(child); err != nil {
				return err
			}
		}
		return nil
	}
	for _, part := range parts {
		if err := visit(part); err != nil {
			return err
		}
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

func (b *Builder) sourceBudget(usedTokens int, entry builderEntry) SourceBudget {
	if b.budget == nil {
		return SourceBudget{
			MaxTokens:             unlimitedTokens,
			RemainingPromptTokens: unlimitedTokens,
			Required:              entry.required,
		}
	}
	remaining := max(b.budget.promptLimit()-usedTokens, 0)
	maxTokens := remaining
	if entry.hasSourceCap && entry.sourceCap < maxTokens {
		maxTokens = entry.sourceCap
	}
	return SourceBudget{
		Tokenizer:                  b.budget.Tokenizer,
		MaxTokens:                  maxTokens,
		RemainingPromptTokens:      remaining,
		Required:                   entry.required,
		RenderOverheadReserveRatio: b.budget.RenderOverheadReserveRatio,
		Summarizer:                 b.budget.Summarizer,
	}
}

func (b *Builder) partsFit(parts map[Section][]Part, section Section, next []Part) (bool, int, int, error) {
	if b.budget == nil {
		return true, 0, unlimitedTokens, nil
	}
	candidate := clonePartsMap(parts)
	candidate[section] = cloneParts(next)
	count := b.countPrompt(candidate)
	limit := b.budget.promptLimit()
	return count <= limit, count, limit, nil
}

func (b *Builder) partsFitAfterDroppingOptionalContext(parts map[Section][]Part, section Section, next []Part) (map[Section][]Part, bool, int, int, error) {
	if b.budget == nil {
		return parts, true, 0, unlimitedTokens, nil
	}
	candidate := clonePartsMap(parts)
	candidate[SectionContext] = keepRequiredParts(candidate[SectionContext])
	if section == SectionContext {
		candidate[SectionContext] = keepRequiredParts(next)
	} else {
		candidate[section] = cloneParts(next)
	}
	count := b.countPrompt(candidate)
	limit := b.budget.promptLimit()
	return candidate, count <= limit, count, limit, nil
}

func (b *Builder) summarizeOptionalPart(ctx stdcontext.Context, parts map[Section][]Part, entry builderEntry, part Part, usedTokens int) (Part, bool, int, int, int, error) {
	if b.budget == nil || b.budget.Summarizer == nil || b.budget.Tokenizer == nil || entry.required {
		return Part{}, false, 0, 0, 0, nil
	}
	remaining := b.budget.promptLimit() - usedTokens
	if remaining <= 0 {
		return Part{}, false, 0, 0, b.budget.promptLimit(), nil
	}
	summary, err := b.budget.Summarizer.Summarize(ctx, SummaryRequest{
		ID:        entry.id,
		Text:      part.Text,
		MaxTokens: remaining,
		Required:  false,
		Meta:      cloneMeta(part.Meta),
	})
	if err != nil {
		return Part{}, false, 0, 0, remaining, nil
	}
	summaryTokens, err := b.budget.Tokenizer.CountTokens(ctx, summary)
	if err != nil {
		return Part{}, false, 0, 0, remaining, err
	}
	summarized := Part{
		ID:       part.ID,
		Text:     summary,
		Tokens:   summaryTokens,
		Required: false,
		Meta:     cloneMeta(part.Meta),
	}
	next := append(cloneParts(parts[entry.section]), summarized)
	ok, count, available, err := b.partsFit(parts, entry.section, next)
	if !ok || err != nil {
		return Part{}, false, 0, count, available, err
	}
	return summarized, true, summaryTokens, count, remaining, nil
}

func (b *Builder) countPrompt(parts map[Section][]Part) int {
	var count int
	for _, section := range parts {
		for _, part := range section {
			count += part.tokenCount()
		}
	}
	return count + renderOverheadTokens(count, b.budget.RenderOverheadReserveRatio)
}

func renderOverheadTokens(tokens int, ratio float64) int {
	if tokens <= 0 || ratio <= 0 {
		return 0
	}
	return int(math.Ceil(float64(tokens) * ratio))
}

func promptBudgetError(id string, used, available int) error {
	return fmt.Errorf("%w: prompt with %q would use %d tokens, only %d available", ErrPromptBudget, id, used, available)
}

func markRequired(parts []Part) {
	for i := range parts {
		parts[i].Required = true
		markRequired(parts[i].Children)
	}
}

func partsTokenCount(parts []Part) int {
	tokens := 0
	for _, part := range parts {
		tokens += part.tokenCount()
	}
	return tokens
}

func setTraceTokens(entry *BuildTraceEntry, entryTokens, promptTokens int) {
	entry.EntryTokens = entryTokens
	entry.PromptTokens = promptTokens
}

func (b *Builder) emitEntry(ctx stdcontext.Context, name string, entry BuildTraceEntry) {
	fields := map[string]any{
		"id":               entry.ID,
		"section":          string(entry.Section),
		"kind":             string(entry.Kind),
		"status":           entry.Status,
		"required":         entry.Required,
		"parts":            len(entry.Parts),
		"entry_tokens":     entry.EntryTokens,
		"prompt_tokens":    entry.PromptTokens,
		"token_count":      entry.EntryTokens,
		"available_tokens": entry.AvailableTokens,
	}
	if entry.Reason != "" {
		fields["reason"] = entry.Reason
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
		ID:        e.id,
		Section:   e.section,
		Kind:      e.kind,
		Required:  e.required,
		Tokens:    e.tokens,
		SourceCap: e.sourceCap,
		Text:      e.text,
		Meta:      cloneMeta(e.meta),
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

func clonePartsMap(parts map[Section][]Part) map[Section][]Part {
	cloned := map[Section][]Part{}
	for section, sectionParts := range parts {
		cloned[section] = cloneParts(sectionParts)
	}
	return cloned
}

func clonePartIDMap(seen map[string]Section) map[string]Section {
	cloned := make(map[string]Section, len(seen))
	maps.Copy(cloned, seen)
	return cloned
}

func rebuildPartIDs(parts map[Section][]Part) map[string]Section {
	ids := map[string]Section{}
	for section, sectionParts := range parts {
		addPartIDs(ids, section, sectionParts)
	}
	return ids
}

func addPartIDs(ids map[string]Section, section Section, parts []Part) {
	for _, part := range parts {
		ids[part.ID] = section
		addPartIDs(ids, section, part.Children)
	}
}

func keepRequiredParts(parts []Part) []Part {
	kept := make([]Part, 0, len(parts))
	for _, part := range parts {
		if part.Required {
			kept = append(kept, part)
		}
	}
	return kept
}

func markDroppedOptionalContextEntries(entries []BuildTraceEntry) []BuildTraceEntry {
	for i := range entries {
		if entries[i].Section == SectionContext && !entries[i].Required && entries[i].Status == "emitted" {
			entries[i].Status = "dropped"
			entries[i].Reason = "dropped_for_required_content"
		}
	}
	return entries
}

func cloneParts(parts []Part) []Part {
	cloned := make([]Part, len(parts))
	for i, part := range parts {
		cloned[i] = part
		cloned[i].Meta = cloneMeta(part.Meta)
		cloned[i].Children = cloneParts(part.Children)
	}
	return cloned
}

func cloneMeta(meta map[string]any) map[string]any {
	if len(meta) == 0 {
		return nil
	}
	cloned := make(map[string]any, len(meta))
	maps.Copy(cloned, meta)
	return cloned
}
