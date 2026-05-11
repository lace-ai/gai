package context

import (
	stdcontext "context"
	"encoding/xml"
	"fmt"
	"strings"

	"github.com/lace-ai/gai/ai"
)

type Section string

const (
	SectionSystem  Section = "system"
	SectionContext Section = "context"
	SectionUser    Section = "user"
)

type Part struct {
	Name     string
	Text     string
	Tokens   int
	Required bool
}

func StaticPart(name, text string) Part {
	return Part{
		Name: name,
		Text: text,
	}
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
	BuildParts(ctx stdcontext.Context, conv Conversation) ([]Part, error)
}

type SourceFunc func(ctx stdcontext.Context, conv Conversation) ([]Part, error)

func (f SourceFunc) BuildParts(ctx stdcontext.Context, conv Conversation) ([]Part, error) {
	return f(ctx, conv)
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
		builder.WriteString(`<part name="`)
		writeEscaped(&builder, part.Name)
		builder.WriteString(`">`)
		if part.Text != "" {
			builder.WriteString("\n")
			builder.WriteString(part.Text)
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
	sections map[Section][]builderEntry
}

type builderEntry struct {
	name     string
	required bool
	part     *Part
	source   Source
}

func NewPromptBuilder() *Builder {
	return &Builder{
		renderer: XMLRenderer{},
		sections: map[Section][]builderEntry{
			SectionSystem:  nil,
			SectionContext: nil,
			SectionUser:    nil,
		},
	}
}

func (b *Builder) Renderer(renderer Renderer) *Builder {
	if renderer != nil {
		b.renderer = renderer
	}
	return b
}

func (b *Builder) System(parts ...Part) *Builder {
	return b.addParts(SectionSystem, parts...)
}

func (b *Builder) Context(parts ...Part) *Builder {
	return b.addParts(SectionContext, parts...)
}

func (b *Builder) User(parts ...Part) *Builder {
	return b.addParts(SectionUser, parts...)
}

func (b *Builder) SystemSource(name string, source Source, required bool) *Builder {
	return b.addSource(SectionSystem, name, source, required)
}

func (b *Builder) ContextSource(name string, source Source, required bool) *Builder {
	return b.addSource(SectionContext, name, source, required)
}

func (b *Builder) UserSource(name string, source Source, required bool) *Builder {
	return b.addSource(SectionUser, name, source, required)
}

func (b *Builder) BuildPrompt(ctx stdcontext.Context, conv Conversation) (ai.Prompt, error) {
	if b == nil {
		return ai.Prompt{}, ErrPromptBuilderNil
	}
	renderer := b.renderer
	if renderer == nil {
		renderer = XMLRenderer{}
	}

	system, err := b.buildSection(ctx, conv, SectionSystem)
	if err != nil {
		return ai.Prompt{}, err
	}
	context, err := b.buildSection(ctx, conv, SectionContext)
	if err != nil {
		return ai.Prompt{}, err
	}
	user, err := b.buildSection(ctx, conv, SectionUser)
	if err != nil {
		return ai.Prompt{}, err
	}

	return ai.Prompt{
		System:  renderer.Render(SectionSystem, system),
		Context: renderer.Render(SectionContext, context),
		Prompt:  renderer.Render(SectionUser, user),
	}, nil
}

func (b *Builder) addParts(section Section, parts ...Part) *Builder {
	if b.sections == nil {
		b.sections = make(map[Section][]builderEntry)
	}
	for _, part := range parts {
		p := part
		b.sections[section] = append(b.sections[section], builderEntry{
			name:     part.Name,
			required: part.Required,
			part:     &p,
		})
	}
	return b
}

func (b *Builder) addSource(section Section, name string, source Source, required bool) *Builder {
	if b.sections == nil {
		b.sections = make(map[Section][]builderEntry)
	}
	b.sections[section] = append(b.sections[section], builderEntry{
		name:     name,
		required: required,
		source:   source,
	})
	return b
}

func (b *Builder) buildSection(ctx stdcontext.Context, conv Conversation, section Section) ([]Part, error) {
	entries := b.sections[section]
	parts := make([]Part, 0, len(entries))

	for _, entry := range entries {
		if entry.part != nil {
			parts = append(parts, *entry.part)
			continue
		}
		if entry.source == nil {
			if entry.required {
				return nil, fmt.Errorf("%w: section %s source %q is nil", ErrPromptSource, section, entry.name)
			}
			continue
		}

		sourceParts, err := entry.source.BuildParts(ctx, conv)
		if err != nil {
			if entry.required {
				return nil, fmt.Errorf("%w: section %s source %q: %w", ErrPromptSource, section, entry.name, err)
			}
			continue
		}
		if entry.required {
			for i := range sourceParts {
				sourceParts[i].Required = true
			}
		}
		parts = append(parts, sourceParts...)
	}

	return parts, nil
}
