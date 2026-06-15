package history

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

// Content is a minimal persisted history message representation.
type Content struct {
	Text string
	Role gaictx.Role
}

func (c Content) String() string {
	return c.Text
}

func (c Content) Type() string {
	return gaictx.ContentTypeText
}

func (c Content) Marshal() ([]byte, error) {
	return json.Marshal(c)
}

func MapMessageToContent(msg gaictx.Message) Content {
	text, err := msg.Content.Marshal()
	if err != nil {
		text = []byte{}
	}
	return Content{
		Text: string(text),
		Role: msg.Role,
	}
}

// Part is the rendered prompt part emitted by HistorySource.
type Part struct {
	Contents   []Content
	TokenCount map[string]int
}

func (p *Part) Name() string {
	return "history"
}

func (p *Part) Marshal(ctx context.Context) ([]byte, error) {
	if p == nil || len(p.Contents) == 0 {
		return []byte{}, nil
	}

	var builder strings.Builder
	for i, content := range p.Contents {
		if i > 0 {
			builder.WriteString("\n")
		}
		builder.WriteString(content.String())
	}
	return []byte(builder.String()), nil
}

func (p *Part) Tokens(ctx context.Context, tokenizer ai.Tokenizer) (int, error) {
	if tokenizer == nil {
		return 0, gaictx.ErrTokenizerNotFound
	}
	tokenizerID := tokenizer.ID()
	if count, ok := p.TokenCount[tokenizerID]; ok {
		return count, nil
	}

	count := 0
	for _, content := range p.Contents {
		tokens, err := tokenizer.CountTokens(ctx, content.String())
		if err != nil {
			return 0, err
		}
		count += tokens
	}
	p.saveTokens(tokenizerID, count)
	return count, nil
}

func (p *Part) saveTokens(tokenizerID string, tokens int) {
	if p.TokenCount == nil {
		p.TokenCount = make(map[string]int)
	}
	p.TokenCount[tokenizerID] = tokens
}
