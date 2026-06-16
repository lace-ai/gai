package history

import (
	"context"
	"encoding/json"

	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
)

// Content is a minimal persisted history message representation.
type Content struct {
	Text  string
	Role  gaictx.Role
	Value gaictx.Content `json:"-"`
}

func (c Content) String() string {
	if c.Value != nil {
		return c.Value.String()
	}
	return c.Text
}

func (c Content) Type() string {
	if c.Value != nil {
		return c.Value.Type()
	}
	return gaictx.ContentTypeText
}

func (c Content) Marshal() ([]byte, error) {
	return json.Marshal(c)
}

func (c Content) Render(ctx context.Context) (gaictx.RenderNode, error) {
	if c.Value != nil {
		return c.Value.Render(ctx)
	}
	return gaictx.RenderNode{Type: gaictx.ContentTypeText, Value: c.Text}, nil
}

func MapMessageToContent(msg gaictx.Message) Content {
	if msg.Content == nil {
		return Content{Role: msg.Role}
	}
	return Content{
		Text:  msg.Content.String(),
		Role:  msg.Role,
		Value: msg.Content,
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

func (p *Part) Render(ctx context.Context) (gaictx.RenderNode, error) {
	node := gaictx.RenderNode{Type: "history"}
	if p == nil || len(p.Contents) == 0 {
		return node, nil
	}

	for _, content := range p.Contents {
		child, err := content.Render(ctx)
		if err != nil {
			return gaictx.RenderNode{}, err
		}
		if content.Role == "summary" {
			node.Children = append(node.Children, gaictx.RenderNode{
				Type:     "summary",
				Children: []gaictx.RenderNode{child},
			})
			continue
		}
		node.Children = append(node.Children, gaictx.RenderNode{
			Type:     "message",
			Fields:   []gaictx.RenderField{{Key: "role", Value: string(content.Role)}},
			Children: []gaictx.RenderNode{child},
		})
	}
	return node, nil
}

func (p *Part) Tokens(ctx context.Context, tokenizer ai.Tokenizer) (int, error) {
	if tokenizer == nil {
		return 0, gaictx.ErrTokenizerNotFound
	}
	tokenizerID := tokenizer.ID()
	if count, ok := p.TokenCount[tokenizerID]; ok && count >= 0 {
		return count, nil
	} else if ok {
		delete(p.TokenCount, tokenizerID)
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
