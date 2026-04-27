package mocks

import (
	"context"
	"errors"

	"github.com/lace-ai/gai/ai"
)

type MockModelResponse struct {
	Res ai.AIResponse
	Err error
}

type MockModel struct {
	ModelName string
	Count     int
	Responses []MockModelResponse
}

func (m *MockModel) Name() string {
	return m.ModelName
}

var ErrNoMockResponses = errors.New("mock model: no scripted responses left")

func (m *MockModel) Generate(ctx context.Context, req ai.AIRequest) (*ai.AIResponse, error) {
	if m.Count >= len(m.Responses) {
		return nil, ErrNoMockResponses
	}
	r := m.Responses[m.Count]
	m.Count++
	return &r.Res, r.Err
}

func (m *MockModel) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
	out := make(chan ai.Token, 1)

	go func() {
		defer close(out)

		res, err := m.Generate(ctx, req)
		if err != nil {
			out <- ai.Token{Type: ai.TokenTypeErr, Err: err}
			return
		}
		if res == nil || res.Text == "" {
			return
		}

		out <- ai.Token{Type: ai.TokenTypeText, Data: []byte(res.Text)}
	}()

	return out
}

func (m *MockModel) Close() error {
	return nil
}
