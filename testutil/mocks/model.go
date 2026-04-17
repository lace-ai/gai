package mocks

import (
	"context"
	"errors"

	"github.com/HecoAI/gai/ai"
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

func (m *MockModel) Close() error {
	return nil
}
