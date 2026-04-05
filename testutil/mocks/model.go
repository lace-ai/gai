package mocks

import (
	"context"

	"agent-backend/gai/ai"
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

func (m *MockModel) Generate(ctx context.Context, req ai.AIRequest) (*ai.AIResponse, error) {
	if m.Count >= len(m.Responses) {
		return nil, nil
	}
	r := m.Responses[m.Count]
	m.Count++
	return &r.Res, r.Err
}

func (m *MockModel) Close() error {
	return nil
}
