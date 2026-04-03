package loop_test

import (
	"context"

	"agent-backend/gai/ai"
)

type MockModel struct {
	name      string
	count     int
	responses []struct {
		res ai.AIResponse
		err error
	}
}

func (m *MockModel) Name() string {
	return m.name
}

func (m *MockModel) Generate(ctx context.Context, req ai.AIRequest) (*ai.AIResponse, error) {
	if m.count >= len(m.responses) {
		return nil, nil
	}
	r := m.responses[m.count]
	m.count++
	return &r.res, r.err
}

func (m *MockModel) Close() error {
	return nil
}
