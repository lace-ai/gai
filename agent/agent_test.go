package agent_test

import (
	"testing"

	"github.com/lace-ai/gai/agent"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/testutil/mocks"
)

func TestNewLoopCreatesReusableAgentLoop(t *testing.T) {
	t.Parallel()

	model := &mocks.MockModel{}
	l, err := agent.NewLoop(agent.Definition{
		Model: model,
		PromptBuilderFactory: func(input agent.RunInput) gaictx.PromptBuilder {
			return gaictx.NewPromptBuilder().
				System("system", "system", gaictx.Required()).
				User("request", input.Text, gaictx.Required())
		},
		MaxLoopIterations: 2,
		RetryCount:        1,
		MaxTokens:         9,
	}, agent.RunInput{Text: "input"})
	if err != nil {
		t.Fatalf("NewLoop failed: %v", err)
	}
	if l.Model != model {
		t.Fatal("expected configured model")
	}
	if l.MaxLoopIterations != 2 {
		t.Fatalf("expected max iterations 2, got %d", l.MaxLoopIterations)
	}
	if l.RetryCount != 1 {
		t.Fatalf("expected retry count 1, got %d", l.RetryCount)
	}
	if l.MaxTokens != 9 {
		t.Fatalf("expected max tokens 9, got %d", l.MaxTokens)
	}
}
