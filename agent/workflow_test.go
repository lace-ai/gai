package agent_test

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"

	"github.com/lace-ai/gai/agent"
	"github.com/lace-ai/gai/ai"
	gaictx "github.com/lace-ai/gai/context"
	"github.com/lace-ai/gai/loop"
	"github.com/lace-ai/gai/testutil/mocks"
)

func workflowAgent(name, response string, middleware ...agent.Middleware) *agent.Agent {
	return agent.New(agent.Definition{
		Name:  name,
		Model: &mocks.MockModel{Responses: []mocks.MockModelResponse{{Res: ai.AIResponse{Text: response}}}},
		Prompt: func(_ context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
			return &testPromptBuilder{prompt: input.Text}, nil
		},
		Middleware: middleware,
	})
}

type consumedWorkflow struct {
	tokens   []ai.Token
	statuses []loop.IterationInformation
	errs     []error
}

func consumeWorkflow(t *testing.T, workflow *agent.Workflow) consumedWorkflow {
	t.Helper()
	tokens, statuses, errs := workflow.Run(context.Background())
	var consumed consumedWorkflow
	var wg sync.WaitGroup
	wg.Add(3)
	go func() {
		defer wg.Done()
		for token := range tokens {
			consumed.tokens = append(consumed.tokens, token)
		}
	}()
	go func() {
		defer wg.Done()
		for status := range statuses {
			consumed.statuses = append(consumed.statuses, status)
		}
	}()
	go func() {
		defer wg.Done()
		for err := range errs {
			if err != nil {
				consumed.errs = append(consumed.errs, err)
			}
		}
	}()
	wg.Wait()
	return consumed
}

func tokensText(tokens []ai.Token) string {
	var text string
	for _, token := range tokens {
		if token.Type != ai.TokenTypeText {
			continue
		}
		if token.Text != "" {
			text += token.Text
		} else {
			text += string(token.Data)
		}
	}
	return text
}

func TestAgentMiddlewareOutputPolicies(t *testing.T) {
	tests := []struct {
		name       string
		policy     agent.OutputPolicy
		wantOutput string
	}{
		{name: "preserve", policy: agent.PreserveOutput, wantOutput: "main"},
		{name: "append", policy: agent.AppendOutput, wantOutput: "mainpost"},
		{name: "replace", policy: agent.ReplaceOutput, wantOutput: "post"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var postInput agent.RunInput
			post := agent.New(agent.Definition{
				Name:  "post",
				Model: &mocks.MockModel{Responses: []mocks.MockModelResponse{{Res: ai.AIResponse{Text: "post"}}}},
				Prompt: func(_ context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
					postInput = input
					return &testPromptBuilder{prompt: input.Text}, nil
				},
			})
			main := workflowAgent("main", "main", agent.NewAgentMiddleware(post, agent.AgentMiddlewareConfig{
				Name:   "post",
				Output: tt.policy,
			}))

			workflow, err := main.NewRun(context.Background(), agent.RunInput{
				ID:   "run-1",
				Text: "question",
				Meta: map[string]any{"session_id": "session-1"},
			})
			if err != nil {
				t.Fatalf("NewRun failed: %v", err)
			}
			consumed := consumeWorkflow(t, workflow)
			if got := tokensText(consumed.tokens); got != tt.wantOutput {
				t.Fatalf("unexpected output: want %q got %q", tt.wantOutput, got)
			}
			if len(consumed.errs) != 0 {
				t.Fatalf("unexpected errors: %v", consumed.errs)
			}
			if len(consumed.statuses) != 1 {
				t.Fatalf("expected only the primary status, got %d", len(consumed.statuses))
			}
			if postInput.ID != "run-1" || postInput.Text != "main" || postInput.Meta["session_id"] != "session-1" {
				t.Fatalf("unexpected automatic post input: %+v", postInput)
			}

			result := workflow.Result()
			if !result.Complete || result.Text != tt.wantOutput || result.Primary.Text != "main" {
				t.Fatalf("unexpected workflow result: %+v", result)
			}
			if len(result.Stages) != 1 || result.Stages[0].Name != "post" || result.Stages[0].Result.Text != "post" {
				t.Fatalf("unexpected stage result: %+v", result.Stages)
			}
		})
	}
}

func TestAgentMiddlewareMapsWorkflowResult(t *testing.T) {
	var mappedResult agent.WorkflowResult
	var postInput agent.RunInput
	post := agent.New(agent.Definition{
		Name:  "post",
		Model: &mocks.MockModel{Responses: []mocks.MockModelResponse{{Res: ai.AIResponse{Text: "post"}}}},
		Prompt: func(_ context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
			postInput = input
			return &testPromptBuilder{prompt: input.Text}, nil
		},
	})
	main := workflowAgent("main", "main", agent.NewAgentMiddleware(post, agent.AgentMiddlewareConfig{
		Output: agent.PreserveOutput,
		MapInput: func(_ context.Context, result agent.WorkflowResult) (agent.RunInput, error) {
			mappedResult = result
			return agent.RunInput{ID: "mapped", Text: "observation"}, nil
		},
	}))
	workflow, err := main.NewRun(context.Background(), agent.RunInput{Text: "question"})
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	consumeWorkflow(t, workflow)
	if mappedResult.Input.Text != "question" || mappedResult.Text != "main" {
		t.Fatalf("input mapper did not receive the workflow result: %+v", mappedResult)
	}
	if postInput.ID != "mapped" || postInput.Text != "observation" {
		t.Fatalf("post agent did not receive mapped input: %+v", postInput)
	}
}

func TestAgentMiddlewareRunsInOrderWithPriorStageResults(t *testing.T) {
	var order []string
	firstStageCount := -1
	secondStageCount := -1
	secondPriorText := ""
	first := agent.New(agent.Definition{
		Name:  "first",
		Model: &mocks.MockModel{Responses: []mocks.MockModelResponse{{Res: ai.AIResponse{Text: "memory"}}}},
		Prompt: func(_ context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
			order = append(order, "first")
			return &testPromptBuilder{prompt: input.Text}, nil
		},
	})
	second := agent.New(agent.Definition{
		Name:  "second",
		Model: &mocks.MockModel{Responses: []mocks.MockModelResponse{{Res: ai.AIResponse{Text: "audit"}}}},
		Prompt: func(_ context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
			order = append(order, "second")
			return &testPromptBuilder{prompt: input.Text}, nil
		},
	})
	main := workflowAgent("main", "main",
		agent.NewAgentMiddleware(first, agent.AgentMiddlewareConfig{
			Output: agent.PreserveOutput,
			MapInput: func(_ context.Context, result agent.WorkflowResult) (agent.RunInput, error) {
				firstStageCount = len(result.Stages)
				return agent.RunInput{Text: result.Text}, nil
			},
		}),
		agent.NewAgentMiddleware(second, agent.AgentMiddlewareConfig{
			Output: agent.AppendOutput,
			MapInput: func(_ context.Context, result agent.WorkflowResult) (agent.RunInput, error) {
				secondStageCount = len(result.Stages)
				if len(result.Stages) > 0 {
					secondPriorText = result.Stages[0].Result.Text
				}
				return agent.RunInput{Text: result.Text}, nil
			},
		}),
	)

	workflow, err := main.NewRun(context.Background(), agent.RunInput{Text: "question"})
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	consumed := consumeWorkflow(t, workflow)
	if got := tokensText(consumed.tokens); got != "mainaudit" {
		t.Fatalf("unexpected output: %q", got)
	}
	if !reflect.DeepEqual(order, []string{"first", "second"}) {
		t.Fatalf("unexpected middleware order: %v", order)
	}
	if firstStageCount != 0 || secondStageCount != 1 || secondPriorText != "memory" {
		t.Fatalf("unexpected mapped stage state: first=%d second=%d prior=%q", firstStageCount, secondStageCount, secondPriorText)
	}
	if got := workflow.Result().Stages; len(got) != 2 || got[0].Name != "first" || got[1].Name != "second" {
		t.Fatalf("unexpected stages: %+v", got)
	}
}

func TestAgentMiddlewareErrorPolicy(t *testing.T) {
	stageErr := errors.New("stage failed")
	newPost := func() *agent.Agent {
		return agent.New(agent.Definition{
			Name: "post",
			Model: &mocks.MockModel{Responses: []mocks.MockModelResponse{
				{Err: stageErr},
				{Err: stageErr},
			}},
			Prompt: func(_ context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
				return &testPromptBuilder{prompt: input.Text}, nil
			},
			Limits: agent.Limits{RetryCount: 1},
		})
	}

	for _, tt := range []struct {
		name        string
		output      agent.OutputPolicy
		errorPolicy agent.ErrorPolicy
		wantError   bool
	}{
		{name: "preserve and propagate", output: agent.PreserveOutput, errorPolicy: agent.PropagateError, wantError: true},
		{name: "append and record", output: agent.AppendOutput, errorPolicy: agent.RecordError, wantError: false},
		{name: "replace and record", output: agent.ReplaceOutput, errorPolicy: agent.RecordError, wantError: false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			main := workflowAgent("main", "main", agent.NewAgentMiddleware(newPost(), agent.AgentMiddlewareConfig{
				Output:      tt.output,
				ErrorPolicy: tt.errorPolicy,
			}))
			workflow, err := main.NewRun(context.Background(), agent.RunInput{Text: "question"})
			if err != nil {
				t.Fatalf("NewRun failed: %v", err)
			}
			consumed := consumeWorkflow(t, workflow)
			if got := tokensText(consumed.tokens); got != "main" {
				t.Fatalf("unexpected output: %q", got)
			}
			if (len(consumed.errs) > 0) != tt.wantError {
				t.Fatalf("unexpected streamed errors: %v", consumed.errs)
			}
			result := workflow.Result()
			if (len(result.Errors) > 0) != tt.wantError {
				t.Fatalf("unexpected workflow errors: %v", result.Errors)
			}
			if len(result.Stages) != 1 || len(result.Stages[0].Result.Errors) == 0 {
				t.Fatalf("stage failure was not recorded: %+v", result.Stages)
			}
		})
	}
}

func TestAgentMiddlewareRecordsInputMappingFailure(t *testing.T) {
	mapErr := errors.New("map input")
	postCalled := false
	post := agent.New(agent.Definition{
		Name:  "post",
		Model: &mocks.MockModel{},
		Prompt: func(_ context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
			postCalled = true
			return &testPromptBuilder{prompt: input.Text}, nil
		},
	})
	main := workflowAgent("main", "main", agent.NewAgentMiddleware(post, agent.AgentMiddlewareConfig{
		Output:      agent.PreserveOutput,
		ErrorPolicy: agent.RecordError,
		MapInput: func(context.Context, agent.WorkflowResult) (agent.RunInput, error) {
			return agent.RunInput{}, mapErr
		},
	}))

	workflow, err := main.NewRun(context.Background(), agent.RunInput{Text: "question"})
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	consumed := consumeWorkflow(t, workflow)
	if len(consumed.errs) != 0 || postCalled {
		t.Fatalf("mapping failure propagated or ran post agent: errors=%v called=%v", consumed.errs, postCalled)
	}
	result := workflow.Result()
	if len(result.Errors) != 0 || len(result.Stages) != 1 || len(result.Stages[0].Result.Errors) != 1 || !errors.Is(result.Stages[0].Result.Errors[0], mapErr) {
		t.Fatalf("mapping failure was not isolated to the stage: %+v", result)
	}
}

func TestAgentMiddlewareRunPolicy(t *testing.T) {
	modelErr := errors.New("model failed")
	newMain := func(middleware agent.Middleware) *agent.Agent {
		return agent.New(agent.Definition{
			Name: "main",
			Model: &mocks.MockModel{Responses: []mocks.MockModelResponse{
				{Err: modelErr},
				{Err: modelErr},
			}},
			Prompt: func(_ context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
				return &testPromptBuilder{prompt: input.Text}, nil
			},
			Limits:     agent.Limits{RetryCount: 1},
			Middleware: []agent.Middleware{middleware},
		})
	}

	postCalled := false
	post := agent.New(agent.Definition{
		Name:  "failure-audit",
		Model: &mocks.MockModel{Responses: []mocks.MockModelResponse{{Res: ai.AIResponse{Text: "audited"}}}},
		Prompt: func(_ context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
			postCalled = true
			return &testPromptBuilder{prompt: input.Text}, nil
		},
	})

	workflow, err := newMain(agent.NewAgentMiddleware(post, agent.AgentMiddlewareConfig{})).NewRun(context.Background(), agent.RunInput{Text: "question"})
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	consumed := consumeWorkflow(t, workflow)
	if len(consumed.errs) == 0 || postCalled || len(workflow.Result().Stages) != 0 {
		t.Fatalf("default failure policy did not skip middleware: errors=%v called=%v", consumed.errs, postCalled)
	}

	postCalled = false
	workflow, err = newMain(agent.NewAgentMiddleware(post, agent.AgentMiddlewareConfig{
		ShouldRun: func(result agent.WorkflowResult) bool {
			return len(result.Errors) > 0
		},
	})).NewRun(context.Background(), agent.RunInput{Text: "question"})
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	consumed = consumeWorkflow(t, workflow)
	if len(consumed.errs) == 0 || !postCalled || len(workflow.Result().Stages) != 1 {
		t.Fatalf("custom failure policy did not run middleware: errors=%v called=%v", consumed.errs, postCalled)
	}
}

func TestWorkflowRejectsRepeatedRun(t *testing.T) {
	workflow, err := workflowAgent("main", "main").NewRun(context.Background(), agent.RunInput{})
	if err != nil {
		t.Fatalf("NewRun failed: %v", err)
	}
	consumeWorkflow(t, workflow)
	_, _, errs := workflow.Run(context.Background())
	if err := <-errs; !errors.Is(err, agent.ErrWorkflowAlreadyRun) {
		t.Fatalf("expected repeated-run error, got %v", err)
	}
}

func TestAgentValidatesMiddleware(t *testing.T) {
	_, err := agent.New(agent.Definition{
		Model:      &mocks.MockModel{},
		Prompt:     func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) { return &testPromptBuilder{}, nil },
		Middleware: []agent.Middleware{nil},
	}).NewRun(context.Background(), agent.RunInput{})
	if !errors.Is(err, agent.ErrMiddlewareNotConfigured) {
		t.Fatalf("expected middleware validation error, got %v", err)
	}

	post := workflowAgent("post", "post")
	_, err = agent.New(agent.Definition{
		Model:  &mocks.MockModel{},
		Prompt: func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) { return &testPromptBuilder{}, nil },
		Middleware: []agent.Middleware{agent.NewAgentMiddleware(post, agent.AgentMiddlewareConfig{
			ErrorPolicy: agent.ErrorPolicy(255),
		})},
	}).NewRun(context.Background(), agent.RunInput{})
	if !errors.Is(err, agent.ErrMiddlewareErrorPolicyInvalid) {
		t.Fatalf("expected middleware failure-policy error, got %v", err)
	}

	nested := workflowAgent("nested", "nested")
	postWithMiddleware := workflowAgent("post", "post", agent.NewAgentMiddleware(nested, agent.AgentMiddlewareConfig{}))
	_, err = agent.New(agent.Definition{
		Model:      &mocks.MockModel{},
		Prompt:     func(context.Context, agent.RunInput) (gaictx.PromptBuilder, error) { return &testPromptBuilder{}, nil },
		Middleware: []agent.Middleware{agent.NewAgentMiddleware(postWithMiddleware, agent.AgentMiddlewareConfig{})},
	}).NewRun(context.Background(), agent.RunInput{})
	if !errors.Is(err, agent.ErrMiddlewareAgentNested) {
		t.Fatalf("expected nested middleware-agent error, got %v", err)
	}
}
