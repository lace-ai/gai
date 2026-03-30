package loop_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"agent-backend/gai/ai"
	"agent-backend/gai/loop"
	"agent-backend/gai/memory"
)

type fakeModel struct {
	lastReq     ai.AIRequest
	responses   []string
	callCount   int
	errOnCall   int
	generateErr error
}

func (f *fakeModel) Name() string {
	return "fake"
}

func (f *fakeModel) Close() error {
	return nil
}

func (f *fakeModel) Generate(_ context.Context, req ai.AIRequest) (*ai.AIResponse, error) {
	f.lastReq = req
	f.callCount++
	if f.generateErr != nil && f.errOnCall == f.callCount {
		return nil, f.generateErr
	}
	if len(f.responses) > 0 {
		idx := f.callCount - 1
		if idx >= len(f.responses) {
			idx = len(f.responses) - 1
		}
		return &ai.AIResponse{Text: f.responses[idx]}, nil
	}
	return &ai.AIResponse{Text: "ok"}, nil
}

const testSessionID = 1

func mustNewAgent(t *testing.T, model ai.Model, tools []loop.Tool, systemPrompt string) *loop.Agent {
	t.Helper()
	agent, err := loop.NewAgent(model, tools, systemPrompt, testSessionID)
	if err != nil {
		t.Fatalf("NewAgent returned error: %v", err)
	}
	return agent
}

func mustNewAgentWithPrompts(t *testing.T, model ai.Model, tools []loop.Tool, basePrompt, toolPrompt string) *loop.Agent {
	t.Helper()
	agent, err := loop.NewAgentWithPrompts(model, tools, basePrompt, toolPrompt, testSessionID)
	if err != nil {
		t.Fatalf("NewAgentWithPrompts returned error: %v", err)
	}
	return agent
}

func mustNewAgentWithMemory(t *testing.T, model ai.Model, tools []loop.Tool, systemPrompt string) *loop.Agent {
	t.Helper()
	memorySystem, err := memory.NewMemory(testSessionID)
	if err != nil {
		t.Fatalf("NewMemory returned error: %v", err)
	}
	agent, err := loop.NewAgentWithMemory(model, tools, systemPrompt, memorySystem)
	if err != nil {
		t.Fatalf("NewAgentWithMemory returned error: %v", err)
	}
	return agent
}

func TestFollowUpAppendsMessagesAndBuildsPrompt(t *testing.T) {
	model := &fakeModel{}
	agent := mustNewAgent(t, model, nil, "system prompt")

	msg, err := agent.FollowUp(context.Background(), "hello")
	if err != nil {
		t.Fatalf("FollowUp returned error: %v", err)
	}
	if msg == "" {
		t.Fatalf("expected message, got empty string")
	}

	messages, err := agent.MemorySystem.GetMessages(0)
	if err != nil {
		t.Fatalf("GetMessages returned error: %v", err)
	}
	if len(messages) != 2 {
		t.Fatalf("expected 2 memory messages, got %d", len(messages))
	}
	if messages[0].Role != memory.RoleUser || messages[0].Content != "hello" {
		t.Fatalf("unexpected first message: %+v", messages[0])
	}
	if messages[1].Role != memory.RoleAssistant || messages[1].Content != "ok" {
		t.Fatalf("unexpected second message: %+v", messages[1])
	}

	if !strings.Contains(model.lastReq.SystemPrompt, "system prompt") {
		t.Fatalf("system prompt missing from request: %q", model.lastReq.SystemPrompt)
	}
	if !strings.Contains(model.lastReq.Context, "<conversation>") {
		t.Fatalf("expected conversation context in request, got %q", model.lastReq.Context)
	}
	if !strings.Contains(model.lastReq.Context, "hello") {
		t.Fatalf("expected user message in context, got %q", model.lastReq.Context)
	}
	if strings.Contains(model.lastReq.SystemPrompt, "<system") {
		t.Fatalf("did not expect system prompt to be stored as conversation message: %q", model.lastReq.SystemPrompt)
	}
}

func TestFollowUpValidation(t *testing.T) {
	t.Run("nil agent", func(t *testing.T) {
		var agent *loop.Agent
		_, err := agent.FollowUp(context.Background(), "hello")
		if !errors.Is(err, loop.ErrNilAgent) {
			t.Fatalf("expected ErrNilAgent, got %v", err)
		}
	})

	t.Run("missing model", func(t *testing.T) {
		agent := mustNewAgent(t, nil, nil, "system")
		_, err := agent.FollowUp(context.Background(), "hello")
		if !errors.Is(err, loop.ErrModelNotConfigured) {
			t.Fatalf("expected ErrModelNotConfigured, got %v", err)
		}
	})

	t.Run("empty prompt", func(t *testing.T) {
		agent := mustNewAgent(t, &fakeModel{}, nil, "system")
		_, err := agent.FollowUp(context.Background(), "   ")
		if !errors.Is(err, loop.ErrEmptyPrompt) {
			t.Fatalf("expected ErrEmptyPrompt, got %v", err)
		}
	})

	t.Run("missing memory", func(t *testing.T) {
		agent, err := loop.NewAgentWithMemory(&fakeModel{}, nil, "system", nil)
		if !errors.Is(err, loop.ErrMemoryNotConfigured) {
			t.Fatalf("expected ErrMemoryNotConfigured, got %v", err)
		}
		if agent != nil {
			t.Fatalf("expected nil agent when memory is missing")
		}
	})
}

type fakeTool struct {
	name   string
	desc   string
	params string
	resp   string
	err    error
}

func (t *fakeTool) Name() string {
	return t.name
}

func (t *fakeTool) Description() string {
	if t.desc != "" {
		return t.desc
	}
	return "fake tool"
}

func (t *fakeTool) Params() string {
	if t.params != "" {
		return t.params
	}
	return "{}"
}

func (t *fakeTool) Function(_ *loop.ToolRequest) (*loop.ToolResponse, error) {
	if t.err != nil {
		return nil, t.err
	}
	return &loop.ToolResponse{Text: t.resp}, nil
}

func TestFollowUpReturnsUnknownToolErrorInTranscript(t *testing.T) {
	model := &fakeModel{
		responses: []string{`{"id":"missing","type":"function","arguments":"hello"}`},
	}
	agent := mustNewAgent(t, model, nil, "system")

	msg, err := agent.FollowUp(context.Background(), "run tool")
	if !errors.Is(err, loop.ErrToolNotFound) {
		t.Fatalf("expected ErrToolNotFound, got %v", err)
	}
	if !strings.Contains(msg, "tool not found: missing") {
		t.Fatalf("expected unknown tool text in response, got %q", msg)
	}
	messages, getErr := agent.MemorySystem.GetMessages(0)
	if getErr != nil {
		t.Fatalf("GetMessages returned error: %v", getErr)
	}
	if len(messages) != 3 {
		t.Fatalf("expected user, assistant, tool messages; got %d", len(messages))
	}
	if messages[2].Role != memory.RoleTool {
		t.Fatalf("expected tool message role, got %v", messages[2].Role)
	}
}

func TestFollowUpStopsWhenToolErrors(t *testing.T) {
	model := &fakeModel{
		responses: []string{`{"id":"boom","type":"function","arguments":"hello"}`},
	}
	tool := &fakeTool{name: "boom", err: errors.New("boom failure")}
	agent := mustNewAgent(t, model, []loop.Tool{tool}, "system")

	msg, err := agent.FollowUp(context.Background(), "run tool")
	if err == nil || err.Error() != "boom failure" {
		t.Fatalf("expected boom failure, got %v", err)
	}
	if !strings.Contains(msg, "boom failure") {
		t.Fatalf("expected tool error in transcript, got %q", msg)
	}
	if model.callCount != 1 {
		t.Fatalf("expected one model call when tool fails, got %d", model.callCount)
	}
}

func TestFollowUpContinuesAfterSuccessfulTool(t *testing.T) {
	arguments, _ := json.Marshal(map[string]string{"text": "hello"})
	model := &fakeModel{
		responses: []string{
			`{"id":"echo","type":"function","arguments":` + string(arguments) + `}`,
			"final answer",
		},
	}
	tool := &fakeTool{name: "echo", resp: "hello"}
	agent := mustNewAgent(t, model, []loop.Tool{tool}, "system")

	msg, err := agent.FollowUp(context.Background(), "use tool")
	if err != nil {
		t.Fatalf("FollowUp returned unexpected error: %v", err)
	}
	if !strings.Contains(msg, "Tool echo {\"text\":\"hello\"}") || !strings.Contains(msg, "final answer") {
		t.Fatalf("expected tool and final answer in transcript, got %q", msg)
	}
	if model.callCount != 2 {
		t.Fatalf("expected two model calls, got %d", model.callCount)
	}
}

func TestFollowUpRespectsMaxLoopIterations(t *testing.T) {
	model := &fakeModel{
		responses: []string{`{"id":"echo","type":"function","arguments":{"text":"x"}}`},
	}
	tool := &fakeTool{name: "echo", resp: "x"}
	agent := mustNewAgent(t, model, []loop.Tool{tool}, "system")
	agent.MaxLoopIterations = 1

	_, err := agent.FollowUp(context.Background(), "loop")
	if !errors.Is(err, loop.ErrMaxIterations) {
		t.Fatalf("expected ErrMaxIterations, got %v", err)
	}
}

func TestFollowUpMalformedToolCallReturnsValidationError(t *testing.T) {
	model := &fakeModel{
		responses: []string{`{"id":"echo","type":"function","arguments":`},
	}
	agent := mustNewAgent(t, model, []loop.Tool{loop.NewEchoTool()}, "system")

	msg, err := agent.FollowUp(context.Background(), "use tool")
	if !errors.Is(err, loop.ErrToolCallMalformed) {
		t.Fatalf("expected ErrToolCallMalformed, got %v", err)
	}
	if !strings.Contains(msg, loop.ErrToolCallMalformed.Error()) {
		t.Fatalf("expected tool-call validation message, got %q", msg)
	}
}

func TestFollowUpIncludesToolPromptAndSignatures(t *testing.T) {
	model := &fakeModel{}
	tools := []loop.Tool{
		&fakeTool{name: "zeta", desc: "z desc", params: `{"z":"1"}`},
		&fakeTool{name: "alpha", desc: "a desc", params: `{"a":"1"}`},
	}
	agent := mustNewAgentWithPrompts(t, model, tools, "base prompt", "tool prompt")

	_, err := agent.FollowUp(context.Background(), "hello")
	if err != nil {
		t.Fatalf("FollowUp returned error: %v", err)
	}

	sp := model.lastReq.SystemPrompt
	if !strings.Contains(sp, "base prompt") || !strings.Contains(sp, "tool prompt") {
		t.Fatalf("expected base/tool prompts in system prompt, got %q", sp)
	}
	if !strings.Contains(sp, "<tools>") || !strings.Contains(sp, "<tool name=\"alpha\">") || !strings.Contains(sp, "<tool name=\"zeta\">") {
		t.Fatalf("expected rendered tool signatures, got %q", sp)
	}
	if strings.Index(sp, "<tool name=\"alpha\">") > strings.Index(sp, "<tool name=\"zeta\">") {
		t.Fatalf("expected alphabetical tool ordering, got %q", sp)
	}
}

func TestLoadPromptFromFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "prompt.md")
	if err := os.WriteFile(path, []byte("  file prompt  \n"), 0o600); err != nil {
		t.Fatalf("failed writing prompt file: %v", err)
	}

	prompt, err := loop.LoadPromptFromFile(path)
	if err != nil {
		t.Fatalf("LoadPromptFromFile returned error: %v", err)
	}
	if prompt != "file prompt" {
		t.Fatalf("expected trimmed prompt, got %q", prompt)
	}
}

func TestLoadPromptFromFileValidation(t *testing.T) {
	_, err := loop.LoadPromptFromFile("   ")
	if !errors.Is(err, loop.ErrPromptPathEmpty) {
		t.Fatalf("expected ErrPromptPathEmpty, got %v", err)
	}

	dir := t.TempDir()
	bad := filepath.Join(dir, "prompt.json")
	if err := os.WriteFile(bad, []byte(`{"x":1}`), 0o600); err != nil {
		t.Fatalf("failed writing prompt file: %v", err)
	}

	_, err = loop.LoadPromptFromFile(bad)
	if !errors.Is(err, loop.ErrPromptFileType) {
		t.Fatalf("expected ErrPromptFileType, got %v", err)
	}

	_, err = loop.LoadPromptFromFile(filepath.Join(dir, "missing.md"))
	if !errors.Is(err, loop.ErrPromptMissing) {
		t.Fatalf("expected ErrPromptMissing, got %v", err)
	}
}

func TestNewAgentFromPromptFiles(t *testing.T) {
	dir := t.TempDir()
	base := filepath.Join(dir, "base.txt")
	tool := filepath.Join(dir, "tool.md")
	if err := os.WriteFile(base, []byte("base from file"), 0o600); err != nil {
		t.Fatalf("failed writing base prompt: %v", err)
	}
	if err := os.WriteFile(tool, []byte("tool from file"), 0o600); err != nil {
		t.Fatalf("failed writing tool prompt: %v", err)
	}

	agent, err := loop.NewAgentFromPromptFiles(&fakeModel{}, nil, base, tool, testSessionID)
	if err != nil {
		t.Fatalf("NewAgentFromPromptFiles returned error: %v", err)
	}
	if agent.BaseSystemPrompt != "base from file" || agent.ToolSystemPrompt != "tool from file" {
		t.Fatalf("unexpected prompts on agent: %+v", agent)
	}
}

func TestNewAgentFromPromptFilesUsesDefaultWhenToolPromptMissing(t *testing.T) {
	dir := t.TempDir()
	base := filepath.Join(dir, "base.txt")
	if err := os.WriteFile(base, []byte("base from file"), 0o600); err != nil {
		t.Fatalf("failed writing base prompt: %v", err)
	}

	agent, err := loop.NewAgentFromPromptFiles(&fakeModel{}, nil, base, filepath.Join(dir, "missing-tool.md"), testSessionID)
	if err != nil {
		t.Fatalf("NewAgentFromPromptFiles returned error: %v", err)
	}
	if !strings.Contains(agent.ToolSystemPrompt, "respond ONLY with one JSON object") {
		t.Fatalf("expected fallback tool prompt, got %q", agent.ToolSystemPrompt)
	}
}

func TestDecodeToolArgs(t *testing.T) {
	req := &loop.ToolRequest{
		ID:   "echo",
		Type: "function",
		Args: json.RawMessage(`{"text":"hello"}`),
	}
	var args struct {
		Text string `json:"text"`
	}
	if err := loop.DecodeToolArgs(req, &args); err != nil {
		t.Fatalf("DecodeToolArgs returned error: %v", err)
	}
	if args.Text != "hello" {
		t.Fatalf("expected decoded text, got %q", args.Text)
	}
}

func TestDecodeToolArgsValidation(t *testing.T) {
	req := &loop.ToolRequest{ID: "echo", Type: "function"}

	var args struct {
		Text string `json:"text"`
	}
	err := loop.DecodeToolArgs(req, &args)
	if !errors.Is(err, loop.ErrToolArgsMissing) {
		t.Fatalf("expected ErrToolArgsMissing, got %v", err)
	}

	var nilTarget *struct {
		Text string `json:"text"`
	}
	err = loop.DecodeToolArgs(req, nilTarget)
	if !errors.Is(err, loop.ErrArgsDecodeTarget) {
		t.Fatalf("expected ErrArgsDecodeTarget, got %v", err)
	}

	req.Args = json.RawMessage(`{`)
	err = loop.DecodeToolArgs(req, &args)
	if !errors.Is(err, loop.ErrToolCallMalformed) {
		t.Fatalf("expected ErrToolCallMalformed, got %v", err)
	}
}

func TestAgentStoresMessagesInMemory(t *testing.T) {
	model := &fakeModel{}
	agent := mustNewAgentWithMemory(t, model, nil, "system")
	agent.MaxMessages = 3

	for i := 0; i < 3; i++ {
		if _, err := agent.FollowUp(context.Background(), "hello"); err != nil {
			t.Fatalf("FollowUp returned error: %v", err)
		}
	}

	messages, err := agent.MemorySystem.GetMessages(0)
	if err != nil {
		t.Fatalf("GetMessages returned error: %v", err)
	}
	if len(messages) != 6 {
		t.Fatalf("expected 6 messages in memory, got %d", len(messages))
	}
}
