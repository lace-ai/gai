<div>
    <a href="https://github.com/lace-ai/gai">
        <img alt="GAI Logo" src="/docs/GAI_thumbnail.png">
    </a>
</div>

<p align="center">
    <a href="https://github.com/lace-ai/gai/blob/main/go.mod">
        <img alt="GitHub go.mod Go version" src="https://img.shields.io/github/go-mod/go-version/lace-ai/gai" style="margin: 0 6px;">
    </a>
    <a href="https://github.com/lace-ai/gai/actions">
        <img alt="CI" src="https://img.shields.io/badge/ci-passing-brightgreen.svg" style="margin: 0 6px;">
    </a>
    <a href="https://github.com/lace-ai/gai/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/license-LGPL-informational.svg" style="margin: 0 6px;">
    </a>
    <a href="https://pkg.go.dev/github.com/lace-ai/gai"><img src="https://pkg.go.dev/badge/github.com/lace-ai/gai.svg" alt="Go Reference"></a>
</p>
<p></p>

GAI is a flexible Go framework for building agent-style applications on top of LLMs.
It provides a generic interface for providers and models, prompt and context implementations, and a loop for agentic-calling workflows.

## ✨ Overview

The library is organized around three ideas:

- 🧩 `ai` defines the core provider, model, request, and response abstractions.
- 🧱 `context` builds rendered prompts from system instructions, runtime context sources, conversation messages, and a user prompt.
- 🔁 `loop` runs iterative model and tool execution when a model returns a tool call.

And:

- 🤖 `agent` packages a model, tools, prompt factory, tokenizer, and loop limits into a reusable definition.

## 📋 Requirements

- Go `1.26.x` or newer
- API credentials for whichever provider you use

## 🚀 Quick Start

### 📦 Installation

```bash
go get github.com/lace-ai/gai
```

### 🧭 Usage

The shortest path is to create a provider model, define an agent, and start a run. Import GAI's `context` package with an alias so it does not conflict with the standard library package.

```go
provider := gemini.New(os.Getenv("GEMINI_API_KEY"), nil)
model, err := provider.Model("gemini-3-flash-preview")

assistant := agent.New(agent.Definition{
  Name:  "assistant",
  Model: model,
  Prompt: func(ctx context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
    return gaictx.New(gaictx.Definition{
      SystemInstructions: []gaictx.Part{
        gaictx.NewTextPart("You are a concise, helpful assistant."),
      },
      UserPrompt:         input.Text,
    }), nil
  },
})

workflow, err := assistant.NewRun(ctx, agent.RunInput{Text: "What is the capital of France?"})

tokens, statuses, errs := workflow.Run(ctx)
go func() {
    for range statuses {
    }
}()
for token := range tokens {
    fmt.Print(token.Text)
}
for err := range errs {
    if err != nil {
        panic(err)
    }
}
```

Agents can include stream middleware in their definition. An input mapper can
project the typed upstream workflow result into the ordinary `RunInput` accepted
by the nested agent. This memory agent records its own output while passing the
assistant response through unchanged.

```go
memoryAgent := agent.New(agent.Definition{
  Name:  "memory",
  Model: model,
  Tools: []loop.Tool{saveMemoryTool},
  Prompt: func(ctx context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
    return memoryPrompt(input), nil
  },
})

assistant := agent.New(agent.Definition{
  Name:   "assistant",
  Model:  model,
  Prompt: assistantPrompt,
  Middleware: []agent.Middleware{
    agent.NewAgentMiddleware(memoryAgent, agent.AgentMiddlewareConfig{
      Output:  agent.PreserveOutput,
      Failure: agent.RecordFailure,
      MapInput: func(ctx context.Context, result agent.WorkflowResult) (agent.RunInput, error) {
        observation, err := buildMemoryObservation(ctx, result)
        if err != nil {
          return agent.RunInput{}, err
        }
        return agent.RunInput{
          ID:   result.Input.ID,
          Text: observation,
          Meta: result.Input.Meta,
        }, nil
      },
    }),
  },
})

workflow, err := assistant.NewRun(ctx, agent.RunInput{
  Text: "What is the capital of France?",
  Meta: map[string]any{"session_id": sessionID},
})
tokens, statuses, errs := workflow.Run(ctx)

go func() {
  for range statuses {
  }
}()
for token := range tokens {
  fmt.Print(token.Text)
}
for err := range errs {
  if err != nil {
    panic(err)
  }
}

result := workflow.Result()
fmt.Printf("memory stage output: %s\n", result.Stages[0].Result.Text)
```

For a single non-agent request, call the model directly with `model.Generate(ctx, ai.AIRequest{Prompt: "...", MaxTokens: 100})`.

## 🧱 Package Layout

```text
agent/          Reusable agent definitions and built-in components such as the summary agent.
ai/             Provider, model, tokenizer, request, and response abstractions, plus Gemini and Mistral.
context/        Prompt construction, parts, rendering, messages, and conversation interfaces.
context/history Persisted history sources and optional history summarization.
loop/           Model/tool execution loop and tool helpers.
testutil/       Mocks used by tests.
```

## 🧩 Core Concepts

<details>

<summary>

### 🏢 Provider

</summary>
A provider is responsible for exposing available models and validating its own configuration.
The shared interface is:

```go
type Provider interface {
    Name() string
    Model(name string) (Model, error)
    ListModels() ([]string, error)
    Validate() error
}
```

Use `ModelRepository` when you want to register multiple providers and look up models by name.

</details>

<details>

<summary>

### 🧠 Model

</summary>

A model generates text from an `AIRequest` and can return either a complete `AIResponse` or a stream of tokens.

```go
type Model interface {
    Name() string
    Generate(ctx context.Context, req AIRequest) (*AIResponse, error)
    GenerateStream(ctx context.Context, req AIRequest) <-chan Token
    Close() error
    Tokenizer() Tokenizer
}
```


</details>

<details>

<summary>

### 📝 Request and Response

</summary>

Providers receive a rendered prompt string and an optional output-token cap:

```go
type AIRequest struct {
    Prompt    string
    MaxTokens int
}
```

`AIResponse` contains the generated `Text` plus `InputTokens` and `OutputTokens`. Agent runs normally create the prompt string through a `context.PromptBuilder`; direct model calls can supply it themselves.

</details>

## 🌐 Providers

<details>

<summary>

### ♊ Gemini

</summary>

Package: `ai/gemini`

Constructor:

```go
gemini.New(apiKey string, debug gai.DebugSink) *gemini.Provider
```

Known model names:

- `gemini-3-flash-preview`
- `gemini-2.5-flash`
- `gemini-3.1-flash-lite-preview`
- `gemini-2.5-flash-lite`

</details>

<details>

<summary>

### 🌀 Mistral

</summary>

Package: `ai/mistral`

Constructor:

```go
mistral.New(apiKey string, debug gai.DebugSink) *mistral.Provider
```

Known model names:

- `mistral-small-latest`
- `mistral-medium-latest`
- `mistral-large-latest`
- `codestral-latest`


</details>

## 💾 Prompt Context

The `context` package is not the standard library `context` package.
Import it with an alias such as `gaictx` to avoid name collisions.

```go
import gaictx "github.com/lace-ai/gai/context"
```

<details>

<summary>

### 📨 Messages

</summary>

Messages have one of four roles:

- `system`
- `user`
- `assistant`
- `tool`

Each message wraps a `Content` implementation such as text, tool calls, or tool results; applications can also implement custom content. `MessagePart` maps the role to the corresponding rendered element.


</details>

<details>

<summary>

### 💬 Conversation

</summary>

`Conversation` is a minimal interface passed to dynamic prompt sources so they can inspect the current loop messages:

```go
type Conversation interface {
    Messages() []Message
}
```

</details>

<details>

<summary>

### 🧱 Prompt Builder

</summary>

`context.New` creates a `Builder` from a `Definition`. The definition supplies the renderer, system instructions, context sources, user prompt, token budget, output reserve, tokenizer, and optional debug sink.

```go
builder := gaictx.New(gaictx.Definition{
  Renderer: gaictx.XMLRenderer{},
  SystemInstructions: []gaictx.Part{
    gaictx.NewTextPart("Follow the system policy."),
  },
  ContextSources: []gaictx.ContextSource{
    history.NewHistory(sessionID, historyStore),
  },
  UserPrompt:         "Summarize the project status.",
  TokenBudget:        128000,
  OutputTokenReserve: 4096,
  Tokenizer:          model.Tokenizer(),
})
```

The builder has two phases:

1. `BuildContext` calls each `ContextSource` in order. A source receives the remaining token budget and returns one `Part`.
2. `BuildPrompt` renders system instructions, the built context parts, the current user prompt, and non-user messages from the loop conversation into one string.

`Part` is the unit of token counting and rendering. Built-in parts include `TextPart`, `MessagePart`, and `SystemPart`; custom parts implement `Name`, `Tokens`, and `Render`. `XMLRenderer` is the default, or provide another `Renderer` in the definition.

When `TokenBudget` is positive, the available context budget is `TokenBudget - OutputTokenReserve - system instruction tokens`. The active tokenizer is passed automatically to context sources that implement `TokenizerSetter`. A builder created by an `agent.Agent` also receives the agent tokenizer override, or the model tokenizer when no override is configured.

Use `AppendSystemInstructions`, `AppendContextSource`, and `SetUserPrompt` when the prompt needs to be changed after construction. `SetDebugSink` enables prompt-build diagnostics.

</details>

<details>

<summary>

### 🧭 History Sources

</summary>

`context/history` provides a `ContextSource` backed by a `HistoryStore`. `history.NewHistory(sessionID, store)` loads the persisted `HistoryState`, keeps the newest turns that fit the source budget, and renders them as one history part. The store owns both history persistence and cached per-turn token counts.

Use `history.New(sessionID, store, summarizerDefinition)` when older turns should be summarized under token pressure. A `SummarizerDefinition` can use an existing `summary.Summarizer` or construct one from a model. Summarized state is saved back through the history store.

</details>

<details>

<summary>

### 🤖 Agent Components

</summary>

The `agent` package turns reusable configuration into independent loop runs:

- `Definition` combines a name, model, tools, prompt factory, limits, optional tokenizer override, optional tool-response preprocessor, and ordered stream middleware.
- `Prompt` builds a `context.PromptBuilder` for one `RunInput`.
- `RunInput` carries an ID, user text, per-run output-token override, metadata, and the upstream result when an agent runs as middleware.
- `Limits` configures maximum loop iterations, retries, and default output tokens.
- `Agent` is created with `agent.New`; `NewRun` returns a `Workflow`, and `Workflow.Run` streams the loop through each configured middleware.
- `AgentMiddleware` adapts an ordinary agent with preserve, append, or replace output behavior. `Workflow.Result` exposes the primary result, final output, and named middleware stages.

The `agent/summary` package is a built-in component. `summary.Definition` returns a reusable agent definition, while `summary.New` returns a `Summarizer` that runs that agent and can be attached to a history source.

</details>

<details>

<summary>

### 🔗 Agent Workflows and Middleware

</summary>

`Agent.NewRun` creates a single-use `Workflow`. Calling `Workflow.Run` starts the
primary loop and passes its stream through every entry in `Definition.Middleware`
in declaration order. Callers must consume the token, status, and error channels.
After they close, `Workflow.Result()` contains the immutable primary result, the
final visible output, accumulated errors, and named middleware stages.

An ordinary agent can become middleware with `NewAgentMiddleware`. By default it
receives the current visible text, original run ID, and metadata. Set
`AgentMiddlewareConfig.MapInput` to deliberately project the typed upstream
`WorkflowResult` into a different `RunInput`:

- `PreserveOutput` streams the upstream output unchanged and keeps the nested
  agent result only in `WorkflowResult.Stages`.
- `AppendOutput` streams the upstream output followed by the nested agent output.
- `ReplaceOutput` buffers the upstream output and, when the stage runs, emits
  only the nested agent output.

Agent middleware runs only after a successful upstream result by default. Set
`AgentMiddlewareConfig.ShouldRun` to implement policies such as failure auditing.
Nested-agent errors are sent through the workflow error channel by default. Set
`Failure: RecordFailure` for best-effort stages whose failures should remain in
`StageResult.Result.Errors` without failing the surrounding workflow.

For transformations that do not require another agent, use `MiddlewareFunc`:

```go
passthrough := agent.MiddlewareFunc(func(
  ctx context.Context,
  run *agent.MiddlewareContext,
  upstream agent.Stream,
) agent.Stream {
  // A custom middleware owns all three streams. Returning upstream is a
  // zero-overhead pass-through; a transformer may return replacement channels.
  return upstream
})

assistant := agent.New(agent.Definition{
  Name:       "assistant",
  Model:      model,
  Prompt:     assistantPrompt,
  Middleware: []agent.Middleware{passthrough},
})
```

`MiddlewareContext.Result()` returns a concurrency-safe snapshot for custom
middleware. `Complete` becomes true only after the final workflow streams close.

</details>

<details>

<summary>

### 📄 Prompt Files

</summary>

`LoadPromptFromFile` reads `.md` and `.txt` files, trims whitespace, and returns the prompt text.

</details>

## 🔄 Loop and Tools

The `loop` package is for agent-style execution where the model can request tool calls.

<details>

<summary>

### 🔁 Loop


</summary>

`loop.New(...)` creates a loop with:

- a model
- optional tools
- a structured prompt builder
- an optional tool-response preprocessor

The loop calls `BuildContext` once before the first iteration, then calls `BuildPrompt` for each iteration so the rendered prompt includes the latest assistant and tool messages. The loop stops when the model returns a normal response or when the maximum iteration count is reached.

</details>

<details>

<summary>

### 🧰 Tool Interface

</summary>

Tools must implement:

```go
type Tool interface {
    Name() string
    Description() string
    Params() string
    Function(ctx context.Context, req *ai.ToolCall) *ToolResponse
}
```

Tool calls are expected to arrive as JSON with this shape:

```json
{
  "type": "function",
  "name": "tool_name",
  "arguments": {
    "some": "value"
  }
}
```

Tool call IDs are generated internally by the runtime and are not model-controlled.

</details>

<details>

<summary>

### 🧪 Helper Functions


</summary>

- `DetectToolCallsInStream` detects tool-call JSON objects in streamed text tokens.
- `CallTool` runs a tool by name.
- `DecodeToolArgs` unmarshals tool arguments into a typed struct.
- `Renderer.RenderToolSignatures` formats tool metadata for prompting.

</details>

## ❗ Errors

Common exported errors include:

- `ai.ErrProviderNotFound`
- `ai.ErrProviderAlreadyExists`
- `ai.ErrNilModelRepository`
- `loop.ErrModelNotConfigured`
- `loop.ErrToolNotFound`
- `loop.ErrMaxIterations`
- `context.ErrPromptMissing`
- `context.ErrSessionStoreNotFound`
- `context.ErrPromptSource`
- `context.ErrTokenizerNotFound`
- `gemini.ErrInvalidAPIKey`
- `mistral.ErrInvalidAPIKey`

Handle provider and tool errors at the call site, especially when a model or session store is user-configured.

To see all the errors, check the errors.go file in each package.

## 🧪 Development

Run all tests:

```bash
go test ./...
```

Run a package test suite:

```bash
go test ./ai/...
go test ./loop/...
go test ./context/...
```

## 📝 Notes

- The `context` package name intentionally mirrors the domain it manages, but it is easy to confuse with `context.Context` from the standard library. Use an alias in imports. The context package is likely to be renamed before official `1.0` release.
- History token caches store content-token counts by tokenizer ID; the prompt builder separately budgets system instructions and its output-token reserve.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.
If you add a new provider or tool, document the new constructor, model names, and any required environment variables.

## 📜 Copyright and License

This library is licensed under the GNU LESSER GENERAL PUBLIC LICENSE v2.1. See [LICENSE](LICENSE) for details.

Copyright (c) 2026 lace-ai. All rights reserved.
