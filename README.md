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

GAI is a Go library for building agent workflows that keep the application
contract typed while still letting providers expose native LLM capabilities.
It gives you provider-neutral model requests, structured prompt/context
assembly, iterative model/tool execution, workflow middleware, and observable
run results.

## ✨ Overview

The library is organized around four layers:

- 🧩 `ai` defines provider, model, tokenizer, request, response, tool-call, structured-output, and reasoning abstractions.
- 🧱 `context` builds rendered prompts from system instructions, dynamic context sources, structured run input, conversation messages, and token budgets.
- 🔁 `loop` runs the ordered model/tool iteration stream, including retries, tool responses, and completed iteration state.
- 🤖 `agent` packages a model, tools, prompt factory, tokenizer, loop limits, debug sink, and stream middleware into reusable workflow definitions.

The intended use is to compose application-specific agents without coupling the
rest of your code to Gemini, Mistral, or any other provider adapter.

## 📋 Requirements

- Go `1.26.x` or newer
- API credentials for whichever provider you use

## 🚀 Quick Start

### 📦 Installation

```bash
go get github.com/lace-ai/gai
```

### 🧭 Usage

The shortest path is to create a provider model, define an agent, create a
single-use workflow, and consume all three workflow streams. Import GAI's
`context` package with an alias so it does not conflict with the standard
library package.

```go
provider := gemini.New(os.Getenv("GEMINI_API_KEY"), nil)
model, err := provider.Model("gemini-3-flash-preview")
if err != nil {
  return err
}

assistant := agent.New(agent.Definition{
  Name:  "assistant",
  Model: model,
  Prompt: func(ctx context.Context, input agent.RunInput) (gaictx.PromptBuilder, error) {
    return gaictx.New(gaictx.Definition{
      SystemInstructions: []gaictx.Part{
        gaictx.NewTextPart("You are a concise, helpful assistant."),
      },
    }), nil
  },
})

workflow, err := assistant.NewRun(ctx, agent.RunInput{
  Prompt: gaictx.PromptInput{
    User: gaictx.NewTextContent("What is the capital of France?"),
  },
})
if err != nil {
  return err
}

tokens, statuses, errs := workflow.Run(ctx)
statusDone := make(chan struct{})
go func() {
  defer close(statusDone)
  for range statuses {
  }
}()

var runErr error
errorDone := make(chan struct{})
go func() {
  defer close(errorDone)
  for err := range errs {
    if err != nil && runErr == nil {
      runErr = err
    }
  }
}()

for token := range tokens {
  fmt.Print(token.Text)
}
<-statusDone
<-errorDone
if runErr != nil {
  return runErr
}

result := workflow.Result()
fmt.Printf("\nfinal text: %s\n", result.Text)
```

Agents can also include stream middleware. A middleware stage can be an ordinary
agent that receives the typed upstream `WorkflowResult` and records, appends, or
replaces output. This memory agent records its own result while passing the
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
      Output:      agent.PreserveOutput,
      ErrorPolicy: agent.RecordError,
      MapInput: func(ctx context.Context, result agent.WorkflowResult) (agent.RunInput, error) {
        observation, err := buildMemoryObservation(ctx, result)
        if err != nil {
          return agent.RunInput{}, err
        }
        observationPart, err := gaictx.NewJSONPart("memory_observation", observation)
        if err != nil {
          return agent.RunInput{}, err
        }
        return agent.RunInput{
          ID: result.Input.ID,
          Prompt: gaictx.PromptInput{
            Context: []gaictx.Part{observationPart},
          },
          Meta: result.Input.Meta,
        }, nil
      },
    }),
  },
})

workflow, err := assistant.NewRun(ctx, agent.RunInput{
  Prompt: gaictx.PromptInput{
    User: gaictx.NewTextContent("What is the capital of France?"),
  },
  Meta: map[string]any{"session_id": sessionID},
})
if err != nil {
  return err
}
tokens, statuses, errs := workflow.Run(ctx)

statusDone := make(chan struct{})
go func() {
  defer close(statusDone)
  for range statuses {
  }
}()

var runErr error
errorDone := make(chan struct{})
go func() {
  defer close(errorDone)
  for err := range errs {
    if err != nil && runErr == nil {
      runErr = err
    }
  }
}()

for token := range tokens {
  fmt.Print(token.Text)
}
<-statusDone
<-errorDone
if runErr != nil {
  return runErr
}

result := workflow.Result()
fmt.Printf("memory stage output: %s\n", result.Stages[0].Result.Text)
```

For a single non-agent request, call the model directly. Direct requests are
where you opt into provider-native capabilities such as structured JSON output,
tool choice, or reasoning hints:

```go
res, err := model.Generate(ctx, ai.AIRequest{
  Prompt:    "Return one JSON object describing Paris.",
  MaxTokens: 100,
  ResponseFormat: ai.ResponseFormat{
    Type: ai.ResponseFormatJSONObject,
  },
  Reasoning: ai.ReasoningConfig{
    Enabled:         true,
    IncludeThoughts: false,
    Effort:          ai.ReasoningEffortLow,
  },
})
if err != nil {
  return err
}
fmt.Println(res.Text)
```

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

Providers receive a rendered prompt string plus provider-neutral capability
requests for native tools, structured responses, and reasoning:

```go
type AIRequest struct {
    Prompt         string
    MaxTokens      int
    Tools          []ToolDefinition
    ToolChoice     ToolChoice
    ResponseFormat ResponseFormat
    Reasoning      ReasoningConfig
}
```

`AIResponse` contains visible `Text`, separate `Reasoning`, structured `ToolCalls`, optional provider `Raw` data, and token usage. Agent runs normally create the prompt string through a `context.PromptBuilder`; direct model calls can supply it themselves. Workflow results follow the same split: `WorkflowResult.Text` is visible output and `WorkflowResult.Reasoning` is thought-token output.

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

`context.New` creates a `Builder` from a `Definition`. The definition supplies the renderer, system instructions, context sources, structured prompt input, token budget, output reserve, tokenizer, and optional debug sink.

```go
builder := gaictx.New(gaictx.Definition{
  Renderer: &gaictx.XMLRenderer{},
  SystemInstructions: []gaictx.Part{
    gaictx.NewTextPart("Follow the system policy."),
  },
  ContextSources: []gaictx.ContextSource{
    history.NewHistory(sessionID, historyStore),
  },
  PromptInput: gaictx.PromptInput{
    User: gaictx.NewTextContent("Summarize the project status."),
  },
  TokenBudget:        128000,
  OutputTokenReserve: 4096,
  Tokenizer:          model.Tokenizer(),
})
```

The builder has two phases:

1. `BuildContext` calls each `ContextSource` in order. A source receives the remaining token budget and returns one `Part`.
2. `BuildPrompt` renders system instructions, built context sources, input context parts, current user content, and non-user messages from the loop conversation into one string.

`Part` is the unit of token counting and rendering. Built-in parts include `TextPart`, `NamedPart`, `MessagePart`, and `SystemPart`; `NewJSONPart` creates named structured JSON context. Custom parts implement `Name`, `Tokens`, and `Render`. `XMLRenderer` is the default, or provide another `Renderer` in the definition.

When `TokenBudget` is positive, the available context budget is `TokenBudget - OutputTokenReserve - system instruction tokens`. The active tokenizer is passed automatically to context sources that implement `TokenizerSetter`. A builder created by an `agent.Agent` also receives the agent tokenizer override, or the model tokenizer when no override is configured.

Use `AppendSystemInstructions`, `AppendContextSource`, and `SetInput` when the prompt needs to be changed after construction. `SetDebugSink` enables prompt-build diagnostics.

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

- `Definition` combines a name, model, tools, prompt factory, limits, optional tokenizer override, optional tool-response processor, and ordered stream middleware.
- Agent tools are exposed automatically through a `tool_definitions` source placed before application context sources.
- `Prompt` builds a `context.PromptBuilder` for one `RunInput`.
- `RunInput` carries an ID, structured `PromptInput`, per-run output-token override, and metadata. `PromptInput` separates genuine user content from named machine context.
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

#### Retry streaming trade-off

Workflow tokens are forwarded immediately. If a model attempt fails and is
retried, its earlier tokens remain in the token stream and workflow result; the
retry status sets `DiscardIteration` so consumers that maintain visible state can
retract that attempt. This preserves real-time streaming. Applications requiring
only accepted-attempt output must consume the ordered `loop.Event` stream and
correlate tokens with its retry events, or choose buffering and accept delayed
output.

An ordinary agent can become middleware with `NewAgentMiddleware`. By default it
receives the current visible text as named `upstream_output` context, plus the
original run ID and metadata. Set
`AgentMiddlewareConfig.MapInput` to deliberately project the typed upstream
`WorkflowResult` into a different `RunInput`:

- `PreserveOutput` streams the upstream output unchanged and keeps the nested
  agent result only in `WorkflowResult.Stages`.
- `AppendOutput` streams the upstream output, then emits the nested agent output
  after that stage completes successfully.
- `ReplaceOutput` buffers the upstream output and emits the nested agent output
  only after the stage completes successfully.

Failed append and replace stages leave the upstream output unchanged. An agent
used through `NewAgentMiddleware` cannot define middleware of its own; compose
stages on the parent workflow instead.

Agent middleware runs only after a successful upstream result by default. Set
`AgentMiddlewareConfig.ShouldRun` to implement policies such as failure auditing.
Nested-agent errors are sent through the workflow error channel by default. Set
`ErrorPolicy: RecordError` for best-effort stages whose failures should remain in
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

Set `Definition.DebugSink` to receive structured agent lifecycle events for run
creation, workflow start/completion, primary completion, and middleware
start/skip/success/failure. Agent execution also emits `agent.run.create`,
`agent.workflow.run`, and `agent.middleware.run` OpenTelemetry spans. Event
payloads contain counts and policy names by default; input and output text are
included only when `DebugSink.IncludeSensitiveData()` returns true.

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
- an optional tool-response processor

The loop calls `BuildContext` once before the first iteration, then calls
`BuildPrompt` for each iteration so the rendered prompt includes the latest
assistant and tool messages. `Loop.Run` returns one ordered `Event` stream with
attempt starts, tokens, retries, completed iterations, and terminal errors. The
loop stops when the model returns a normal response or when the maximum
iteration count is reached.

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
    Params() ai.ToolParameters
    Function(ctx context.Context, req *ai.ToolCall) *ToolResponse
}
```

`Params` returns a structured object schema. The runtime serializes it to JSON
Schema for provider-native tool calling and for the prompt fallback:

```go
func (t *SaveMemoryTool) Params() ai.ToolParameters {
    return ai.ToolParameters{
        Strict: true,
        Properties: []ai.ToolParameter{
            {
                Name:        "memory",
                Type:        ai.ToolParameterString,
                Description: "The memory text to save.",
                Required:    true,
            },
        },
    }
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
When a provider supports native tool calling, the loop sends tool definitions
through `AIRequest.Tools`. Prompt-rendered JSON tool calls are retained as a
fallback compatibility protocol.

</details>

<details>

<summary>

### 🧪 Helper Functions


</summary>

- `ai.DetectToolCallsInStream` detects text-encoded tool-call JSON objects in streamed text tokens.
- `loop.CallTool` runs a tool by name.
- `loop.DecodeToolArgs` unmarshals tool arguments into a typed struct.
- `context.Renderer.RenderToolSignatures` formats tool metadata for prompting and returns schema errors.

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
