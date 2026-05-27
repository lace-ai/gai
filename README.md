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
- 🗂️ `context` composes prompts, budgets context, renders message history, and loads prompt files.
- 🔁 `loop` runs iterative model and tool execution when a model returns a tool call.

## 📋 Requirements

- Go `1.26.x` or newer
- API credentials for whichever provider you use

## 🚀 Quick Start

### 📦 Installation

```bash
go get github.com/lace-ai/gai
```

### 🧭 Usage

#### 🛠️ Creating Providers and Models

To start, first create a provider. For example, for Gemini:

```go
geminiProvider := gemini.New("your_api_key", nil)
```

<details>

<summary>🗂️ Use the Model Repository to manage multiple providers and dynamic model selection</summary>

You can use a `ModelRepository` to register multiple providers and look up models by name across providers.

```go
modelRepo := ai.NewModelRepository(nil)
err := modelRepo.RegisterProvider(context.Background(), geminiProvider)
if err != nil {
    // handle error
}
```

To get a model from the repo just use the provider name and the model name:

```go
model, err := modelRepo.GetModel(context.Background(), "gemini", "gemini-3-flash-preview")
if err != nil {
    // handle error
}
```

</details>

#### 💬 Generate Text

Now you can access models from that provider, and generate text:

```go
model, err := geminiProvider.Model("gemini-3-flash-preview")
if err != nil {
    // handle error
}
response, err := model.Generate(context.Background(), ai.AIRequest{
    Prompt: ai.Prompt{
        System: "You are a helpful assistant.",
        Prompt: "What is the capital of France?",
    },
    MaxTokens: 100,
})
```

<details>

<summary>🔌 Implement Your Own Provider</summary>

Currently, the library includes Gemini and Mistral implementations. Gemini uses the official `go-genai` library, and Mistral uses direct HTTP calls to the Mistral API.

But you can implement your own provider by implementing the `Provider`, `Model`, and `Tokenizer` interfaces defined in the `ai` package.

### Example:

**Provider Implementation:**

```go
type MyProvider struct {
    // any configuration fields you need, e.g. API key
}

func (p *MyProvider) Name() string {
    return "myprovider"
}

func (p *MyProvider) Model(name string) (ai.Model, error) {
    // return a model implementation based on the name
}

func (p *MyProvider) ListModels() ([]string, error) {
    // return a list of available model names
}

func (p *MyProvider) Validate() error {
    // validate the provider configuration, e.g. check API key is set
}
```

**Model Implementation:**

```go
type MyModel struct {
    // any configuration fields you need, e.g. model name, provider reference
    name string
}

func (m *MyModel) Name() string {
    return m.name
}

func (m *MyModel) Generate(ctx context.Context, req ai.AIRequest) (*ai.AIResponse, error) {
    // implement the logic to call your model API and return the response
}

func (m *MyModel) GenerateStream(ctx context.Context, req ai.AIRequest) <-chan ai.Token {
    // implement streaming token generation
}

func (m *MyModel) Close() error {
    // clean up any resources if needed
}

func (m *MyModel) Tokenizer() ai.Tokenizer {
    // return a tokenizer implementation for your model
}
```

**Tokenizer Implementation:**

```go
type MyTokenizer struct{}

func (t MyTokenizer) ID() string {
    return "myprovider.my-model"
}

func (t MyTokenizer) Tokenize(ctx context.Context, text string) ([]string, error) {
    // split text into model-specific tokens
}

func (t MyTokenizer) CountTokens(ctx context.Context, text string) (int, error) {
    // return the token count for text
}
```

Now you can use your custom provider just like the built-in ones

</details>

### 🔁 Agentic Tool Calling

To build an agent with tools, use the `loop` package:

> [!TIP]
> Use an alias for the `context` package to avoid conflicts with `context` package from the standard library. For example:
>
> ```go
> import gaictx "github.com/lace-ai/gai/context"
> ```

```go
prompt := gaictx.NewPromptBuilder().
    System(
        "base-system",
        "You are a helpful assistant that can call tools to get information.",
        gaictx.Required(),
    ).
    User(
        "request",
        "What is the weather in New York?",
        gaictx.Required(),
    )

l := loop.New(
    model, // the model you want to use
    []loop.Tool{myTool}, // any tools you want to provide, one echo tool is included for testing
    prompt, // structured prompt builder from the context package
    nil, // optional tool response preprocessor, if nil the loop will append tool results as-is
)

tokenCh, _, errCh := l.Loop(context.Background())
for range tokenCh {
    // handle streamed tokens
}
for err := range errCh {
    if err != nil {
        // handle error
    }
}
messages := l.Messages() // get final conversation messages, including tool calls and responses

var builder strings.Builder
gaictx.RenderMessages(messages, &builder)
fmt.Println(builder.String()) // render the messages for display
```

<details>

<summary>🧩 Implement Your Own Tool</summary>

To implement your own tool, create a struct that implements the `Tool` interface:

```go
type myToolArgs struct {
    Query string `json:"query"`
}

type MyTool struct {
    // any configuration fields you need
}

func (t *MyTool) Name() string {
    return "my_tool"
}

func (t *MyTool) Description() string {
    return "A tool that does something useful."
}

func (t *MyTool) Params() string {
    return `{"type":"object","required":["query"],"properties":{"query":{"type":"string","description":"Search query"}}}`
}

func (t *MyTool) Function(req *ai.ToolCall) *loop.ToolResponse {
    var args myToolArgs
    if err := loop.DecodeToolArgs(req, &args); err != nil {
        return &loop.ToolResponse{Err: err}
    }

    // implement your tool logic here using args.Query
    return &loop.ToolResponse{Text: "result for: " + args.Query}
}
```

Then include an instance of your tool in the `loop.New(...)` call.

</details>

### 🧠 Session Management

To manage conversation history and build prompts from it, use the `context` package:

```go
store := mySessionStore // your implementation of SessionStore (e.g. in-memory, database, etc.)
sessionID := 1
prompt := gaictx.NewPromptBuilder().
    Budget(gaictx.PromptBudget{
        Tokenizer:            model.Tokenizer(),
        ContextWindowTokens:  128000,
        ReservedOutputTokens: 4096,
    }).
    System(
        "base-system",
        "You are a helpful assistant that can call tools to get information.",
        gaictx.Required(),
    ).
    Source(gaictx.SectionContext, "history", gaictx.History(store, sessionID), gaictx.Required(), gaictx.SourceTokenCap(10000)).
    User(
        "request",
        "What is the weather in New York?",
        gaictx.Required(),
    )

l := loop.New(
    model, // the model you want to use
    []loop.Tool{myTool}, // any tools you want to provide
    prompt, // prompt builder owns system, history, and user prompt parts
    nil, // optional tool response preprocessor
)
tokenCh, _, errCh := l.Loop(context.Background())
for range tokenCh {
    // handle streamed tokens
}
for err := range errCh {
    if err != nil {
        // handle error
    }
}
```

<details>

<summary>🗄️ Implement Your Own Session Store</summary>

To implement your own session store, please visit the [SessionStore interface](./context/store.go) and implement the required methods.

</details>

## 🧱 Package Layout

```text
agent/       Reusable loop-backed agent definitions and built-in agents such as summary.
ai/          Core abstractions: Provider, Model, Tokenizer, AIRequest, AIResponse, ModelRepository. With implementations for Gemini and Mistral.
context/     Prompt building, token budgeting, dynamic sources, sessions, RAG, summaries, and message rendering.
loop/        Agent loop, tool parsing, tool execution helpers.
testutil/    Mocks used by tests.
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

### 📝 Prompt and Request

</summary>

`ai.Prompt` combines three flat strings that providers send upstream:

- `System`: system instructions
- `Context`: prior conversation or external context
- `Prompt`: the current (user) request

`Prompt.CombinedPrompt()` concatenates those parts onto one string in this order: system, context, prompt.
For agent loops, prefer building this value through `context.NewPromptBuilder()` so system, context, and user prompt content can be composed from structured parts and dynamic sources.

`AIRequest` contains:

- `Prompt`
- `MaxTokens`, which providers pass through where the upstream API supports an output-token cap

`AIResponse` returns:

- `Text`
- `InputTokens`
- `OutputTokens`

## 🌐 Providers

</details>

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

## 💾 Context and Sessions

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

Each message wraps a `Content` implementation such as text, tool calls, or tool results, (you can also implement your own).
`RenderMessages` formats history as tagged blocks for prompt sources such as `History`.


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

### 🗃️ SessionStore


</summary>

`SessionStore` is an interface, not a built-in database implementation.
You provide your own store that can:

- create sessions
- fetch sessions and messages
- add one or many messages
- update cached per-message token counts

Token counts are keyed by `Tokenizer.ID()` and represent the message content, not the rendered role wrapper produced by `RenderMessages`.

</details>

<details>

<summary>

### 🔎 RAGStore

</summary>

`RAGStore` is an interface for retrieval sources. It can:

- return relevant documents for a query
- add documents
- update cached per-document token counts

</details>

<details>

<summary>

### 🧱 PromptBuilder

</summary>

`PromptBuilder` composes the full `ai.Prompt` from structured parts:

```go
prompt := gaictx.NewPromptBuilder().
    Budget(gaictx.PromptBudget{
        Tokenizer:            model.Tokenizer(),
        ContextWindowTokens:  128000,
        ReservedOutputTokens: 4096,
    }).
    System("base-system", "Follow the system policy.", gaictx.Required()).
    Source(gaictx.SectionContext, "history", gaictx.History(store, sessionID), gaictx.Required(), gaictx.SourceTokenCap(1000)).
    Source(gaictx.SectionContext, "rag", ragSource, gaictx.Optional(), gaictx.SourceTokenCap(2000)).
    User("request", "Summarize the project status.", gaictx.Required())
```

Entries use stable IDs and are rendered in fixed section order (`system`, `context`, `user`). Inside each section, required entries render before optional entries; configured order is preserved within the required group and within the optional group. This lets required sources receive budget before optional context can consume it.

With `Budget(...)` configured, the builder counts the rendered prompt with the configured tokenizer. `ContextWindowTokens - ReservedOutputTokens` is the prompt budget, and `ConversationReserveTokens` reserves space for loop messages appended by incremental prompt sessions. Required entries must fit or `BuildPrompt` returns `ErrPromptBudget`. Optional entries are best-effort: the builder includes them when they fit, asks the configured `Summarizer` to compact optional static parts when they do not fit, and drops them when there is no fitting summary. Optional source failures are skipped, and optional source overflow is dropped.

Dynamic sources implement:

```go
type Source interface {
    BuildParts(ctx context.Context, view PromptView, budget SourceBudget) ([]Part, error)
}
```

`PromptView` exposes the current conversation and a read-only view of the whole configured prompt plan, so sources can inspect planned entries by ID or section before emitting parts. `SourceBudget` provides the source's cap, remaining prompt budget, tokenizer, render-overhead reserve ratio, required flag, and optional summarizer. `SourceTokenCap(...)` limits one source's budget; uncapped sources receive the remaining prompt budget.

The default renderer is XML-like and can be replaced with a custom renderer. Grouped parts render as one outer part with child items, which lets sources like RAG budget individual documents without adding a full XML wrapper around every document. `LastTrace()` returns the most recent build trace, including emitted, skipped, dropped, summarized, token counts, available tokens, and reasons. `Debug(gai.DebugSink)` emits the same prompt-build decisions. Rendered part text is only included in debug events when the sink allows sensitive data.

</details>

<details>

<summary>

### 🧭 History Sources

</summary>

`History(store, sessionID)` returns a prompt source that loads stored messages and renders fitting messages as `history-*` parts. Use `SourceTokenCap(...)` on the source entry to control how many tokens history may consume. History reuses cached message token counts when all messages in a batch have a non-negative value for the active tokenizer; otherwise it counts message content and asks the store to save the count asynchronously. Stored messages that exceed the source budget are left out.


</details>

<details>

<summary>

### 🔎 RAG Sources

</summary>

`RAG(store, documentLimit, queryFunc)` returns a prompt source that queries a `RAGStore`, budgets relevant documents in relevance order, and emits a grouped `rag` part. Each document remains a child part for tracing and budgeting, while rendering stays compact. Overflow documents are summarized when a summarizer is available and budget remains; required RAG fails with `ErrPromptBudget` if no document or summary can fit.

</details>

<details>

<summary>

### 🤖 Agent Definitions and Built-ins

</summary>

The `agent` package provides a small factory for reusable loop-backed agents. A definition combines a model, optional tools, a prompt-builder factory, and loop settings.

Built-in agents live in their own packages. `agent/summary` embeds its default system prompt from `system.md` and exposes a summarizer that implements the context summarizer interface by running a summary agent through `loop`, collecting streamed text tokens, and returning the summary. Pass that summarizer through `PromptBudget.Summarizer` when optional prompt content should be compacted before it is dropped.

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

When the prompt builder supports incremental prompts, the loop builds the base prompt once and appends each iteration's delta messages through the prompt session. Other prompt builders are called on every loop iteration, so dynamic sources can include the current loop conversation, session history, summaries, RAG results, or other runtime context.

The loop stops when the model returns a normal response or when the maximum iteration count is reached.

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
    Function(req *ai.ToolCall) *ToolResponse
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
- `RenderToolSignatures` formats tool metadata for prompting.

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
- `context.ErrSessionNotFound`
- `context.ErrSessionStoreNotFound`
- `context.ErrRAGStoreNotFound`
- `context.ErrPromptBuilderNil`
- `context.ErrPromptEntryID`
- `context.ErrPromptSource`
- `context.ErrPromptBudget`
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
- History and RAG token caches store content-token counts by tokenizer ID. Rendered wrappers and renderer overhead are handled by prompt budgeting, not by those cached per-item counts.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.
If you add a new provider or tool, document the new constructor, model names, and any required environment variables.

## 📜 Copyright and License

This library is licensed under the GNU LESSER GENERAL PUBLIC LICENSE v2.1. See [LICENSE](LICENSE) for details.

Copyright (c) 2026 lace-ai. All rights reserved.
