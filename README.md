# 🤖 GAI (Go Agent Interface or Go AI)

![GAI Logo](/docs/GAI_thumbnail.png)

[![GitHub go.mod Go version](https://img.shields.io/github/go-mod/go-version/HecoAI/gai)](#)
[![CI](https://img.shields.io/badge/ci-passing-brightgreen.svg)](#)
[![License](https://img.shields.io/badge/license-LGPL-informational.svg)](#copyright-and-license)

GAI is a flexible Go library for building agent-style applications on top of LLMs.
It provides a generic interface for providers and models, prompt and context helpers, and a loop for agentic-calling workflows.

## ✨ Overview

The library is organized around three ideas:

- 🧩 `ai` defines the core provider, model, request, and response abstractions.
- 🗂️ `context` stores conversations, renders message history, and loads prompt files.
- 🔁 `loop` runs iterative model and tool execution when a model returns a tool call.

## 📋 Requirements

- Go `1.26.1` or newer
- API credentials for whichever provider you use

## 🚀 Quick Start

### 📦 Installation

```bash
go get github.com/HecoAI/gai
```

### 🧭 Usage

#### 🛠️ Creating Providers and Models

To start, first create a provider. For example, for Gemini:

```go
geminiProvider := gemini.New("your_api_key")
```

<details>

<summary>🗂️ Use the Model Repository to manage multiple providers and dynamic model selection</summary>

You can use a `ModelRepository` to register multiple providers and look up models by name across providers.

```go
modelRepo := ai.NewModelRepository()
err := modelRepo.RegisterProvider(geminiProvider)
if err != nil {
    // handle error
}
```

To get a model from the repo just use the provider name and the model name:

```go
model, err := modelRepo.GetModel("gemini", "gemini-3-flash-preview")
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

But you can implement your own provider by implementing the `Provider` and `Model` interfaces defined in the `ai` package.

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

func (m *MyModel) Close() error {
    // clean up any resources if needed
}
```

Now you can use your custom provider just like the built-in ones

</details>

### 🔁 Agentic Tool Calling

To build an agent with tools, use the `loop` package:

> [!TIP]
> User a alias for the `context` package to avoid conflicts with `context` package from the standard library. For example:
>
> ```go
> import aicontext "github.com/HecoAI/gai/context"
> ```

```go
agentLoop := loop.New(
    model, // the model you want to use
    []loop.Tool{myTool}, // any tools you want to provide, one echo too is included for testing
    "What is the weather in New York?", // initial (user) prompt
    "You are a helpful assistant that can call tools to get information.", // system prompt
    nil, // optional context builder, if nil the loop will render prior messages itself
    nil, // optional tool response preprocessor, if nil the loop will append tool results as-is
)

err := agentLoop.Loop(context.Background())
if err != nil {
    // handle error
}
messages := agentLoop.Messages() // get final conversation messages, including tool calls and responses

var builder strings.Builder
aicontext.RenderMessages(messages, &builder)
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

func (t *MyTool) Function(req *loop.ToolRequest) (*loop.ToolResponse, error) {
    var args myToolArgs
    if err := loop.DecodeToolArgs(req, &args); err != nil {
        return nil, err
    }

    // implement your tool logic here using args.Query
    return &loop.ToolResponse{Text: "result for: " + args.Query}, nil
}
```

Then include an instance of your tool in the `loop.New(...)` call.

</details>

### 🧠 Session Management

To manage conversation history and build prompts from it, use the `context` package:

```go
store := mySessionStore // your implementation of SessionStore (e.g. in-memory, database, etc.)
sessionManager := aicontext.NewSessionManager(store, 1) // the second argument is the session ID

agentLoop := loop.New(
    model, // the model you want to use
    []loop.Tool{myTool}, // any tools you want to provide
    "What is the weather in New York?", // initial (user) prompt
    "You are a helpful assistant that can call tools to get information.", // system prompt
    sessionManager, // session manager implements loop.ContextBuilder
    nil, // optional tool response preprocessor
)
err := agentLoop.Loop(context.Background())
if err != nil {
    // handle error
}
```

<details>

<summary>🗄️ Implement Your Own Session Store</summary>

To implement your own session store, please visit the [SessionStore interface](./context/session_store.go) and implement the required methods.

</details>

## 🧱 Package Layout

```text
ai/          Core abstractions: Provider, Model, AIRequest, AIResponse, ModelRepository
ai_gemini/   Gemini provider and model implementation
ai_mistral/  Mistral provider and model implementation
context/     Context management: Conversation/session types, prompt loading, message rendering
loop/        Agent loop, tool parsing, tool execution helpers
testutil/    Mocks used by tests
```

## 🧩 Core Concepts

### 🏢 Provider

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

### 🧠 Model

A model generates text from an `AIRequest` and returns an `AIResponse`.

```go
type Model interface {
    Name() string
    Generate(ctx context.Context, req AIRequest) (*AIResponse, error)
    Close() error
}
```

### 📝 Prompt and Request

`ai.Prompt` combines three pieces of input:

- `System`: system instructions
- `Context`: prior conversation or external context
- `Prompt`: the current (user) request

`Prompt.CombinedPrompt()` concatenates those parts onto one string in that order.

`AIRequest` currently contains:

- `Prompt`
- `MaxTokens` maxtokens are ignored by some providers, and might be removed in future versions.

`AIResponse` returns:

- `Text`
- `InputTokens`
- `OutputTokens`

## 🌐 Providers

### ♊ Gemini

Package: `ai_gemini`

Constructor:

```go
gemini.New(apiKey string) *gemini.Provider
```

Known model names:

- `gemini-3-flash-preview`
- `gemini-2.5-flash`
- `gemini-3.1-flash-lite-preview`
- `gemini-2.5-flash-lite`

### 🌀 Mistral

Package: `ai_mistral`

Constructor:

```go
mistral.New(apiKey string) *mistral.Provider
```

Known model names:

- `mistral-small-latest`
- `mistral-medium-latest`
- `mistral-large-latest`
- `codestral-latest`

### ⚙️ Configuration Note

This library does not read environment variables automatically.
Create the provider with the API key you want to use, then register it in the repository.

## 💾 Context and Sessions

The `context` package is not the standard library `context` package.
Import it with an alias such as `aicontext` to avoid name collisions.

```go
import aicontext "github.com/HecoAI/gai/context"
```

### 📨 Messages

Messages have one of four roles:

- `system`
- `user`
- `assistant`
- `tool`

Each message wraps a `Content` implementation such as text, tool calls, or tool results, (you can also implement your own).
The renderer formats history as tagged blocks, which is what the loop uses when it builds context automatically.

### 💬 Conversation

`Conversation` is a minimal interface used by the `SessionManager` to load and render message history:

```go
type Conversation interface {
    Messages() []Message
}
```

### 🗃️ SessionStore

`SessionStore` is an interface, not a built-in database implementation.
You provide your own store that can:

- create sessions
- fetch sessions and messages
- add one or many messages

### 🧭 SessionManager (WIP)

`SessionManager` builds prompt context from stored history.
It loads the last 5 messages for the configured session, renders them, and appends the current loop messages.

> [!NOTE]
> `NewSessionManager(store, id)` expects an integer session ID. If you want to start a new session, create one first.

### 📄 Prompt Files

`LoadPromptFromFile` reads `.md` and `.txt` files, trims whitespace, and returns the prompt text.

## 🔄 Loop and Tools

The `loop` package is for agent-style execution where the model can request tool calls.

### 🔁 Loop

`loop.New(...)` creates a loop with:

- a model
- optional tools
- an initial user prompt
- an optional system prompt
- an optional context builder
- an optional tool-response preprocessor

If no context builder is provided, the loop renders prior messages itself.

The loop stops when the model returns a normal response or when the maximum iteration count is reached.

### 🧰 Tool Interface

Tools must implement:

```go
type Tool interface {
    Name() string
    Description() string
    Params() string
    Function(req *ToolRequest) (*ToolResponse, error)
}
```

Tool calls are expected to arrive as JSON with this shape:

```json
{
  "id": "tool_name",
  "type": "function",
  "arguments": {
    "some": "value"
  }
}
```

> [!TIP]
> Keep tool `Params()` aligned with the JSON fields your `Function(...)` decodes through `DecodeToolArgs`.

### 🧪 Helper Functions

- `DetectToolCall` checks whether a model response looks like a tool call.
- `CallTool` runs a tool by name.
- `DecodeToolArgs` unmarshals tool arguments into a typed struct.
- `RenderToolSignatures` formats tool metadata for prompting.

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
- `SessionManager` currently uses a fixed history window of 5 messages.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.
If you add a new provider or tool, document the new constructor, model names, and any required environment variables.

## 📜 Copyright and License

This library is licensed under the GNU LESSER GENERAL PUBLIC LICENSE v2.1. See [LICENSE](LICENSE) for details.

Copyright (c) 2024 HecoAI. All rights reserved.
