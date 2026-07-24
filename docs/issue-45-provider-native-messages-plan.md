# Provider-native multi-turn messages: implementation plan

Issue: #45

## Goal

Send the current loop's assistant and tool-result history to provider adapters as
structured messages, while preserving the fully rendered `AIRequest.Prompt` as
the compatibility fallback.

## Design

`ai.AIRequest` will receive an opt-in `Messages []RequestMessage` field. Each
message preserves a role, assistant text, structured tool calls (including the
provider call ID and JSON arguments), or a tool result that references its
call ID. Empty history means prompt-only behavior exactly as before.

The loop will always build both forms. It will retain the rendered prompt for
providers or histories that cannot use native messages. It will derive native
history directly from completed `Iteration.Parts`, rather than
`Iteration.Messages()`, because the latter intentionally renders tool data and
does not retain tool-call IDs.

Provider adapters will prefer non-empty native history when they can map it
without losing tool-call correlation:

- OpenAI: assistant `tool_calls` and `tool` messages keyed by `tool_call_id`.
- Anthropic: assistant `tool_use` blocks followed by user `tool_result` blocks.
- Mistral: OpenAI-style `tool_calls` and `tool_call_id` payload fields.
- Gemini: model `FunctionCall` and user `FunctionResponse` parts. Because the
  Gemini SDK has no tool-call ID, duplicate calls to the same function in a
  native history are rejected rather than correlated incorrectly.

Persisted historical context is out of scope: existing persisted tool content
does not retain provider tool-call IDs, so it remains safely rendered into the
fallback prompt. No IDs will be synthesized for it.

## Steps

1. Add provider-neutral request-message types and validation/copy helpers in
   `ai`; document empty-history fallback and native-message precedence.
2. Add a loop test for a tool call followed by its result, run it red, then
   convert completed loop iterations into native request history while keeping
   the existing rendered prompt unchanged.
3. Add red/green provider-builder tests for OpenAI and Anthropic native
   histories, including call ID, JSON arguments, result text, and tool errors.
4. Add red/green provider-builder tests for Mistral and Gemini native
   histories. Cover Gemini's duplicate-function-name safety failure.
5. Update public request/provider documentation. Run focused package tests,
   the complete Go suite, vet, formatting, and `git diff --check`.

## Acceptance checks

- The second request in a user -> assistant tool call -> tool result exchange
  includes structured assistant/tool history with the exact call ID and JSON
  arguments.
- Providers that support native messages receive native payloads; prompt-only
  requests preserve their current payloads.
- Tool errors retain their provider-native error representation.
- Tests cover the multi-turn exchange and provider payload mappings without
  calling live provider APIs.
