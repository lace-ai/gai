# Provider-native multi-turn messages: implementation plan

Issue: #45

## Current compatibility contract

The loop renders the complete conversation—including assistant tool calls and
tool results—into `ai.AIRequest.Prompt`. Provider adapters consume that prompt
and the existing provider-neutral tool fields. This remains the only execution
path in this change.

The request construction is now isolated in `renderedPromptRequest` so a future
native-message path can be added beside it without changing today's rendered
prompt behavior.

## Non-goals of this change

- No `AIRequest` native-message field is added.
- No Gemini or Mistral request payload changes are made.
- No provider capability detection or fallback selection is added.
- No conversation, tool-call, or tool-result serialization changes are made.

## Follow-up implementation sequence

1. Define provider-neutral request-message types that preserve roles, tool-call
   identifiers, arguments, result content, and error state.
2. Add an opt-in message-history field to `AIRequest` while retaining `Prompt`
   as the compatibility fallback. Document precedence and empty-history rules.
3. Convert loop iterations into that message history at the request boundary;
   keep the existing rendered prompt available for providers without native
   message support.
4. Implement adapter-specific mappings and capability checks one provider at a
   time. Use the rendered prompt only for adapters that do not opt in.
5. Add cross-provider tests for a user turn, assistant tool call, tool result,
   and final assistant response. Keep regression tests asserting unchanged
   rendered-prompt behavior for fallback providers.
6. Update public documentation and release notes once at least one provider
   uses the native path, including any compatibility or migration guidance.