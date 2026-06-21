// Package context builds the model-facing context used by GAI agents.
//
// A Builder combines system instructions, dynamic ContextSource values, the
// current user prompt, and messages from a Conversation. Parts first produce a
// renderer-neutral RenderNode tree, which a Renderer converts into the final
// prompt string. Optional token budgets reserve space for model output and limit
// how much dynamic context is included.
//
// This package is commonly imported with an alias such as gaictx to distinguish
// it from the standard library context package.
package context
