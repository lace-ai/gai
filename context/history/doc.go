// Package history provides persisted conversation history as a GAI context
// source.
//
// HistorySource loads turns through a HistoryStore, selects the newest history
// that fits the available prompt budget, and renders it as one context part. An
// optional summarizer can compact older turns when the complete history no
// longer fits. Persisted history remains canonical; token-budget trimming only
// changes the prompt projection produced for the current run.
package history
