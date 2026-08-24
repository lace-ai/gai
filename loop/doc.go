// Package loop executes iterative model conversations with function tools.
//
// It is GAI's deliberate low-level execution API: Loop exposes mutable run
// configuration and completed Iterations for applications that need direct
// orchestration control. Higher-level applications should use agent for
// reusable definitions, per-run configuration, workflow lifecycle, middleware,
// and aggregated copied results.
//
// A Loop builds a prompt, streams model tokens, executes requested Tools, adds
// their results to subsequent prompts, and stops when the model returns a final
// response. Each run exposes one ordered Event stream containing tokens,
// attempt starts, retries, completed iterations, and terminal results.
// Iteration values retain the transcript needed to render later prompt turns or
// inspect the completed run.
package loop
