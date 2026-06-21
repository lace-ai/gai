// Package loop executes iterative model conversations with function tools.
//
// A Loop builds a prompt, streams model tokens, executes requested Tools, adds
// their results to subsequent prompts, and stops when the model returns a final
// response. Each run exposes separate token, iteration-status, and error
// streams. Iteration values retain the transcript needed to render later prompt
// turns or inspect the completed run.
package loop
