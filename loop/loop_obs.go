package loop

import (
	"context"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

const loopTracerName = "github.com/lace-ai/gai/loop"

type loopRunState struct {
	obs        *loopObserver
	err        error
	retryCount int
	stats      loopRunStats
}

type loopRunStats struct {
	IterationCount    int
	TokenCount        int
	ToolCallCount     int
	IncrementalPrompt bool
}

type loopIterationState struct {
	obs   *iterationObserver
	stats loopIterationStats
}

type loopIterationStats struct {
	Retrying       bool
	Final          bool
	AttemptID      int
	PartCount      int
	RetryCount     int
	ToolCallCount  int
	ToolErrorCount int
}

type loopObserver struct {
	span trace.Span
}

func newLoopRunState(ctx context.Context, l *Loop) (context.Context, *loopRunState) {
	maxIterations := 0
	retryLimit := 0
	maxTokens := 0
	toolCount := 0
	modelName := ""
	if l != nil {
		maxIterations = l.MaxLoopIterations
		retryLimit = l.RetryCount
		maxTokens = l.MaxTokens
		toolCount = len(l.Tools)
		if l.Model != nil {
			modelName = l.Model.Name()
		}
	}

	ctx, span := gai.StartOperationSpan(ctx, loopTracerName, "loop", "loop.operation", "run",
		attribute.Int("loop.max_iterations", maxIterations),
		attribute.Int("loop.retry_limit", retryLimit),
		attribute.Int("loop.max_tokens", maxTokens),
		attribute.Int("loop.tool_count", toolCount),
		attribute.String("ai.model", modelName),
	)
	return ctx, &loopRunState{obs: &loopObserver{span: span}}
}

func (s *loopRunState) startIteration(ctx context.Context, count int, attempt int) (context.Context, *loopIterationState) {
	incrementalPrompt := false
	if s != nil {
		s.stats.IterationCount = count
		incrementalPrompt = s.stats.IncrementalPrompt
	}
	ctx, span := gai.StartOperationSpan(ctx, loopTracerName, "loop", "loop.operation", "iteration",
		attribute.Int("loop.iteration", count),
		attribute.Int("loop.attempt", attempt),
		attribute.Bool("loop.incremental_prompt", incrementalPrompt),
	)
	return ctx, &loopIterationState{
		obs:   &iterationObserver{span: span},
		stats: loopIterationStats{AttemptID: attempt},
	}
}

func (s *loopRunState) recordToken(token ai.Token) {
	if s == nil {
		return
	}
	s.stats.TokenCount++
	if token.Type == ai.TokenTypeToolCall && token.ToolCall != nil {
		s.stats.ToolCallCount++
	}
}

func (s *loopRunState) canRetry(limit int) bool {
	if s == nil {
		return false
	}
	return s.retryCount < limit
}

func (s *loopRunState) retry() {
	if s == nil {
		return
	}
	s.retryCount++
}

func (s *loopRunState) resetRetries() {
	if s == nil {
		return
	}
	s.retryCount = 0
}

func (s *loopRunState) retryStatus(iteration Iteration) IterationInformation {
	retryCount := 0
	if s != nil {
		retryCount = s.retryCount
	}
	return IterationInformation{
		IterationCount:   iteration.Count,
		PartCount:        len(iteration.Parts),
		RetryCount:       retryCount,
		Retrying:         true,
		AttemptID:        retryCount,
		DiscardIteration: true,
	}
}

func (s *loopRunState) fail(err error) error {
	if s == nil {
		return err
	}
	s.err = err
	return err
}

func (s *loopRunState) finish() {
	if s == nil || s.obs == nil {
		return
	}
	s.obs.finish(s.err, s.stats)
}

func (o *loopObserver) finish(err error, stats loopRunStats) {
	if o == nil || o.span == nil {
		return
	}
	o.span.SetAttributes(
		attribute.Int("loop.iteration_count", stats.IterationCount),
		attribute.Int("loop.token_count", stats.TokenCount),
		attribute.Int("loop.tool_call_count", stats.ToolCallCount),
		attribute.Bool("loop.incremental_prompt", stats.IncrementalPrompt),
	)
	gai.EndSpan(o.span, err)
}

type iterationObserver struct {
	span trace.Span
}

func (s *loopIterationState) recordToken(token ai.Token) {
	if s == nil {
		return
	}
	s.stats.PartCount++
	if token.Type == ai.TokenTypeToolCall && token.ToolCall != nil {
		s.stats.ToolCallCount++
	}
}

func (s *loopIterationState) recordToolResponses(iteration Iteration) {
	if s == nil {
		return
	}
	s.stats.PartCount = len(iteration.Parts)
	for _, part := range iteration.Parts {
		if part.ToolResp != nil && part.ToolResp.Err != nil {
			s.stats.ToolErrorCount++
		}
	}
}

func (s *loopIterationState) markRetrying(retryCount int) {
	if s == nil {
		return
	}
	s.stats.Retrying = true
	s.stats.RetryCount = retryCount
}

func (s *loopIterationState) markFinal() {
	if s == nil {
		return
	}
	s.stats.Final = true
}

func (s *loopIterationState) attemptID() int {
	if s == nil {
		return 0
	}
	return s.stats.AttemptID
}

func (s *loopIterationState) finish(err error) {
	if s == nil || s.obs == nil {
		return
	}
	s.obs.finish(err, s.stats)
}

func (o *iterationObserver) finish(err error, stats loopIterationStats) {
	if o == nil || o.span == nil {
		return
	}
	attrs := []attribute.KeyValue{
		attribute.Int("loop.attempt", stats.AttemptID),
		attribute.Int("loop.part_count", stats.PartCount),
		attribute.Int("loop.retry_count", stats.RetryCount),
		attribute.Int("loop.tool_call_count", stats.ToolCallCount),
		attribute.Int("loop.tool_error_count", stats.ToolErrorCount),
	}
	if stats.Retrying {
		attrs = append(attrs, attribute.Bool("loop.retrying", true))
	}
	if stats.Final {
		attrs = append(attrs, attribute.Bool("loop.final_iteration", true))
	}
	o.span.SetAttributes(attrs...)
	gai.EndSpan(o.span, err)
}
