// Package obstest provides OpenTelemetry assertions shared by provider tests.
package obstest

import (
	"context"
	"testing"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// Install installs an in-memory span recorder for the duration of the test.
func Install(t testing.TB) *tracetest.SpanRecorder {
	t.Helper()
	previous := otel.GetTracerProvider()
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	otel.SetTracerProvider(provider)
	t.Cleanup(func() {
		_ = provider.Shutdown(context.Background())
		otel.SetTracerProvider(previous)
	})
	return recorder
}

// RequireGenerationSpansEventually waits for asynchronous provider cleanup to
// end exactly count generation spans.
func RequireGenerationSpansEventually(t testing.TB, recorder *tracetest.SpanRecorder, count int, timeout time.Duration) []sdktrace.ReadOnlySpan {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for {
		spans := generationSpans(recorder)
		if len(spans) == count {
			return spans
		}
		if time.Now().After(deadline) {
			t.Fatalf("generation spans = %d, want %d before %s", len(spans), count, timeout)
		}
		time.Sleep(time.Millisecond)
	}
}

// RequireGenerationSpans returns exactly count ended semantic generation spans.
func RequireGenerationSpans(t testing.TB, recorder *tracetest.SpanRecorder, count int) []sdktrace.ReadOnlySpan {
	t.Helper()
	spans := generationSpans(recorder)
	if len(spans) != count {
		t.Fatalf("generation spans = %d, want %d", len(spans), count)
	}
	return spans
}

func generationSpans(recorder *tracetest.SpanRecorder) []sdktrace.ReadOnlySpan {
	spans := make([]sdktrace.ReadOnlySpan, 0)
	for _, span := range recorder.Ended() {
		if Attributes(span)["langfuse.observation.type"].AsString() == "generation" {
			spans = append(spans, span)
		}
	}
	return spans
}

// Attributes returns span attributes indexed by key.
func Attributes(span sdktrace.ReadOnlySpan) map[string]attribute.Value {
	result := make(map[string]attribute.Value, len(span.Attributes()))
	for _, attr := range span.Attributes() {
		result[string(attr.Key)] = attr.Value
	}
	return result
}
