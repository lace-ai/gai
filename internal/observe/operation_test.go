package observe_test

import (
	"context"
	"errors"
	"testing"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/internal/observe"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestOperationPreservesSpanAndDebugEventSemantics(t *testing.T) {
	previousProvider := otel.GetTracerProvider()
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	otel.SetTracerProvider(provider)
	t.Cleanup(func() {
		_ = provider.Shutdown(context.Background())
		otel.SetTracerProvider(previousProvider)
	})

	var events []gai.DebugEvent
	sink := gai.DebugSinkFunc(func(_ context.Context, event gai.DebugEvent) {
		events = append(events, event)
	})
	ctx, operation := observe.Start(t.Context(), sink, "test-tracer", "test", "test.operation", "run", "test:observer", attribute.String("test.initial", "value"))
	operation.Set(attribute.Int("test.count", 2))
	operation.Emit(ctx, "completed", map[string]any{"result": "ok"}, nil)
	operation.Finish(errors.New("failed"))

	if len(events) != 1 {
		t.Fatalf("events = %d, want 1", len(events))
	}
	if events[0].Name != "completed" || events[0].Source != "test:observer" || events[0].Fields["result"] != "ok" {
		t.Fatalf("event = %#v", events[0])
	}

	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("spans = %d, want 1", len(spans))
	}
	span := spans[0]
	if span.Name() != "test.run" || span.Status().Code.String() != "Error" {
		t.Fatalf("span = %q status=%s", span.Name(), span.Status().Code)
	}
	attrs := map[string]string{}
	for _, attr := range span.Attributes() {
		attrs[string(attr.Key)] = attr.Value.Emit()
	}
	if attrs["test.operation"] != "run" || attrs["test.initial"] != "value" || attrs["test.count"] != "2" {
		t.Fatalf("attrs = %#v", attrs)
	}
}

func TestOperationHandlesNilSink(t *testing.T) {
	ctx, operation := observe.Start(t.Context(), nil, "test-tracer", "test", "test.operation", "run", "test:observer")
	operation.Emit(ctx, "completed", nil, nil)
	operation.Finish(nil)
}
