package loop

import (
	"context"
	"testing"

	"github.com/lace-ai/gai/ai"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestToolSpanPreservesMissingResponseError(t *testing.T) {
	previousProvider := otel.GetTracerProvider()
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	otel.SetTracerProvider(provider)
	t.Cleanup(func() {
		_ = provider.Shutdown(context.Background())
		otel.SetTracerProvider(previousProvider)
	})

	_, finish := startToolSpan(t.Context(), ai.ToolCall{ID: "call-missing", Name: "missing"})
	finish(nil)

	for _, span := range recorder.Ended() {
		if span.Name() == "loop.tool" {
			if got := span.Status().Description; got != ErrToolErrorMissing.Error() {
				t.Fatalf("tool span status = %q, want %q", got, ErrToolErrorMissing)
			}
			return
		}
	}
	t.Fatal("loop.tool span was not recorded")
}
