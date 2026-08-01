package gai

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestTraceContextNormalizesAndCopies(t *testing.T) {
	input := TraceContext{
		Name:        "  chat  ",
		UserID:      "user-1",
		SessionID:   "session-1",
		Tags:        []string{" alpha ", "alpha", "", "beta"},
		Release:     " 2026.08 ",
		Environment: "staging_1",
		Metadata: map[string]string{
			"tier":         " pro ",
			"invalid key":  "hidden",
			"invalid_key":  "hidden",
			"invalid-key":  "hidden",
			"invalid.key":  "hidden",
			"empty":        "  ",
			"oversized":    strings.Repeat("x", maxTraceMetadataValue+1),
			"validnested1": "yes",
		},
	}
	ctx := WithTraceContext(t.Context(), input)
	input.Tags[0] = "mutated"
	input.Metadata["tier"] = "mutated"

	got, ok := TraceContextFromContext(ctx)
	if !ok {
		t.Fatal("trace context was not installed")
	}
	if got.Name != "chat" || got.UserID != "user-1" || got.SessionID != "session-1" || got.Release != "2026.08" || got.Environment != "staging_1" {
		t.Fatalf("unexpected normalized scalars: %#v", got)
	}
	if strings.Join(got.Tags, ",") != "alpha,beta" {
		t.Fatalf("normalized tags = %#v", got.Tags)
	}
	if len(got.Metadata) != 2 || got.Metadata["tier"] != "pro" || got.Metadata["validnested1"] != "yes" {
		t.Fatalf("normalized metadata = %#v", got.Metadata)
	}

	got.Tags[0] = "changed"
	got.Metadata["tier"] = "changed"
	again, _ := TraceContextFromContext(ctx)
	if again.Tags[0] != "alpha" || again.Metadata["tier"] != "pro" {
		t.Fatalf("returned trace context aliases stored data: %#v", again)
	}
}

func TestTraceContextUsesLangfuseCompatibleStringBoundaries(t *testing.T) {
	valid := strings.Repeat("x", 200)
	invalid := strings.Repeat("x", 201)
	validCtx := WithTraceContext(t.Context(), TraceContext{
		UserID:    valid,
		SessionID: valid,
		Tags:      []string{valid},
		Metadata:  map[string]string{"key1": valid},
	})
	got, _ := TraceContextFromContext(validCtx)
	if got.UserID != valid || got.SessionID != valid || len(got.Tags) != 1 || got.Metadata["key1"] != valid {
		t.Fatalf("200-byte values were omitted: %#v", got)
	}

	invalidCtx := WithTraceContext(t.Context(), TraceContext{
		UserID:    invalid,
		SessionID: invalid,
		Tags:      []string{invalid},
		Metadata:  map[string]string{"key1": invalid},
	})
	got, _ = TraceContextFromContext(invalidCtx)
	if got.UserID != "" || got.SessionID != "" || len(got.Tags) != 0 || len(got.Metadata) != 0 {
		t.Fatalf("201-byte values were retained: %#v", got)
	}
}

func TestTraceContextOmitsInvalidAndOversizedScalars(t *testing.T) {
	ctx := WithTraceContext(t.Context(), TraceContext{
		Name:        strings.Repeat("n", maxTraceScalarBytes+1),
		UserID:      "user\nsecret",
		SessionID:   string([]byte{0xff}),
		Release:     "ok",
		Environment: "us-east.prod",
	})
	got, ok := TraceContextFromContext(ctx)
	if !ok {
		t.Fatal("expected an installed empty trace context")
	}
	if got.Name != "" || got.UserID != "" || got.SessionID != "" || got.Environment != "" || got.Release != "ok" {
		t.Fatalf("invalid values were retained: %#v", got)
	}
}

func TestTraceContextNormalizesEnvironmentToLowercase(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{input: "Production", want: "production"},
		{input: " Staging ", want: "staging"},
		{input: "prod-US", want: "prod-us"},
		{input: "Langfuse-Cloud", want: ""},
	}
	for _, test := range tests {
		t.Run(test.input, func(t *testing.T) {
			ctx := WithTraceContext(t.Context(), TraceContext{Environment: test.input})
			got, _ := TraceContextFromContext(ctx)
			if got.Environment != test.want {
				t.Fatalf("environment = %q, want %q", got.Environment, test.want)
			}
		})
	}
}

func TestTraceContextBoundsCollectionsDeterministically(t *testing.T) {
	tags := make([]string, maxTraceTags+2)
	metadata := make(map[string]string, maxTraceMetadata+2)
	for index := range tags {
		tags[index] = fmt.Sprintf("tag-%02d", index)
		metadata[fmt.Sprintf("key%02d", index)] = fmt.Sprintf("value-%02d", index)
	}
	ctx := WithTraceContext(t.Context(), TraceContext{Tags: tags, Metadata: metadata})
	got, _ := TraceContextFromContext(ctx)
	if len(got.Tags) != maxTraceTags || got.Tags[0] != "tag-00" || got.Tags[maxTraceTags-1] != "tag-31" {
		t.Fatalf("bounded tags = %#v", got.Tags)
	}
	if len(got.Metadata) != maxTraceMetadata || got.Metadata["key00"] != "value-00" || got.Metadata["key31"] != "value-31" {
		t.Fatalf("bounded metadata = %#v", got.Metadata)
	}
	if _, exists := got.Metadata["key32"]; exists {
		t.Fatalf("metadata limit was not deterministic: %#v", got.Metadata)
	}
}

func TestTraceContextAttributesAreDeterministic(t *testing.T) {
	ctx := WithTraceContext(t.Context(), TraceContext{Metadata: map[string]string{"z": "last", "a": "first"}})
	attributes := traceContextAttributes(ctx)
	if len(attributes) != 2 || string(attributes[0].Key) != "gai.trace.metadata.a" || string(attributes[1].Key) != "gai.trace.metadata.z" {
		t.Fatalf("metadata attributes are not sorted: %#v", attributes)
	}
}

func TestTraceContextIsNotInjectedAsBaggage(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	t.Cleanup(func() { _ = provider.Shutdown(context.Background()) })

	ctx := WithTraceContext(t.Context(), TraceContext{UserID: "user-secret", SessionID: "session-secret"})
	ctx, span := provider.Tracer("trace-context-test").Start(ctx, "request")
	defer span.End()

	carrier := propagation.HeaderCarrier(http.Header{})
	propagation.NewCompositeTextMapPropagator(propagation.TraceContext{}, propagation.Baggage{}).Inject(ctx, carrier)
	if baggage := carrier.Get("baggage"); baggage != "" {
		t.Fatalf("trace context leaked through baggage: %q", baggage)
	}
	for key, values := range http.Header(carrier) {
		joined := strings.Join(values, ",")
		if strings.Contains(joined, "user-secret") || strings.Contains(joined, "session-secret") {
			t.Fatalf("trace context leaked through header %q: %q", key, joined)
		}
	}
}

func TestStartOperationSpanCopiesTraceContextAttributesWithoutCrossContamination(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	previousProvider := otel.GetTracerProvider()
	otel.SetTracerProvider(provider)
	t.Cleanup(func() {
		_ = provider.Shutdown(context.Background())
		otel.SetTracerProvider(previousProvider)
	})

	done := make(chan struct{}, 2)
	for _, userID := range []string{"user-a", "user-b"} {
		go func(userID string) {
			ctx := WithTraceContext(context.Background(), TraceContext{UserID: userID})
			_, span := StartOperationSpan(ctx, "trace-context-test", "test", "test.operation", userID)
			span.End()
			done <- struct{}{}
		}(userID)
	}
	<-done
	<-done

	seen := map[string]string{}
	for _, span := range recorder.Ended() {
		for _, attr := range span.Attributes() {
			if string(attr.Key) == "gai.trace.user_id" {
				seen[span.Name()] = attr.Value.AsString()
			}
		}
	}
	if seen["test.user-a"] != "user-a" || seen["test.user-b"] != "user-b" {
		t.Fatalf("trace contexts crossed runs: %#v", seen)
	}
}
