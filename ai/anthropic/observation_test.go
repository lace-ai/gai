package anthropic

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestGenerateEmitsRequestObservationWithoutContentCapture(t *testing.T) {
	previousProvider := otel.GetTracerProvider()
	providerTracer := sdktrace.NewTracerProvider()
	otel.SetTracerProvider(providerTracer)
	t.Cleanup(func() {
		_ = providerTracer.Shutdown(context.Background())
		otel.SetTracerProvider(previousProvider)
	})

	var events []gai.Observation
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"answer"}],"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer server.Close()

	provider := New("test-key", gai.ObservationSinkFunc(func(_ context.Context, event gai.Observation) {
		events = append(events, event)
	}))
	provider.baseURL = server.URL
	model, err := provider.Model(ClaudeSonnet4_6)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Generate(t.Context(), ai.AIRequest{Prompt: "secret prompt"}); err != nil {
		t.Fatal(err)
	}

	var request, finished *gai.Observation
	for index := range events {
		event := &events[index]
		if event.Name == "anthropic_generate_request" {
			request = event
			if _, ok := event.Fields["prompt"]; ok {
				t.Fatalf("request observation exposed prompt without a capture policy: %#v", event.Fields)
			}
		}
		if event.Name == "generation_finished" {
			finished = event
		}
	}
	if request == nil || finished == nil {
		t.Fatalf("request or generation_finished observation not emitted: %#v", events)
	}
	if request.TraceID != finished.TraceID || request.SpanID != finished.SpanID {
		t.Fatalf("request correlation = (%q, %q), generation_finished = (%q, %q)", request.TraceID, request.SpanID, finished.TraceID, finished.SpanID)
	}
}

func TestGenerateEmitsRequestAndContentToOTelWithoutSink(t *testing.T) {
	previousProvider := otel.GetTracerProvider()
	recorder := tracetest.NewSpanRecorder()
	providerTracer := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	otel.SetTracerProvider(providerTracer)
	t.Cleanup(func() {
		_ = providerTracer.Shutdown(context.Background())
		otel.SetTracerProvider(previousProvider)
	})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"answer"}],"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer server.Close()

	provider := New("test-key", nil)
	provider.baseURL = server.URL
	model, err := provider.Model(ClaudeSonnet4_6)
	if err != nil {
		t.Fatal(err)
	}
	ctx := gai.WithContentCapturePolicy(t.Context(), gai.ContentCapturePolicy{Prompt: gai.CaptureEnabled})
	if _, err := model.Generate(ctx, ai.AIRequest{Prompt: "allowed prompt"}); err != nil {
		t.Fatal(err)
	}

	for _, span := range recorder.Ended() {
		for _, event := range span.Events() {
			if event.Name != "debug.anthropic_generate_request" {
				continue
			}
			for _, attribute := range event.Attributes {
				if string(attribute.Key) == "debug.prompt" && attribute.Value.AsString() == "allowed prompt" {
					return
				}
			}
			t.Fatalf("request event omitted policy-enabled prompt: %#v", event.Attributes)
		}
	}
	t.Fatalf("OTel-only generation omitted request observation: %#v", recorder.Ended())
}
