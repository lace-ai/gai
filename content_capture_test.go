package gai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestCaptureContentDefaultsToDisabled(t *testing.T) {
	if _, ok := CaptureContent(t.Context(), ContentKindPrompt, []byte("secret")); ok {
		t.Fatal("content captured without an installed policy")
	}
	ctx := WithContentCapturePolicy(t.Context(), ContentCapturePolicy{})
	if _, ok := CaptureContent(ctx, ContentKindPrompt, []byte("secret")); ok {
		t.Fatal("content captured with a zero-value policy")
	}
}

func TestCaptureContentCategoriesAreIndependent(t *testing.T) {
	tests := []struct {
		name   string
		kind   ContentKind
		policy ContentCapturePolicy
	}{
		{"prompt", ContentKindPrompt, ContentCapturePolicy{Prompt: CaptureEnabled}},
		{"completion", ContentKindCompletion, ContentCapturePolicy{Completion: CaptureEnabled}},
		{"reasoning", ContentKindReasoning, ContentCapturePolicy{Reasoning: CaptureEnabled}},
		{"tool input", ContentKindToolInput, ContentCapturePolicy{ToolInput: CaptureEnabled}},
		{"tool output", ContentKindToolOutput, ContentCapturePolicy{ToolOutput: CaptureEnabled}},
		{"memory", ContentKindMemory, ContentCapturePolicy{Memory: CaptureEnabled}},
	}
	allKinds := []ContentKind{ContentKindPrompt, ContentKindCompletion, ContentKindReasoning, ContentKindToolInput, ContentKindToolOutput, ContentKindMemory}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := WithContentCapturePolicy(t.Context(), tt.policy)
			for _, kind := range allKinds {
				captured, ok := CaptureContent(ctx, kind, []byte("value"))
				if kind == tt.kind {
					if !ok || string(captured.Value) != "value" {
						t.Fatalf("enabled kind %q was not captured: %#v, %v", kind, captured, ok)
					}
				} else if ok {
					t.Fatalf("disabled kind %q was captured", kind)
				}
			}
		})
	}
}

func TestCaptureContentRedactsBeforeTruncating(t *testing.T) {
	ctx := WithContentCapturePolicy(t.Context(), ContentCapturePolicy{
		Prompt:   CaptureEnabled,
		MaxBytes: 8,
		Redact: func(_ context.Context, kind ContentKind, value []byte) ([]byte, error) {
			if kind != ContentKindPrompt {
				t.Fatalf("redactor kind = %q", kind)
			}
			return bytes.ReplaceAll(value, []byte("secret"), []byte("public")), nil
		},
	})
	captured, ok := CaptureContent(ctx, ContentKindPrompt, []byte("secret-abcdef"))
	if !ok {
		t.Fatal("content not captured")
	}
	if got := string(captured.Value); got != "public-a" {
		t.Fatalf("captured value = %q, want redaction before truncation", got)
	}
	if captured.OriginalBytes != len("secret-abcdef") || captured.CapturedBytes != 8 || !captured.Truncated || !captured.RedactionApplied {
		t.Fatalf("unexpected capture metadata: %#v", captured)
	}
}

func TestCaptureContentReportsSuccessfulNoOpRedaction(t *testing.T) {
	ctx := WithContentCapturePolicy(t.Context(), ContentCapturePolicy{
		Prompt: CaptureEnabled,
		Redact: func(_ context.Context, _ ContentKind, value []byte) ([]byte, error) {
			return value, nil
		},
	})
	captured, ok := CaptureContent(ctx, ContentKindPrompt, []byte("unchanged"))
	if !ok || !captured.RedactionApplied {
		t.Fatalf("successful redactor invocation was not reported: %#v, %v", captured, ok)
	}
}

func TestCaptureContentRedactorFailuresFailClosed(t *testing.T) {
	tests := []struct {
		name   string
		redact ContentRedactor
	}{
		{"error", func(context.Context, ContentKind, []byte) ([]byte, error) { return nil, errors.New("redaction failed") }},
		{"panic", func(context.Context, ContentKind, []byte) ([]byte, error) { panic("redaction failed") }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := WithContentCapturePolicy(t.Context(), ContentCapturePolicy{Prompt: CaptureEnabled, Redact: tt.redact})
			if _, ok := CaptureContent(ctx, ContentKindPrompt, []byte("sentinel-secret")); ok {
				t.Fatal("content captured after redactor failure")
			}
		})
	}
}

func TestCaptureContentCopiesBuffersAndBoundsOutput(t *testing.T) {
	input := []byte("ééé-secret")
	ctx := WithContentCapturePolicy(t.Context(), ContentCapturePolicy{
		Prompt:   CaptureEnabled,
		MaxBytes: 5,
		Redact: func(_ context.Context, _ ContentKind, value []byte) ([]byte, error) {
			value[0] = 'x'
			return value, nil
		},
	})
	captured, ok := CaptureContent(ctx, ContentKindPrompt, input)
	if !ok {
		t.Fatal("content not captured")
	}
	if string(input) != "ééé-secret" {
		t.Fatalf("redactor mutated caller input: %q", input)
	}
	if len(captured.Value) > 5 || !json.Valid([]byte(fmt.Sprintf("%q", string(captured.Value)))) {
		t.Fatalf("capture is not bounded valid UTF-8: %q", captured.Value)
	}
	before := append([]byte(nil), captured.Value...)
	input[0] = 'z'
	if !bytes.Equal(before, captured.Value) {
		t.Fatal("captured output aliases caller input")
	}
}

func TestCaptureContentKeepsOversizedJSONValidAndDeterministic(t *testing.T) {
	ctx := WithContentCapturePolicy(t.Context(), ContentCapturePolicy{ToolInput: CaptureEnabled, MaxBytes: 24})
	value := []byte(`{"query":"a very long secret value"}`)
	first, ok := CaptureContent(ctx, ContentKindToolInput, value)
	if !ok {
		t.Fatal("content not captured")
	}
	second, ok := CaptureContent(ctx, ContentKindToolInput, value)
	if !ok {
		t.Fatal("content not captured on second call")
	}
	if len(first.Value) > 24 || !json.Valid(first.Value) {
		t.Fatalf("truncated JSON = %q", first.Value)
	}
	if !bytes.Equal(first.Value, second.Value) || !first.Truncated {
		t.Fatalf("truncation is not deterministic: %q != %q", first.Value, second.Value)
	}
}

type capturePolicyTestSink struct {
	event Observation
}

type panickingJSONMarshaler struct{}

func (panickingJSONMarshaler) MarshalJSON() ([]byte, error) { panic("marshal panic") }

type countingJSONMarshaler struct{ calls *int }

func (m countingJSONMarshaler) MarshalJSON() ([]byte, error) {
	(*m.calls)++
	return []byte(`{"value":"serialized"}`), nil
}

func TestAddObservationContentSerializationPanicFailsClosed(t *testing.T) {
	ctx := WithContentCapturePolicy(t.Context(), ContentCapturePolicy{Prompt: CaptureEnabled})
	fields := map[string]any{}
	AddObservationContent(ctx, &capturePolicyTestSink{}, fields, "prompt", ContentKindPrompt, panickingJSONMarshaler{})
	if len(fields) != 0 {
		t.Fatalf("panicking serializer emitted fields: %#v", fields)
	}
}

func TestAddObservationContentDisabledCategorySkipsSerialization(t *testing.T) {
	calls := 0
	ctx := WithContentCapturePolicy(t.Context(), ContentCapturePolicy{Completion: CaptureEnabled})
	fields := map[string]any{}
	AddObservationContent(ctx, &capturePolicyTestSink{}, fields, "prompt", ContentKindPrompt, countingJSONMarshaler{calls: &calls})
	if calls != 0 || len(fields) != 0 {
		t.Fatalf("disabled category serialized content: calls=%d fields=%#v", calls, fields)
	}
}

func (s *capturePolicyTestSink) Emit(_ context.Context, event Observation) { s.event = event }

func TestAddObservationContentRequiresAnInstalledPolicy(t *testing.T) {
	sink := &capturePolicyTestSink{}
	fields := map[string]any{}
	AddObservationContent(t.Context(), sink, fields, "payload", ContentKindPrompt, "must-not-leak")
	if len(fields) != 0 {
		t.Fatalf("content captured without a policy: %#v", fields)
	}

	policyCtx := WithContentCapturePolicy(t.Context(), ContentCapturePolicy{Completion: CaptureEnabled, MaxBytes: 64})
	AddObservationContent(policyCtx, sink, fields, "prompt", ContentKindPrompt, "must-not-leak")
	AddObservationContent(policyCtx, sink, fields, "completion", ContentKindCompletion, "allowed")
	if _, ok := fields["prompt"]; ok {
		t.Fatal("disabled policy category captured content")
	}
	if fields["completion"] != "allowed" || fields["completion_content_kind"] != "completion" {
		t.Fatalf("enabled policy content missing: %#v", fields)
	}
}

func TestAddObservationContentCapturesForActiveSpanWithoutSink(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	t.Cleanup(func() { _ = provider.Shutdown(context.Background()) })

	ctx := WithContentCapturePolicy(t.Context(), ContentCapturePolicy{Prompt: CaptureEnabled})
	ctx, span := provider.Tracer("content-capture-test").Start(ctx, "capture")
	fields := map[string]any{}
	AddObservationContent(ctx, nil, fields, "prompt", ContentKindPrompt, "allowed-content")
	EmitObservation(ctx, nil, Observation{Name: "capture", Source: "test", Fields: fields})
	span.End()

	if fields["prompt"] != "allowed-content" {
		t.Fatalf("policy-enabled content missing: %#v", fields)
	}
	if len(recorder.Ended()) != 1 || !containsAttribute(recorder.Ended()[0].Events()[0].Attributes, "debug.prompt", "allowed-content") {
		t.Fatalf("policy-enabled content missing from OTel event: %#v", recorder.Ended())
	}
}

func TestRedactedOrDisabledContentNeverReachesOTelEvent(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	t.Cleanup(func() { _ = provider.Shutdown(context.Background()) })

	ctx := WithContentCapturePolicy(t.Context(), ContentCapturePolicy{
		Completion: CaptureEnabled,
		Redact: func(_ context.Context, _ ContentKind, value []byte) ([]byte, error) {
			return bytes.ReplaceAll(value, []byte("sentinel-secret"), []byte("[REDACTED]")), nil
		},
	})
	ctx, span := provider.Tracer("content-capture-test").Start(ctx, "capture")
	sink := ObservationSinkFunc(func(context.Context, Observation) {})
	fields := map[string]any{}
	AddObservationContent(ctx, sink, fields, "prompt", ContentKindPrompt, "disabled-sentinel-secret")
	AddObservationContent(ctx, sink, fields, "completion", ContentKindCompletion, "enabled-sentinel-secret")
	EmitObservation(ctx, sink, Observation{Name: "capture", Source: "test", Fields: fields})
	span.End()

	var exported strings.Builder
	for _, ended := range recorder.Ended() {
		for _, event := range ended.Events() {
			for _, attr := range event.Attributes {
				exported.WriteString(string(attr.Key))
				exported.WriteString(attr.Value.String())
			}
		}
	}
	if strings.Contains(exported.String(), "sentinel-secret") || strings.Contains(exported.String(), "disabled-") {
		t.Fatalf("raw or disabled content reached OTel: %s", exported.String())
	}
	if !strings.Contains(exported.String(), "[REDACTED]") {
		t.Fatalf("redacted value missing from OTel: %s", exported.String())
	}
	if !containsAttribute(recorder.Ended()[0].Events()[0].Attributes, "debug.completion_redaction_applied", "true") {
		t.Fatal("redaction metadata missing from OTel")
	}
}

func containsAttribute(attrs []attribute.KeyValue, key, value string) bool {
	for _, attr := range attrs {
		if string(attr.Key) == key && attr.Value.String() == value {
			return true
		}
	}
	return false
}

func TestEmitObservationDoesNotExportRawErrorsWithoutContentPolicy(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	t.Cleanup(func() { _ = provider.Shutdown(context.Background()) })

	ctx, span := provider.Tracer("content-capture-test").Start(t.Context(), "capture")
	sink := &capturePolicyTestSink{}
	EmitObservation(ctx, sink, Observation{
		Name:   "failed",
		Source: "test",
		Fields: map[string]any{"retained": "yes", "error": "caller-supplied"},
		Err:    errors.New("sentinel-error-content"),
	})
	span.End()

	if sink.event.Err != nil || strings.Contains(fmt.Sprintf("%#v", sink.event), "sentinel-error-content") {
		t.Fatalf("raw error reached observation sink: %#v", sink.event)
	}
	if sink.event.Fields["outcome"] != "error" || sink.event.Fields["error_type"] != "*errors.errorString" || sink.event.Fields["retained"] != "yes" {
		t.Fatalf("safe error metadata = %#v", sink.event.Fields)
	}
	if _, ok := sink.event.Fields["error"]; ok {
		t.Fatalf("caller-supplied error field reached observation sink: %#v", sink.event.Fields)
	}

	var exported strings.Builder
	for _, event := range recorder.Ended()[0].Events() {
		for _, attr := range event.Attributes {
			exported.WriteString(string(attr.Key))
			exported.WriteString(attr.Value.String())
		}
	}
	if strings.Contains(exported.String(), "sentinel-error-content") {
		t.Fatalf("raw error reached OTel observation: %s", exported.String())
	}
}
