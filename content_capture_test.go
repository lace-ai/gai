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
	sensitive bool
	event     DebugEvent
}

type panickingJSONMarshaler struct{}

func (panickingJSONMarshaler) MarshalJSON() ([]byte, error) { panic("marshal panic") }

type countingJSONMarshaler struct{ calls *int }

func (m countingJSONMarshaler) MarshalJSON() ([]byte, error) {
	(*m.calls)++
	return []byte(`{"value":"serialized"}`), nil
}

func TestAddDebugContentSerializationPanicFailsClosed(t *testing.T) {
	ctx := WithContentCapturePolicy(t.Context(), ContentCapturePolicy{Prompt: CaptureEnabled})
	fields := map[string]any{}
	AddDebugContent(ctx, &capturePolicyTestSink{}, fields, "prompt", ContentKindPrompt, panickingJSONMarshaler{})
	if len(fields) != 0 {
		t.Fatalf("panicking serializer emitted fields: %#v", fields)
	}
}

func TestAddDebugContentDisabledCategorySkipsSerialization(t *testing.T) {
	calls := 0
	ctx := WithContentCapturePolicy(t.Context(), ContentCapturePolicy{Completion: CaptureEnabled})
	fields := map[string]any{}
	AddDebugContent(ctx, &capturePolicyTestSink{}, fields, "prompt", ContentKindPrompt, countingJSONMarshaler{calls: &calls})
	if calls != 0 || len(fields) != 0 {
		t.Fatalf("disabled category serialized content: calls=%d fields=%#v", calls, fields)
	}
}

func (s *capturePolicyTestSink) Emit(_ context.Context, event DebugEvent) { s.event = event }
func (s *capturePolicyTestSink) IncludeSensitiveData() bool               { return s.sensitive }

func TestAddDebugContentPreservesLegacyAndPolicyOverridesSink(t *testing.T) {
	legacy := &capturePolicyTestSink{sensitive: true}
	legacyFields := map[string]any{}
	legacyValue := map[string]string{"secret": "raw"}
	AddDebugContent(t.Context(), legacy, legacyFields, "payload", ContentKindPrompt, legacyValue)
	if got, ok := legacyFields["payload"].(map[string]string); !ok || got["secret"] != "raw" {
		t.Fatalf("legacy field shape changed: %#v", legacyFields["payload"])
	}

	strictCtx := WithContentCapturePolicy(t.Context(), ContentCapturePolicy{Completion: CaptureEnabled, MaxBytes: 64})
	strictFields := map[string]any{}
	AddDebugContent(strictCtx, legacy, strictFields, "prompt", ContentKindPrompt, "must-not-leak")
	AddDebugContent(strictCtx, legacy, strictFields, "completion", ContentKindCompletion, "allowed")
	if _, ok := strictFields["prompt"]; ok {
		t.Fatal("explicit policy did not override sensitive legacy sink")
	}
	if strictFields["completion"] != "allowed" || strictFields["completion_content_kind"] != "completion" {
		t.Fatalf("enabled policy content missing: %#v", strictFields)
	}

	nonSensitive := &capturePolicyTestSink{}
	fields := map[string]any{}
	AddDebugContent(strictCtx, nonSensitive, fields, "completion", ContentKindCompletion, "policy-enabled")
	if fields["completion"] != "policy-enabled" {
		t.Fatalf("policy should enable a field independently of the legacy sink boolean: %#v", fields)
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
	sink := DebugSinkFunc(func(context.Context, DebugEvent) {})
	fields := map[string]any{}
	AddDebugContent(ctx, sink, fields, "prompt", ContentKindPrompt, "disabled-sentinel-secret")
	AddDebugContent(ctx, sink, fields, "completion", ContentKindCompletion, "enabled-sentinel-secret")
	sink.Emit(ctx, DebugEvent{Name: "capture", Source: "test", Fields: fields})
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
