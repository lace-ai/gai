package gai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

var errSpanContextNotFound = errors.New("span context not found in context")

func StartOperationSpan(ctx context.Context, tracerName string, spanPrefix string, operationAttr string, operation string, attrs ...attribute.KeyValue) (context.Context, trace.Span) {
	baseAttrs := []attribute.KeyValue{
		attribute.String(operationAttr, operation),
	}
	baseAttrs = append(baseAttrs, attrs...)

	return otel.Tracer(tracerName).Start(ctx, spanPrefix+"."+operation, trace.WithAttributes(baseAttrs...))
}

func EndSpan(span trace.Span, err error) {
	if err != nil {
		RecordSpanError(span, err)
	}
	span.End()
}

func RecordSpanError(span trace.Span, err error) {
	if err == nil {
		return
	}
	span.RecordError(err)
	span.SetStatus(codes.Error, err.Error())
}

func MarkSpanFailure(span trace.Span, reasonAttr string, reason string, err error, attrs ...attribute.KeyValue) {
	attrs = append(attrs, attribute.String(reasonAttr, reason))
	span.SetAttributes(attrs...)
	RecordSpanError(span, err)
}

func SpanContextIDs(ctx context.Context) (traceID string, spanID string, err error) {
	sc := trace.SpanContextFromContext(ctx)
	if !sc.IsValid() {
		return "", "", errSpanContextNotFound
	}

	return sc.TraceID().String(), sc.SpanID().String(), nil
}

func RecordDebugEvent(ctx context.Context, e DebugEvent) {
	span := trace.SpanFromContext(ctx)
	if !span.SpanContext().IsValid() {
		return
	}
	if e.Err != nil {
		RecordSpanError(span, e.Err)
	}
	span.AddEvent("debug."+e.Name, trace.WithAttributes(debugEventAttributes(e)...))
}

func debugEventAttributes(e DebugEvent) []attribute.KeyValue {
	attrs := []attribute.KeyValue{
		attribute.String("debug.name", e.Name),
		attribute.String("debug.source", e.Source),
	}
	if e.Err != nil {
		attrs = append(attrs, attribute.String("error", e.Err.Error()))
	}
	for key, value := range e.Fields {
		attrs = append(attrs, debugFieldAttribute("debug."+key, value))
	}
	return attrs
}

func debugFieldAttribute(key string, value any) attribute.KeyValue {
	switch v := value.(type) {
	case nil:
		return attribute.String(key, "")
	case string:
		return attribute.String(key, v)
	case bool:
		return attribute.Bool(key, v)
	case int:
		return attribute.Int(key, v)
	case int64:
		return attribute.Int64(key, v)
	case float64:
		return attribute.Float64(key, v)
	default:
		if raw, err := json.Marshal(v); err == nil {
			return attribute.String(key, string(raw))
		}
		return attribute.String(key, fmt.Sprint(v))
	}
}
