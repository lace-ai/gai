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
	return startOperationSpan(ctx, tracerName, spanPrefix+"."+operation, operationAttr, operation, nil, attrs...)
}

// StartClientOperationSpan starts an operation span representing a call to a
// remote system while preserving GAI trace-context attributes.
func StartClientOperationSpan(ctx context.Context, tracerName string, spanName string, operationAttr string, operation string, attrs ...attribute.KeyValue) (context.Context, trace.Span) {
	return startOperationSpan(ctx, tracerName, spanName, operationAttr, operation, []trace.SpanStartOption{trace.WithSpanKind(trace.SpanKindClient)}, attrs...)
}

func startOperationSpan(ctx context.Context, tracerName string, spanName string, operationAttr string, operation string, options []trace.SpanStartOption, attrs ...attribute.KeyValue) (context.Context, trace.Span) {
	baseAttrs := []attribute.KeyValue{
		attribute.String(operationAttr, operation),
	}
	baseAttrs = append(baseAttrs, attrs...)
	baseAttrs = append(baseAttrs, traceContextAttributes(ctx)...)

	options = append(options, trace.WithAttributes(baseAttrs...))
	return otel.Tracer(tracerName).Start(ctx, spanName, options...)
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

func RecordObservation(ctx context.Context, e Observation) {
	span := trace.SpanFromContext(ctx)
	if !span.SpanContext().IsValid() {
		return
	}
	span.AddEvent("debug."+e.Name, trace.WithAttributes(debugEventAttributes(e)...))
}

func debugEventAttributes(e Observation) []attribute.KeyValue {
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
