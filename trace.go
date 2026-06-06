package gai

import (
	"context"
	"errors"

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
