package gai

import (
	"context"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"go.opentelemetry.io/otel/attribute"
)

const (
	maxTraceScalarBytes   = 200
	maxTraceTagBytes      = 200
	maxTraceTags          = 32
	maxTraceMetadata      = 32
	maxTraceMetadataKey   = 64
	maxTraceMetadataValue = 200
	maxTraceEnvironment   = 40
)

// TraceContext contains explicitly approved, trace-wide dimensions. It is
// propagated through Go contexts and copied onto GAI spans, but is never added
// to OpenTelemetry baggage or outbound request headers by GAI.
type TraceContext struct {
	Name        string
	UserID      string
	SessionID   string
	Tags        []string
	Release     string
	Environment string
	Metadata    map[string]string
}

type traceContextKey struct{}

// WithTraceContext returns a child context containing a normalized defensive
// copy of traceContext. Invalid, empty, or oversized fields are omitted.
func WithTraceContext(ctx context.Context, traceContext TraceContext) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, traceContextKey{}, normalizeTraceContext(traceContext))
}

// TraceContextFromContext returns a defensive copy of the trace context in ctx.
func TraceContextFromContext(ctx context.Context) (TraceContext, bool) {
	traceContext, ok := traceContextFromContext(ctx)
	if !ok {
		return TraceContext{}, false
	}
	return cloneTraceContext(traceContext), true
}

func traceContextFromContext(ctx context.Context) (TraceContext, bool) {
	if ctx == nil {
		return TraceContext{}, false
	}
	traceContext, ok := ctx.Value(traceContextKey{}).(TraceContext)
	return traceContext, ok
}

func normalizeTraceContext(traceContext TraceContext) TraceContext {
	normalized := TraceContext{
		Name:        normalizeTraceScalar(traceContext.Name, maxTraceScalarBytes),
		UserID:      normalizeTraceScalar(traceContext.UserID, maxTraceScalarBytes),
		SessionID:   normalizeTraceScalar(traceContext.SessionID, maxTraceScalarBytes),
		Release:     normalizeTraceScalar(traceContext.Release, maxTraceScalarBytes),
		Environment: normalizeTraceEnvironment(traceContext.Environment),
	}
	seenTags := make(map[string]struct{}, min(len(traceContext.Tags), maxTraceTags))
	for _, tag := range traceContext.Tags {
		tag = normalizeTraceScalar(tag, maxTraceTagBytes)
		if tag == "" {
			continue
		}
		if _, exists := seenTags[tag]; exists {
			continue
		}
		if len(normalized.Tags) == maxTraceTags {
			break
		}
		seenTags[tag] = struct{}{}
		normalized.Tags = append(normalized.Tags, tag)
	}

	keys := make([]string, 0, len(traceContext.Metadata))
	for key := range traceContext.Metadata {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		if len(normalized.Metadata) == maxTraceMetadata {
			break
		}
		if !validTraceMetadataKey(key) {
			continue
		}
		value := normalizeTraceScalar(traceContext.Metadata[key], maxTraceMetadataValue)
		if value == "" {
			continue
		}
		if normalized.Metadata == nil {
			normalized.Metadata = make(map[string]string)
		}
		normalized.Metadata[key] = value
	}
	return normalized
}

func normalizeTraceScalar(value string, maxBytes int) string {
	value = strings.TrimSpace(value)
	if value == "" || len(value) > maxBytes || !utf8.ValidString(value) {
		return ""
	}
	for _, r := range value {
		if unicode.IsControl(r) {
			return ""
		}
	}
	return value
}

func normalizeTraceEnvironment(value string) string {
	value = normalizeTraceScalar(value, maxTraceEnvironment)
	if value == "" || strings.HasPrefix(value, "langfuse") {
		return ""
	}
	for _, r := range value {
		if (r < 'a' || r > 'z') && (r < '0' || r > '9') && r != '-' && r != '_' {
			return ""
		}
	}
	return value
}

func validTraceMetadataKey(key string) bool {
	if key == "" || len(key) > maxTraceMetadataKey || !utf8.ValidString(key) {
		return false
	}
	for _, r := range key {
		if (r < 'a' || r > 'z') && (r < 'A' || r > 'Z') && (r < '0' || r > '9') {
			return false
		}
	}
	return true
}

func cloneTraceContext(traceContext TraceContext) TraceContext {
	cloned := traceContext
	cloned.Tags = append([]string(nil), traceContext.Tags...)
	if traceContext.Metadata != nil {
		cloned.Metadata = make(map[string]string, len(traceContext.Metadata))
		for key, value := range traceContext.Metadata {
			cloned.Metadata[key] = value
		}
	}
	return cloned
}

func traceContextAttributes(ctx context.Context) []attribute.KeyValue {
	traceContext, ok := traceContextFromContext(ctx)
	if !ok {
		return nil
	}
	attributes := make([]attribute.KeyValue, 0, 6+len(traceContext.Metadata))
	attributes = appendTraceStringAttribute(attributes, "gai.trace.name", traceContext.Name)
	attributes = appendTraceStringAttribute(attributes, "gai.trace.user_id", traceContext.UserID)
	attributes = appendTraceStringAttribute(attributes, "gai.trace.session_id", traceContext.SessionID)
	if len(traceContext.Tags) > 0 {
		attributes = append(attributes, attribute.StringSlice("gai.trace.tags", traceContext.Tags))
	}
	attributes = appendTraceStringAttribute(attributes, "gai.trace.release", traceContext.Release)
	attributes = appendTraceStringAttribute(attributes, "gai.trace.environment", traceContext.Environment)
	keys := make([]string, 0, len(traceContext.Metadata))
	for key := range traceContext.Metadata {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		attributes = append(attributes, attribute.String("gai.trace.metadata."+key, traceContext.Metadata[key]))
	}
	return attributes
}

func appendTraceStringAttribute(attributes []attribute.KeyValue, key, value string) []attribute.KeyValue {
	if value == "" {
		return attributes
	}
	return append(attributes, attribute.String(key, value))
}
