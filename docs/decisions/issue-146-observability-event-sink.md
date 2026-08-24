# Issue 146: observation sink migration

This document records the implementation boundary for the repository-wide migration from `DebugSink`.

## Proposed public contract

```go
type Observation struct {
    Name       string
    Source     string
    OccurredAt time.Time
    RunID      string
    TraceID    string
    SpanID     string
    Fields     map[string]any
    Err        error
}

type ObservationSink interface {
    Emit(context.Context, Observation)
}

type ObservationSinkFunc func(context.Context, Observation)
```

`Observation` is a best-effort projection for tracing, logging, and analytics. It is deliberately distinct from the ordered workflow execution stream exposed by `agent.Workflow.RunEvents`.

## Shared plumbing

The repository-private `internal/observe.Operation` helper should own final event enrichment and one fan-out from each already-normalized domain occurrence:

1. domain observers construct a name, source, domain fields, and error;
2. the helper stamps `OccurredAt`, copies fields, derives active OTel trace/span IDs, and supplies run correlation;
3. the helper emits the completed `Observation` to `ObservationSink` and records the corresponding OTel span/event/status.

Sink implementations only receive finalized observations. `ObservationSink.Emit` is called synchronously, so implementations must return promptly and should enqueue internally if persistence is slow. GAI does not retry or wait for external persistence.

## Migration scope

The current `DebugSink` surface has spread across agent, context, providers, model repositories, tools, and the private observer helper. The migration should be repository-wide rather than exposing both public contracts long-term:

- replace `DebugEvent`/`DebugSink`/`DebugSinkFunc` with `Observation`/`ObservationSink`/`ObservationSinkFunc`;
- remove `SensitiveDebugSinkFunc` and `IncludeSensitiveData`;
- make `ContentCapturePolicy` the sole authority for prompt, completion, reasoning, tool input/output, memory, truncation, and redaction capture;
- snapshot the configured sink into every workflow/run as the existing debug sink is today;
- migrate generation outcomes/usage, retry scheduling, tool outcomes, context budget decisions, and history summarization vertically with recording-sink and tracer tests.

## Decisions for review

1. **Run correlation:** use `RunInput.ID` when provided; otherwise `Agent.NewRun` generates an opaque run ID and includes it on every observation. This keeps sinks useful without an external OTel exporter and avoids overloading the optional trace context as a run identity. The ID is correlation metadata only, not a durable execution/replay protocol.
2. **Public migration:** because GAI is pre-v1, remove the legacy debug types rather than retain aliases with divergent content-capture behavior. This produces one sink contract and makes `ContentCapturePolicy` authoritative.
3. **Emission semantics:** invoke `ObservationSink.Emit` synchronously with a finalized observation. Implementations own queuing, persistence, retries, and panic boundaries; GAI owns neither backend delivery nor a second execution stream.

## Initial implementation sequence

1. Add the public contract, enrichment, run-ID propagation, and a recording-sink regression test.
2. Adapt `internal/observe.Operation` so its one semantic occurrence fans out to the finalized event and OTel projection.
3. Migrate generation, retry, tool, context, and history observer slices; preserve policy-gated content fields in each test.
4. Replace remaining direct debug surfaces (providers, renderers, repositories, and tools), delete the legacy sensitive path, and run the repository test and static-check suites.
