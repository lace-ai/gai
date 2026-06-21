package agent

import (
	"context"
	"errors"
	"sort"

	"github.com/lace-ai/gai"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

const agentTracerName = "github.com/lace-ai/gai/agent"

type runCreationObserver struct {
	debug           gai.DebugSink
	span            trace.Span
	agentName       string
	modelName       string
	toolCount       int
	middlewareCount int
	input           RunInput
}

func newRunCreationObserver(ctx context.Context, agent *Agent, input RunInput) (context.Context, *runCreationObserver) {
	name := agent.name()
	debug := agent.debugSink()
	modelName := ""
	toolCount := 0
	middlewareCount := len(agent.middleware())
	if agent != nil {
		toolCount = len(agent.def.Tools)
		if agent.def.Model != nil {
			modelName = agent.def.Model.Name()
		}
	}
	ctx, span := gai.StartOperationSpan(ctx, agentTracerName, "agent.run", "agent.operation", "create",
		attribute.String("agent.name", name),
		attribute.String("agent.model", modelName),
		attribute.Int("agent.tool_count", toolCount),
		attribute.Int("agent.middleware_count", middlewareCount),
		attribute.Int("agent.input_chars", len(input.Text)),
		attribute.Int("agent.max_tokens", input.MaxTokens),
	)
	return ctx, &runCreationObserver{
		debug:           debug,
		span:            span,
		agentName:       name,
		modelName:       modelName,
		toolCount:       toolCount,
		middlewareCount: middlewareCount,
		input:           input,
	}
}

func (o *runCreationObserver) Created(ctx context.Context) {
	o.emit(ctx, "agent_run_created", o.fields(), nil)
}

func (o *runCreationObserver) Failed(ctx context.Context, stage string, err error) {
	fields := o.fields()
	fields["stage"] = stage
	o.emit(ctx, "agent_run_creation_failed", fields, err)
}

func (o *runCreationObserver) Finish(err error) {
	if o == nil || o.span == nil {
		return
	}
	gai.EndSpan(o.span, err)
}

func (o *runCreationObserver) fields() map[string]any {
	fields := map[string]any{}
	if o == nil {
		return fields
	}
	fields["agent_name"] = o.agentName
	fields["model"] = o.modelName
	fields["tool_count"] = o.toolCount
	fields["middleware_count"] = o.middlewareCount
	fields["input_chars"] = len(o.input.Text)
	fields["max_tokens"] = o.input.MaxTokens
	fields["meta_keys"] = sortedMetaKeys(o.input.Meta)
	if o.input.ID != "" {
		fields["run_id"] = o.input.ID
	}
	if o.debug != nil && o.debug.IncludeSensitiveData() {
		fields["input_text"] = o.input.Text
	}
	return fields
}

func (o *runCreationObserver) emit(ctx context.Context, name string, fields map[string]any, err error) {
	if o == nil || o.debug == nil {
		return
	}
	o.debug.Emit(ctx, gai.DebugEvent{Name: name, Source: "agent:Agent.NewRun", Fields: fields, Err: err})
}

type workflowObserver struct {
	debug           gai.DebugSink
	span            trace.Span
	agentName       string
	middlewareCount int
}

func newWorkflowObserver(ctx context.Context, workflow *Workflow) (context.Context, *workflowObserver) {
	name := ""
	middlewareCount := 0
	var debug gai.DebugSink
	if workflow != nil {
		name = workflow.name
		middlewareCount = len(workflow.middleware)
		debug = workflow.debug
	}
	ctx, span := gai.StartOperationSpan(ctx, agentTracerName, "agent.workflow", "agent.operation", "run",
		attribute.String("agent.name", name),
		attribute.Int("agent.middleware_count", middlewareCount),
	)
	return ctx, &workflowObserver{debug: debug, span: span, agentName: name, middlewareCount: middlewareCount}
}

func (o *workflowObserver) Started(ctx context.Context) {
	o.emit(ctx, "agent_workflow_started", map[string]any{
		"agent_name":       o.agentName,
		"middleware_count": o.middlewareCount,
	}, nil)
}

func (o *workflowObserver) PrimaryFinished(ctx context.Context, result AgentResult) {
	fields := agentResultFields(result)
	fields["agent_name"] = o.agentName
	if o.debug != nil && o.debug.IncludeSensitiveData() {
		fields["output_text"] = result.Text
	}
	o.emit(ctx, "agent_primary_finished", fields, errors.Join(result.Errors...))
}

func (o *workflowObserver) Finished(ctx context.Context, result WorkflowResult) {
	fields := map[string]any{
		"agent_name":       o.agentName,
		"middleware_count": o.middlewareCount,
		"stage_count":      len(result.Stages),
		"token_count":      len(result.Tokens),
		"text_chars":       len(result.Text),
		"error_count":      len(result.Errors),
		"complete":         result.Complete,
	}
	if o != nil && o.span != nil {
		o.span.SetAttributes(
			attribute.Int("agent.stage_count", len(result.Stages)),
			attribute.Int("agent.token_count", len(result.Tokens)),
			attribute.Int("agent.text_chars", len(result.Text)),
			attribute.Int("agent.error_count", len(result.Errors)),
		)
	}
	if o.debug != nil && o.debug.IncludeSensitiveData() {
		fields["output_text"] = result.Text
	}
	err := errors.Join(result.Errors...)
	o.emit(ctx, "agent_workflow_finished", fields, err)
	if o != nil && o.span != nil {
		gai.EndSpan(o.span, err)
	}
}

func (o *workflowObserver) emit(ctx context.Context, name string, fields map[string]any, err error) {
	if o == nil || o.debug == nil {
		return
	}
	o.debug.Emit(ctx, gai.DebugEvent{Name: name, Source: "agent:Workflow.Run", Fields: fields, Err: err})
}

type middlewareObserver struct {
	debug       gai.DebugSink
	span        trace.Span
	agentName   string
	stageName   string
	output      OutputPolicy
	errorPolicy ErrorPolicy
}

func newMiddlewareObserver(ctx context.Context, run *MiddlewareContext, middleware *AgentMiddleware, upstream capturedStream) (context.Context, *middlewareObserver) {
	agentName := ""
	var debug gai.DebugSink
	if run != nil && run.workflow != nil {
		agentName = run.workflow.name
		debug = run.workflow.debug
	}
	stageName := middleware.name()
	ctx, span := gai.StartOperationSpan(ctx, agentTracerName, "agent.middleware", "agent.operation", "run",
		attribute.String("agent.name", agentName),
		attribute.String("agent.middleware.name", stageName),
		attribute.String("agent.output_policy", outputPolicyName(middleware.config.Output)),
		attribute.String("agent.error_policy", errorPolicyName(middleware.config.ErrorPolicy)),
		attribute.Int("agent.upstream_token_count", len(upstream.Tokens)),
		attribute.Int("agent.upstream_error_count", len(upstream.Errors)),
	)
	return ctx, &middlewareObserver{
		debug:       debug,
		span:        span,
		agentName:   agentName,
		stageName:   stageName,
		output:      middleware.config.Output,
		errorPolicy: middleware.config.ErrorPolicy,
	}
}

func (o *middlewareObserver) Started(ctx context.Context) {
	o.emit(ctx, "agent_middleware_started", o.fields(), nil)
}

func (o *middlewareObserver) Skipped(ctx context.Context, reason string) {
	fields := o.fields()
	fields["reason"] = reason
	o.emit(ctx, "agent_middleware_skipped", fields, nil)
	if o != nil && o.span != nil {
		o.span.SetAttributes(attribute.Bool("agent.middleware.skipped", true), attribute.String("agent.middleware.skip_reason", reason))
		gai.EndSpan(o.span, nil)
	}
}

func (o *middlewareObserver) Finished(ctx context.Context, result AgentResult, applied bool) {
	fields := o.fields()
	for key, value := range agentResultFields(result) {
		fields[key] = value
	}
	fields["output_applied"] = applied
	if o != nil && o.span != nil {
		o.span.SetAttributes(
			attribute.Int("agent.middleware.token_count", len(result.Tokens)),
			attribute.Int("agent.middleware.error_count", len(result.Errors)),
			attribute.Bool("agent.middleware.output_applied", applied),
		)
	}
	if o.debug != nil && o.debug.IncludeSensitiveData() {
		fields["output_text"] = result.Text
	}
	err := errors.Join(result.Errors...)
	name := "agent_middleware_finished"
	if err != nil {
		name = "agent_middleware_failed"
	}
	o.emit(ctx, name, fields, err)
	if o != nil && o.span != nil {
		gai.EndSpan(o.span, err)
	}
}

func (o *middlewareObserver) fields() map[string]any {
	if o == nil {
		return map[string]any{}
	}
	return map[string]any{
		"agent_name":    o.agentName,
		"middleware":    o.stageName,
		"output_policy": outputPolicyName(o.output),
		"error_policy":  errorPolicyName(o.errorPolicy),
	}
}

func (o *middlewareObserver) emit(ctx context.Context, name string, fields map[string]any, err error) {
	if o == nil || o.debug == nil {
		return
	}
	o.debug.Emit(ctx, gai.DebugEvent{Name: name, Source: "agent:AgentMiddleware.Process", Fields: fields, Err: err})
}

func agentResultFields(result AgentResult) map[string]any {
	return map[string]any{
		"token_count":     len(result.Tokens),
		"text_chars":      len(result.Text),
		"message_count":   len(result.Messages),
		"iteration_count": len(result.Iterations),
		"error_count":     len(result.Errors),
	}
}

func sortedMetaKeys(meta map[string]any) []string {
	keys := make([]string, 0, len(meta))
	for key := range meta {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func outputPolicyName(policy OutputPolicy) string {
	switch policy {
	case AppendOutput:
		return "append"
	case ReplaceOutput:
		return "replace"
	default:
		return "preserve"
	}
}

func errorPolicyName(policy ErrorPolicy) string {
	if policy == RecordError {
		return "record"
	}
	return "propagate"
}
