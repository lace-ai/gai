package ai

import "fmt"

// FeatureSupport describes whether a model supports a feature. The zero value
// is Unknown, so descriptors can safely omit information they do not know.
type FeatureSupport uint8

const (
	FeatureSupportUnknown FeatureSupport = iota
	FeatureSupportSupported
	FeatureSupportUnsupported
)

// TokenizerFidelity describes how closely tokenizer results match provider
// billing and generation tokenization.
type TokenizerFidelity uint8

const (
	TokenizerFidelityUnknown TokenizerFidelity = iota
	TokenizerFidelityExact
	TokenizerFidelityEstimated
)

// TokenizerDescriptor describes a model tokenizer without constructing it.
type TokenizerDescriptor struct {
	Available FeatureSupport
	Fidelity  TokenizerFidelity
}

// ModelDescriptor describes capabilities known for a model. Unknown features
// are deliberately not rejected during local preflight.
type ModelDescriptor struct {
	// Provider is the provider's stable name when the descriptor is obtained
	// through ModelRepository. Direct model descriptors may leave it empty.
	Provider string
	Model    string
	// NativeMessages reports support for AIRequest.Messages.
	NativeMessages FeatureSupport
	// NativeTools reports support for AIRequest.Tools and tool-call responses.
	NativeTools FeatureSupport
	// ToolChoiceModes lists the tool-choice modes supported by the adapter.
	// An empty list means the adapter does not know the supported modes.
	ToolChoiceModes []ToolChoiceMode
	// Multimodal reports support for non-text request or response content.
	Multimodal FeatureSupport
	// Usage and FinishReason report whether those AIResponse metadata fields
	// are populated when the provider exposes them.
	Usage        FeatureSupport
	FinishReason FeatureSupport
	// StreamingUsage reports whether usage is emitted through Token.TokenUsage.
	StreamingUsage   FeatureSupport
	ToolCalling      FeatureSupport
	JSONOutput       FeatureSupport
	JSONSchemaOutput FeatureSupport
	Reasoning        FeatureSupport
	// ReasoningEfforts enumerates supported values. An empty list means the
	// supported values are not known. ReasoningEffort remains the compatibility
	// summary for callers that only need a yes/no/unknown answer.
	ReasoningEfforts []ReasoningEffort
	ReasoningEffort  FeatureSupport
	Tokenizer        TokenizerDescriptor
}

// Copy returns an independent copy of d. It is provided so callers need not
// rely on the descriptor's current value-only representation.
func (d ModelDescriptor) Copy() ModelDescriptor {
	d.ToolChoiceModes = append([]ToolChoiceMode(nil), d.ToolChoiceModes...)
	d.ReasoningEfforts = append([]ReasoningEffort(nil), d.ReasoningEfforts...)
	return d
}

// ModelDescriber is an optional extension to Model. Model intentionally does
// not include Descriptor so existing custom Model implementations keep source
// compatibility.
type ModelDescriber interface {
	Descriptor() ModelDescriptor
}

// UnsupportedCapabilityError reports a request feature known to be unsupported
// by a model before a provider request is made.
type UnsupportedCapabilityError struct {
	Model      string
	Capability string
}

func (e *UnsupportedCapabilityError) Error() string {
	if e.Model == "" {
		return fmt.Sprintf("%v: %s", ErrUnsupportedCapability, e.Capability)
	}
	return fmt.Sprintf("%v: model %q does not support %s", ErrUnsupportedCapability, e.Model, e.Capability)
}

func (e *UnsupportedCapabilityError) Unwrap() error { return ErrUnsupportedCapability }

// ValidateRequest validates req and rejects only features explicitly marked
// unsupported by d.
func (d ModelDescriptor) ValidateRequest(req AIRequest) error {
	if err := req.Validate(); err != nil {
		return err
	}
	if len(req.Messages) > 0 && d.NativeMessages == FeatureSupportUnsupported {
		return d.unsupported("native messages")
	}
	if len(req.Tools) > 0 && (d.ToolCalling == FeatureSupportUnsupported || d.NativeTools == FeatureSupportUnsupported) {
		return d.unsupported("tool calling")
	}
	if len(req.Tools) > 0 && req.ToolChoice.Mode != "" && len(d.ToolChoiceModes) > 0 && !containsToolChoiceMode(d.ToolChoiceModes, req.ToolChoice.Mode) {
		return d.unsupported("tool choice mode " + string(req.ToolChoice.Mode))
	}
	switch req.ResponseFormat.Type {
	case ResponseFormatJSONObject:
		if d.JSONOutput == FeatureSupportUnsupported {
			return d.unsupported("JSON output")
		}
	case ResponseFormatJSONSchema:
		if d.JSONSchemaOutput == FeatureSupportUnsupported {
			return d.unsupported("JSON schema output")
		}
	}
	if req.Reasoning.Enabled || req.Reasoning.IncludeThoughts || req.Reasoning.BudgetTokens > 0 {
		if d.Reasoning == FeatureSupportUnsupported {
			return d.unsupported("reasoning")
		}
	}
	if req.Reasoning.Effort != "" && d.ReasoningEffort == FeatureSupportUnsupported {
		return d.unsupported("reasoning effort")
	}
	if req.Reasoning.Effort != "" && len(d.ReasoningEfforts) > 0 && !containsReasoningEffort(d.ReasoningEfforts, req.Reasoning.Effort) {
		return d.unsupported("reasoning effort " + string(req.Reasoning.Effort))
	}
	return nil
}

func containsToolChoiceMode(modes []ToolChoiceMode, want ToolChoiceMode) bool {
	for _, mode := range modes {
		if mode == want {
			return true
		}
	}
	return false
}

func containsReasoningEffort(efforts []ReasoningEffort, want ReasoningEffort) bool {
	for _, effort := range efforts {
		if effort == want {
			return true
		}
	}
	return false
}

func (d ModelDescriptor) unsupported(capability string) error {
	return &UnsupportedCapabilityError{Model: d.Model, Capability: capability}
}

// ValidateModelRequest applies common request validation and, when available,
// the optional model descriptor's local preflight checks.
func ValidateModelRequest(model Model, req AIRequest) error {
	if describer, ok := model.(ModelDescriber); ok {
		return describer.Descriptor().ValidateRequest(req)
	}
	return req.Validate()
}
