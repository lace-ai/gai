package ai

// Provider resolves models made available by an AI service.
type Provider interface {
	// Name returns the stable name used to identify the provider.
	Name() string
	// Model returns the model with the given provider-specific name.
	Model(name string) (Model, error)
	// ListModels returns the provider-specific names of available models.
	ListModels() ([]string, error)
	// Validate checks whether the provider is configured and usable.
	Validate() error
}
