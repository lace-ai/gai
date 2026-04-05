package ai

type Provider interface {
	Name() string
	Model(name string) (Model, error)
	ListModels() ([]string, error)
	Validate() error
}
