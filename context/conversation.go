package context

// Conversation exposes the ordered messages rendered into an iterative prompt.
type Conversation interface {
	Messages() []Message
}
