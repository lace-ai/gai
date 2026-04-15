package context

type Conversation interface {
	Messages() []Message
}
