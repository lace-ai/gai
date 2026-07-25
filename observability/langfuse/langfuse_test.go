package langfuse

import (
	"context"
	"testing"
)

func TestNewTracerProviderRejectsMissingCredentials(t *testing.T) {
	_, err := NewTracerProvider(context.Background(), Config{Endpoint: "https://langfuse.example/api/public/otel"})
	if err == nil {
		t.Fatal("expected missing credentials error")
	}
}

func TestNewTracerProviderRejectsInvalidEndpoint(t *testing.T) {
	_, err := NewTracerProvider(context.Background(), Config{PublicKey: "pk", SecretKey: "sk", Endpoint: "://invalid"})
	if err == nil {
		t.Fatal("expected invalid endpoint error")
	}
}
