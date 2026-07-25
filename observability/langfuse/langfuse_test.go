package langfuse

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
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

func TestNewTracerProviderSendsLangfuseIngestionVersion(t *testing.T) {
	headers := make(chan http.Header, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		headers <- r.Header
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	provider, err := NewTracerProvider(context.Background(), Config{
		Endpoint:  server.URL,
		PublicKey: "pk",
		SecretKey: "sk",
	})
	if err != nil {
		t.Fatalf("NewTracerProvider() error = %v", err)
	}
	_, span := provider.Tracer("test").Start(context.Background(), "test-span")
	span.End()
	if err := provider.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown() error = %v", err)
	}

	select {
	case got := <-headers:
		if got.Get("x-langfuse-ingestion-version") != "4" {
			t.Fatalf("x-langfuse-ingestion-version = %q, want %q", got.Get("x-langfuse-ingestion-version"), "4")
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for exported span")
	}
}
