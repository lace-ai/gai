package anthropic

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
)

func TestGenerateEmitsRequestObservationWithoutContentCapture(t *testing.T) {
	var events []gai.Observation
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"answer"}],"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer server.Close()

	provider := New("test-key", gai.ObservationSinkFunc(func(_ context.Context, event gai.Observation) {
		events = append(events, event)
	}))
	provider.baseURL = server.URL
	model, err := provider.Model(ClaudeSonnet4_6)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Generate(t.Context(), ai.AIRequest{Prompt: "secret prompt"}); err != nil {
		t.Fatal(err)
	}

	for _, event := range events {
		if event.Name == "anthropic_generate_request" {
			if _, ok := event.Fields["prompt"]; ok {
				t.Fatalf("request observation exposed prompt without a capture policy: %#v", event.Fields)
			}
			return
		}
	}
	t.Fatalf("anthropic_generate_request observation not emitted: %#v", events)
}
