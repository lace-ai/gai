package mistral

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/lace-ai/gai"
	"github.com/lace-ai/gai/ai"
)

func TestRequestObservationsEmitWithoutContentCapture(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		t.Run(map[bool]string{false: "generate", true: "stream"}[streaming], func(t *testing.T) {
			var events []gai.Observation
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				if streaming {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = w.Write([]byte("data: [DONE]\n\n"))
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`))
			}))
			defer server.Close()

			provider := New("test-key", gai.ObservationSinkFunc(func(_ context.Context, event gai.Observation) {
				events = append(events, event)
			}))
			provider.baseURL = server.URL
			model, err := provider.Model(MistralSmallLatest)
			if err != nil {
				t.Fatal(err)
			}
			if streaming {
				for range model.GenerateStream(t.Context(), ai.AIRequest{Prompt: "secret prompt"}) {
				}
			} else if _, err := model.Generate(t.Context(), ai.AIRequest{Prompt: "secret prompt"}); err != nil {
				t.Fatal(err)
			}

			want := "mistral_generate_request"
			if streaming {
				want = "mistral_stream_request"
			}
			for _, event := range events {
				if event.Name == want {
					if _, ok := event.Fields["prompt"]; ok {
						t.Fatalf("request observation exposed prompt without a capture policy: %#v", event.Fields)
					}
					return
				}
			}
			t.Fatalf("%s observation not emitted: %#v", want, events)
		})
	}
}
