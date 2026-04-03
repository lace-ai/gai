package context_test

import (
	"strings"
	"sync"
	"testing"

	"agent-backend/gai/context"
)

func TestNewMemoryValidation(t *testing.T) {
	_, err := context.NewMemory(0)
	if err == nil || err != context.ErrSessionIDInvalid {
		t.Fatalf("expected ErrSessionIDInvalid, got %v", err)
	}
}

func TestMemoryAddGetAndLimit(t *testing.T) {
	m, err := context.NewMemory(1)
	if err != nil {
		t.Fatalf("NewMemory returned error: %v", err)
	}

	if got := m.SessionID(); got != "1" {
		t.Fatalf("expected session id 1, got %q", got)
	}

	if _, err := m.AddMessage("first", context.RoleUser); err != nil {
		t.Fatalf("AddMessage first returned error: %v", err)
	}
	if _, err := m.AddMessage("second", context.RoleAssistant); err != nil {
		t.Fatalf("AddMessage second returned error: %v", err)
	}
	if _, err := m.AddMessage("third", context.RoleTool); err != nil {
		t.Fatalf("AddMessage third returned error: %v", err)
	}

	all, err := m.GetMessages(0)
	if err != nil {
		t.Fatalf("GetMessages returned error: %v", err)
	}
	if len(all) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(all))
	}
	if all[0].Content != "first" || all[2].Content != "third" {
		t.Fatalf("unexpected message ordering: %+v", all)
	}

	limited, err := m.GetMessages(2)
	if err != nil {
		t.Fatalf("GetMessages(limit) returned error: %v", err)
	}
	if len(limited) != 2 || limited[0].Content != "second" || limited[1].Content != "third" {
		t.Fatalf("unexpected limited messages: %+v", limited)
	}
}

func TestMemoryEnrichPromptConversationOnly(t *testing.T) {
	m, err := context.NewMemory(1)
	if err != nil {
		t.Fatalf("NewMemory returned error: %v", err)
	}

	if _, err := m.AddMessage("hello", context.RoleUser); err != nil {
		t.Fatalf("AddMessage returned error: %v", err)
	}
	if _, err := m.AddMessage("hi there", context.RoleAssistant); err != nil {
		t.Fatalf("AddMessage returned error: %v", err)
	}

	enriched, err := m.EnrichPrompt("original prompt should not be included")
	if err != nil {
		t.Fatalf("EnrichPrompt returned error: %v", err)
	}

	if !strings.HasPrefix(enriched, "<conversation>") || !strings.HasSuffix(enriched, "</conversation>") {
		t.Fatalf("expected conversation wrapper, got %q", enriched)
	}
	if strings.Contains(enriched, "original prompt should not be included") {
		t.Fatalf("expected conversation-only output, got %q", enriched)
	}
	if !strings.Contains(enriched, "<user key=0>") || !strings.Contains(enriched, "<assistant key=1>") {
		t.Fatalf("expected rendered message roles and keys, got %q", enriched)
	}
}

func TestMemoryAddMessageValidation(t *testing.T) {
	m, err := context.NewMemory(1)
	if err != nil {
		t.Fatalf("NewMemory returned error: %v", err)
	}

	if _, err := m.AddMessage("   ", context.RoleUser); err != context.ErrMessageContentEmpty {
		t.Fatalf("expected ErrMessageContentEmpty, got %v", err)
	}
	if _, err := m.AddMessage("ok", context.Role("invalid")); err != context.ErrRoleInvalid {
		t.Fatalf("expected ErrRoleInvalid, got %v", err)
	}
}

func TestMemoryConcurrentAdds(t *testing.T) {
	m, err := context.NewMemory(1)
	if err != nil {
		t.Fatalf("NewMemory returned error: %v", err)
	}

	const n = 50
	var wg sync.WaitGroup
	wg.Add(n)
	for range n {
		go func() {
			defer wg.Done()
			if _, addErr := m.AddMessage("msg", context.RoleUser); addErr != nil {
				t.Errorf("AddMessage returned error: %v", addErr)
			}
		}()
	}
	wg.Wait()

	msgs, err := m.GetMessages(0)
	if err != nil {
		t.Fatalf("GetMessages returned error: %v", err)
	}
	if len(msgs) != n {
		t.Fatalf("expected %d messages after concurrent writes, got %d", n, len(msgs))
	}
}
