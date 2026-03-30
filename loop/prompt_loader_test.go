package loop

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadOptionalPromptFromFile(t *testing.T) {
	fallback := "default prompt"

	got, err := LoadOptionalPromptFromFile("", fallback)
	if err != nil {
		t.Fatalf("LoadOptionalPromptFromFile empty path returned error: %v", err)
	}
	if got != fallback {
		t.Fatalf("expected fallback for empty path, got %q", got)
	}

	dir := t.TempDir()
	missing := filepath.Join(dir, "missing.md")
	got, err = LoadOptionalPromptFromFile(missing, fallback)
	if err != nil {
		t.Fatalf("LoadOptionalPromptFromFile missing file returned error: %v", err)
	}
	if got != fallback {
		t.Fatalf("expected fallback for missing path, got %q", got)
	}
}

func TestLoadOptionalPromptFromFileInvalidType(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "prompt.json")
	if err := os.WriteFile(path, []byte(`{"x":1}`), 0o600); err != nil {
		t.Fatalf("failed writing prompt file: %v", err)
	}

	_, err := LoadOptionalPromptFromFile(path, "fallback")
	if !errors.Is(err, ErrPromptFileType) {
		t.Fatalf("expected ErrPromptFileType, got %v", err)
	}
}
