package context

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func (b *Builder) SystemPrompt(ctx context.Context, path string) error {
	sysPrompt, err := loadPromptFromFile(path)
	if err != nil {
		return err
	}
	part := NewTextPart(sysPrompt)
	b.AppendSystemInstructions(ctx, part)
	return nil
}

func loadPromptFromFile(path string) (string, error) {
	cleanPath := strings.TrimSpace(path)
	if cleanPath == "" {
		return "", ErrPromptPathEmpty
	}

	ext := strings.ToLower(filepath.Ext(cleanPath))
	if ext != ".md" && ext != ".txt" {
		return "", fmt.Errorf("%w: %s", ErrPromptFileType, cleanPath)
	}

	content, err := os.ReadFile(cleanPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", fmt.Errorf("%w: %s", ErrPromptMissing, cleanPath)
		}
		return "", err
	}

	return strings.TrimSpace(string(content)), nil
}
