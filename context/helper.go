package context

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func (b *Builder) SystemPrompt(path string) (*Builder, error) {
	sysPrompt, err := loadPromptFromFile(path)
	if err != nil {
		return b, err
	}

	return b.System(StaticPart(
		"base",
		sysPrompt,
	).RequiredPart()), nil
}

func (b *Builder) ToolSysPrompt(path string) (*Builder, error) {
	sysPrompt, err := loadPromptFromFile(path)
	if err != nil {
		return b, err
	}

	return b.System(StaticPart(
		"tool",
		sysPrompt,
	).RequiredPart()), nil
}

func (b *Builder) UserPrompt(text string) *Builder {
	return b.User(StaticPart(
		"request",
		text,
	).RequiredPart())
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
