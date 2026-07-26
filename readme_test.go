package gai_test

import (
	"os"
	"strings"
	"testing"
)

func TestREADMEReflectsLicenseAndBuiltInProviders(t *testing.T) {
	license, err := os.ReadFile("LICENSE")
	if err != nil {
		t.Fatalf("read LICENSE: %v", err)
	}
	readme, err := os.ReadFile("README.md")
	if err != nil {
		t.Fatalf("read README.md: %v", err)
	}

	licenseText := string(license)
	readmeText := string(readme)
	for _, want := range []string{
		"MIT License",
		"Copyright (c) 2026 Samuel Konrad",
	} {
		if !strings.Contains(licenseText, want) {
			t.Errorf("LICENSE is missing %q", want)
		}
	}
	for _, want := range []string{
		"license-MIT",
		"actions/workflows/go.yml/badge.svg",
		"[MIT License](LICENSE)",
		"Copyright (c) 2026 Samuel Konrad",
		"### 🟣 Anthropic",
		"### ♊ Gemini",
		"### 🌀 Mistral",
		"### 🟢 OpenAI",
	} {
		if !strings.Contains(readmeText, want) {
			t.Errorf("README.md is missing %q", want)
		}
	}
}
