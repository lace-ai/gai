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
		"Bundled fallback models include:",
		"`gpt-5.6`, `gpt-5.6-terra`, `gpt-5.6-sol`, `gpt-5.6-luna`",
		"`gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`",
		"`gpt-4o`, `gpt-4o-mini`",
		"`o3`, `o3-mini`, `o4-mini`",
	} {
		if !strings.Contains(readmeText, want) {
			t.Errorf("README.md is missing %q", want)
		}
	}
}
