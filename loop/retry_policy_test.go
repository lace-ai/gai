package loop_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/lace-ai/gai/ai"
	"github.com/lace-ai/gai/loop"
)

func TestRetryPolicyClassifiesProviderErrors(t *testing.T) {
	policy := loop.RetryPolicy{MaxRetries: 1}
	if !policy.ShouldRetry(0, &ai.ProviderError{Kind: ai.ProviderErrorRateLimited}) {
		t.Fatal("rate limit should retry")
	}
	if policy.ShouldRetry(0, &ai.ProviderError{Kind: ai.ProviderErrorInvalidRequest}) {
		t.Fatal("invalid request must not retry")
	}
	if policy.ShouldRetry(1, &ai.ProviderError{Kind: ai.ProviderErrorTransient}) {
		t.Fatal("exhausted retries must not retry")
	}
}

func TestRetryPolicyAttemptTimeoutRetriesButCallerCancellationDoesNot(t *testing.T) {
	policy := loop.RetryPolicy{MaxRetries: 1, AttemptTimeout: time.Millisecond}
	if !policy.ShouldRetry(0, loop.ErrAttemptTimeout) {
		t.Fatal("policy attempt timeout should retry")
	}
	if policy.ShouldRetry(0, context.Canceled) || policy.ShouldRetry(0, context.DeadlineExceeded) {
		t.Fatal("caller cancellation and deadline must not retry")
	}
	if !errors.Is(loop.ErrAttemptTimeout, context.DeadlineExceeded) {
		t.Fatal("attempt timeout must retain deadline identity")
	}
}
