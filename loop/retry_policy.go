package loop

import (
	"context"
	"errors"
	"math/rand/v2"
	"time"

	"github.com/lace-ai/gai/ai"
)

// ErrAttemptTimeout identifies a deadline created by RetryPolicy, rather than
// a caller or total-run deadline. It wraps context.DeadlineExceeded so callers
// can retain standard context handling while policy can retry it safely.
var ErrAttemptTimeout = attemptTimeoutError{}

type attemptTimeoutError struct{}

func (attemptTimeoutError) Error() string { return "retry attempt timeout" }
func (attemptTimeoutError) Unwrap() error { return context.DeadlineExceeded }

// RetryPolicy controls classified model-generation retries. A non-nil policy
// takes precedence over RetryCount; MaxRetries: 0 explicitly disables retries.
type RetryPolicy struct {
	MaxRetries     int
	InitialBackoff time.Duration
	MaxBackoff     time.Duration
	Multiplier     float64
	Jitter         float64
	// JitterSource supplies a value in [0, 1) when applying jitter. Nil uses
	// math/rand/v2's default source.
	JitterSource func() float64
	// Wait blocks for a retry delay or returns the context error when canceled.
	// Nil waits using a wall-clock timer.
	Wait              func(context.Context, time.Duration) error
	RespectRetryAfter bool
	AttemptTimeout    time.Duration
	TotalTimeout      time.Duration
}

func (p RetryPolicy) ShouldRetry(retries int, err error) bool {
	if retries >= p.MaxRetries || err == nil || errors.Is(err, context.Canceled) {
		return false
	}
	if errors.Is(err, ErrAttemptTimeout) {
		return true
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var provider *ai.ProviderError
	if !errors.As(err, &provider) {
		return false
	}
	return provider.Kind == ai.ProviderErrorRateLimited || provider.Kind == ai.ProviderErrorTransient
}

func (p RetryPolicy) Backoff(retries int, err error) time.Duration {
	var provider *ai.ProviderError
	if p.RespectRetryAfter && errors.As(err, &provider) && provider.RetryAfter > 0 {
		return provider.RetryAfter
	}
	d := p.InitialBackoff
	if d <= 0 {
		return 0
	}
	m := p.Multiplier
	if m <= 0 {
		m = 2
	}
	for range retries {
		d = time.Duration(float64(d) * m)
	}
	if p.MaxBackoff > 0 && d > p.MaxBackoff {
		d = p.MaxBackoff
	}
	if p.Jitter > 0 {
		source := p.JitterSource
		if source == nil {
			source = rand.Float64
		}
		d = time.Duration(float64(d) * (1 - p.Jitter + source()*2*p.Jitter))
	}
	if p.MaxBackoff > 0 && d > p.MaxBackoff {
		d = p.MaxBackoff
	}
	return d
}

func (p RetryPolicy) wait(ctx context.Context, delay time.Duration) error {
	if p.Wait != nil {
		return p.Wait(ctx, delay)
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
