package llm

import (
	"context"
	"fmt"
	"math/rand"
	"time"
)

// retryLLM implements automatic retry logic with exponential backoff.
// This handles transient failures by retrying requests with increasing
// delays while respecting circuit breaker and timeout constraints.
type retryLLM struct {
	next       CoreLLM
	maxRetries int
	baseDelay  time.Duration
	maxDelay   time.Duration
}

// RetryMiddleware creates middleware that automatically retries failed requests
// with exponential backoff. This helps handle transient failures and improves
// overall reliability of LLM interactions.
func RetryMiddleware(maxRetries int, baseDelay, maxDelay time.Duration) Middleware {
	return func(next CoreLLM) CoreLLM {
		return &retryLLM{
			next:       next,
			maxRetries: maxRetries,
			baseDelay:  baseDelay,
			maxDelay:   maxDelay,
		}
	}
}

// DoRequest executes the request with automatic retry logic.
// It implements exponential backoff and respects circuit breaker states
// and context cancellation to avoid unnecessary retries.
func (r *retryLLM) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
	var lastErr error

	for attempt := 0; attempt <= r.maxRetries; attempt++ {
		response, tokensIn, tokensOut, err := r.next.DoRequest(ctx, prompt, opts)
		if err == nil {
			return response, tokensIn, tokensOut, nil
		}

		lastErr = err

		if err == ErrCircuitOpen || ctx.Err() != nil {
			break
		}

		if attempt == r.maxRetries {
			break
		}

		delay := r.calculateDelay(attempt)

		select {
		case <-ctx.Done():
			return "", 0, 0, ctx.Err()
		case <-time.After(delay):
			// Continue to next attempt.
		}
	}

	return "", 0, 0, fmt.Errorf("request failed after %d attempts: %w", r.maxRetries+1, lastErr)
}

func (r *retryLLM) calculateDelay(attempt int) time.Duration {
	// Exponential backoff with jitter.
	if attempt < 0 {
		attempt = 0
	}
	if attempt > 30 {
		attempt = 30
	}
	// #nosec G115 - attempt is bounded between 0 and 30
	multiplier := 1 << uint(attempt)
	delay := time.Duration(float64(r.baseDelay) * float64(multiplier))

	// Add jitter (±25%)
	// #nosec G404 - Using weak RNG is acceptable for jitter calculation
	jitter := time.Duration(rand.Float64() * float64(delay) * 0.5)
	delay = delay + jitter - (delay / 4)

	if delay > r.maxDelay {
		delay = r.maxDelay
	}

	return delay
}

// GetModel returns the model name from the wrapped implementation.
func (r *retryLLM) GetModel() string { return r.next.GetModel() }

// SetModel updates the model name in the wrapped implementation.
func (r *retryLLM) SetModel(m string) { r.next.SetModel(m) }
