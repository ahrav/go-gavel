package llm

import (
	"context"
	"fmt"

	"golang.org/x/time/rate"
)

// rateLimitedLLM implements rate limiting using a token bucket algorithm.
// This prevents overwhelming LLM provider rate limits and ensures
// consistent request pacing across the application.
type rateLimitedLLM struct {
	next    CoreLLM
	limiter *rate.Limiter
}

// RateLimitMiddleware creates middleware that enforces rate limiting using a token bucket algorithm.
// The limit parameter sets requests per second, while burst allows
// temporary spikes above the sustained rate.
func RateLimitMiddleware(limit rate.Limit, burst int) Middleware {
	limiter := rate.NewLimiter(limit, burst)

	return func(next CoreLLM) CoreLLM {
		return &rateLimitedLLM{
			next:    next,
			limiter: limiter,
		}
	}
}

// DoRequest waits for rate limit permission before forwarding the request.
// This blocks the calling goroutine until a token is available,
// ensuring compliance with configured rate limits.
func (r *rateLimitedLLM) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
	if err := r.limiter.Wait(ctx); err != nil {
		return "", 0, 0, fmt.Errorf("rate limit: %w", err)
	}
	return r.next.DoRequest(ctx, prompt, opts)
}

// GetModel returns the model name from the wrapped implementation.
func (r *rateLimitedLLM) GetModel() string { return r.next.GetModel() }

// SetModel updates the model name in the wrapped implementation.
func (r *rateLimitedLLM) SetModel(m string) { r.next.SetModel(m) }
