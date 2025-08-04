package llm

import (
	"context"
	"time"
)

// timeoutLLM implements request timeout functionality.
// This ensures requests don't hang indefinitely and provides
// predictable response times for distributed systems.
type timeoutLLM struct {
	next    CoreLLM
	timeout time.Duration
}

// TimeoutMiddleware creates middleware that enforces request timeouts.
// This prevents requests from hanging indefinitely and enables
// proper timeout handling in distributed systems.
func TimeoutMiddleware(timeout time.Duration) Middleware {
	return func(next CoreLLM) CoreLLM {
		return &timeoutLLM{
			next:    next,
			timeout: timeout,
		}
	}
}

// DoRequest executes the request with a timeout context.
// If the request doesn't complete within the timeout duration,
// it returns a context deadline exceeded error.
func (t *timeoutLLM) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
	ctx, cancel := context.WithTimeout(ctx, t.timeout)
	defer cancel()
	return t.next.DoRequest(ctx, prompt, opts)
}

// GetModel returns the model name from the wrapped implementation.
func (t *timeoutLLM) GetModel() string { return t.next.GetModel() }

// SetModel updates the model name in the wrapped implementation.
func (t *timeoutLLM) SetModel(m string) { t.next.SetModel(m) }
