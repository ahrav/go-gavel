// Package llm provides a middleware client that adds retry capabilities to any
// object implementing the ports.LLMClient interface. It is designed to handle
// transient network errors and rate limiting by wrapping LLM calls with
// configurable exponential backoff and jitter.
package llm

import (
	"context"
	"fmt"
	"math/rand/v2"
	"strings"
	"time"

	"github.com/ahrav/go-gavel/internal/ports"
)

// Default retry configuration constants.
const (
	// DefaultMaxAttempts is the default number of retry attempts.
	DefaultMaxAttempts = 3
	// DefaultBaseDelay is the default initial delay before the first retry.
	DefaultBaseDelay = 1 * time.Second
	// DefaultMaxDelay is the default maximum delay between retry attempts.
	DefaultMaxDelay = 30 * time.Second
	// DefaultJitterPercent is the default jitter percentage.
	DefaultJitterPercent = 0.1
)

// RetryConfig defines the configuration for retry behavior. These settings
// control the exponential backoff and jitter logic used in the client.
type RetryConfig struct {
	// MaxAttempts specifies the maximum number of times to retry a failed
	// operation. A value of 0 means no retries will be attempted.
	MaxAttempts int

	// BaseDelay sets the initial delay for the first retry attempt.
	// Subsequent delays are calculated using exponential backoff.
	BaseDelay time.Duration

	// MaxDelay caps the maximum delay between retry attempts to prevent
	// excessively long waits during exponential backoff.
	MaxDelay time.Duration

	// JitterPercent adds a random percentage of the current delay to prevent
	// a "thundering herd" scenario. It should be between 0.0 and 1.0.
	JitterPercent float64
}

// DefaultRetryConfig returns a RetryConfig with sensible default values
// suitable for most use cases.
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts:   DefaultMaxAttempts,
		BaseDelay:     DefaultBaseDelay,
		MaxDelay:      DefaultMaxDelay,
		JitterPercent: DefaultJitterPercent,
	}
}

var _ ports.LLMClient = (*RetryingLLMClient)(nil)

// RetryingLLMClient wraps an existing LLMClient with retry functionality.
// It implements the ports.LLMClient interface, adding retry logic to its
// methods. The client is thread-safe and can be used concurrently.
type RetryingLLMClient struct {
	client ports.LLMClient
	config RetryConfig
}

// NewRetryingLLMClient creates a new RetryingLLMClient that wraps the
// provided client. The retry behavior is controlled by the provided config.
func NewRetryingLLMClient(client ports.LLMClient, config RetryConfig) *RetryingLLMClient {
	return &RetryingLLMClient{
		client: client,
		config: config,
	}
}

// Complete sends a completion request to the LLM provider with retry logic.
// It implements exponential backoff with jitter for transient errors.
func (r *RetryingLLMClient) Complete(ctx context.Context, prompt string, options map[string]any) (string, error) {
	var lastErr error
	for attempt := 0; attempt <= r.config.MaxAttempts; attempt++ {
		response, err := r.client.Complete(ctx, prompt, options)
		if err == nil {
			return response, nil
		}

		lastErr = err
		if attempt == r.config.MaxAttempts || !r.isRetryableError(err) {
			break
		}

		select {
		case <-ctx.Done():
			return "", fmt.Errorf("context cancelled during retry: %w", ctx.Err())
		case <-time.After(r.calculateRetryDelay(attempt)):
		}
	}

	return "", fmt.Errorf("LLM call failed after %d attempts: %w", r.config.MaxAttempts+1, lastErr)
}

// CompleteWithUsage sends a completion request with usage tracking and retry
// logic. It implements the same retry behavior as Complete but also returns
// token usage information.
func (r *RetryingLLMClient) CompleteWithUsage(
	ctx context.Context,
	prompt string,
	options map[string]any,
) (string, int, int, error) {
	var lastErr error
	for attempt := 0; attempt <= r.config.MaxAttempts; attempt++ {
		response, tokensIn, tokensOut, err := r.client.CompleteWithUsage(ctx, prompt, options)
		if err == nil {
			return response, tokensIn, tokensOut, nil
		}

		lastErr = err
		if attempt == r.config.MaxAttempts || !r.isRetryableError(err) {
			break
		}

		select {
		case <-ctx.Done():
			return "", 0, 0, fmt.Errorf("context cancelled during retry: %w", ctx.Err())
		case <-time.After(r.calculateRetryDelay(attempt)):
		}
	}

	return "", 0, 0, fmt.Errorf("LLM call failed after %d attempts: %w", r.config.MaxAttempts+1, lastErr)
}

// EstimateTokens delegates token estimation to the wrapped client.
// This method does not implement retry logic as it is typically a local,
// deterministic calculation.
func (r *RetryingLLMClient) EstimateTokens(text string) (int, error) {
	return r.client.EstimateTokens(text)
}

// GetModel returns the model identifier from the wrapped client.
// This method does not implement retry logic as it returns static
// configuration information.
func (r *RetryingLLMClient) GetModel() string {
	return r.client.GetModel()
}

// isRetryableError determines if an error is likely transient and worth
// retrying by checking for common transient error messages.
func (r *RetryingLLMClient) isRetryableError(err error) bool {
	if err == nil {
		return false
	}

	errStr := strings.ToLower(err.Error())
	retryablePatterns := []string{
		"rate limit", "too many requests", "timeout", "connection refused",
		"connection reset", "temporary failure", "service unavailable",
		"internal server error", "bad gateway", "gateway timeout", "network",
	}

	for _, pattern := range retryablePatterns {
		if strings.Contains(errStr, pattern) {
			return true
		}
	}

	return false
}

// calculateRetryDelay calculates the appropriate delay for an exponential
// backoff strategy, including jitter to prevent request storms.
func (r *RetryingLLMClient) calculateRetryDelay(attempt int) time.Duration {
	delay := r.config.BaseDelay * time.Duration(1<<attempt)
	if delay > r.config.MaxDelay {
		delay = r.config.MaxDelay
	}

	jitter := int64(float64(delay) * r.config.JitterPercent)
	if jitter > 0 {
		//nolint:gosec // G404: math/rand is acceptable for retry jitter timing.
		delay += time.Duration(rand.Int64N(2*jitter) - jitter)
	}

	if delay < r.config.BaseDelay {
		return r.config.BaseDelay
	}

	return delay
}
