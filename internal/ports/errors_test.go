package ports

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// TestLLMError tests the functionality of the LLMError error type.
// It covers error creation, message formatting, and retryable logic.
func TestLLMError(t *testing.T) {
	t.Run("basic error", func(t *testing.T) {
		err := NewLLMError("gpt-4", "Complete", ErrTokenLimitExceeded)

		assert.Equal(t, "LLM error: model=gpt-4, operation=Complete, err=token limit exceeded", err.Error())
		assert.Equal(t, "gpt-4", err.Model)
		assert.Equal(t, "Complete", err.Operation)
		assert.True(t, errors.Is(err, ErrTokenLimitExceeded))
	})

	t.Run("with tokens used", func(t *testing.T) {
		err := &LLMError{
			Model:      "claude-3",
			Operation:  "Complete",
			Err:        ErrTokenLimitExceeded,
			TokensUsed: 8192,
		}

		assert.Contains(t, err.Error(), "tokens_used=8192")
	})

	t.Run("with retry after", func(t *testing.T) {
		retryAfter := 30 * time.Second
		err := &LLMError{
			Model:      "gpt-3.5",
			Operation:  "Complete",
			Err:        ErrRateLimited,
			RetryAfter: &retryAfter,
		}

		assert.Contains(t, err.Error(), "retry_after=30s")
	})

	t.Run("retryable errors", func(t *testing.T) {
		retryableErrors := []error{
			ErrRateLimited,
			ErrServiceUnavailable,
			ErrTimeout,
		}

		for _, baseErr := range retryableErrors {
			err := NewLLMError("test-model", "Test", baseErr)
			assert.True(t, err.IsRetryable(), "%v should be retryable", baseErr)
		}

		nonRetryableErrors := []error{
			ErrTokenLimitExceeded,
			ErrInvalidResponse,
			ErrAuthenticationFailed,
		}

		for _, baseErr := range nonRetryableErrors {
			err := NewLLMError("test-model", "Test", baseErr)
			assert.False(t, err.IsRetryable(), "%v should not be retryable", baseErr)
		}
	})
}

// TestCacheError tests the functionality of the CacheError error type.
// It verifies that the error message is formatted correctly and contains the expected context.
func TestCacheError(t *testing.T) {
	tests := []struct {
		name      string
		key       string
		operation string
		err       error
		wantMsg   string
	}{
		{
			name:      "cache miss",
			key:       "test-key",
			operation: "Get",
			err:       errors.New("key not found"),
			wantMsg:   "cache error: operation=Get, key=test-key, err=key not found",
		},
		{
			name:      "cache corruption",
			key:       "user:123",
			operation: "Get",
			err:       ErrCacheCorrupted,
			wantMsg:   "cache error: operation=Get, key=user:123, err=cache corrupted",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := NewCacheError(tt.key, tt.operation, tt.err)

			assert.Equal(t, tt.wantMsg, err.Error())
			assert.Equal(t, tt.key, err.Key)
			assert.Equal(t, tt.operation, err.Operation)
			assert.True(t, errors.Is(err, tt.err))
		})
	}
}

// TestMetricsError tests the functionality of the MetricsError error type.
// It ensures that the error message is formatted correctly and includes the necessary context.
func TestMetricsError(t *testing.T) {
	err := NewMetricsError("api_latency", "RecordHistogram", errors.New("connection refused"))

	assert.Equal(t, "metrics error: operation=RecordHistogram, metric=api_latency, err=connection refused", err.Error())
	assert.Equal(t, "api_latency", err.Metric)
	assert.Equal(t, "RecordHistogram", err.Operation)
}

// TestConfigError tests the functionality of the ConfigError error type.
// It verifies that the error message is formatted correctly and contains the relevant configuration key.
func TestConfigError(t *testing.T) {
	err := NewConfigError("database.url", ErrConfigNotFound)

	assert.Equal(t, "config error: key=database.url, err=configuration not found", err.Error())
	assert.Equal(t, "database.url", err.ConfigKey)
	assert.True(t, errors.Is(err, ErrConfigNotFound))
}

// TestCommonInfrastructureErrors tests that the common infrastructure errors are defined.
// It checks that each error has the expected error message.
func TestCommonInfrastructureErrors(t *testing.T) {
	tests := []struct {
		err     error
		message string
	}{
		{ErrTokenLimitExceeded, "token limit exceeded"},
		{ErrRateLimited, "rate limited"},
		{ErrServiceUnavailable, "service unavailable"},
		{ErrTimeout, "operation timed out"},
		{ErrInvalidResponse, "invalid response"},
		{ErrAuthenticationFailed, "authentication failed"},
		{ErrCacheCorrupted, "cache corrupted"},
		{ErrConfigNotFound, "configuration not found"},
	}

	for _, tt := range tests {
		t.Run(tt.message, func(t *testing.T) {
			assert.Equal(t, tt.message, tt.err.Error())
		})
	}
}

// TestErrorUnwrapping tests that all custom error types in the package support unwrapping.
// It ensures that the underlying error can be extracted correctly using errors.Is and Unwrap.
func TestErrorUnwrapping(t *testing.T) {
	baseErr := errors.New("underlying error")

	errorList := []interface {
		error
		Unwrap() error
	}{
		NewLLMError("model", "op", baseErr),
		NewCacheError("key", "op", baseErr),
		NewMetricsError("metric", "op", baseErr),
		NewConfigError("key", baseErr),
	}

	for _, err := range errorList {
		unwrapped := err.Unwrap()
		assert.Equal(t, baseErr, unwrapped, "%T should unwrap to base error", err)
		assert.True(t, errors.Is(err, baseErr), "%T should match base error with Is", err)
	}
}
