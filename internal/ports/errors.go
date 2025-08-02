package ports

import (
	"errors"
	"fmt"
	"time"
)

// Common infrastructure errors that can occur during external service
// interactions.
var (
	// ErrTokenLimitExceeded indicates that the LLM token limit has been
	// exceeded.
	ErrTokenLimitExceeded = errors.New("token limit exceeded")

	// ErrRateLimited indicates that the service has rate limited the request.
	ErrRateLimited = errors.New("rate limited")

	// ErrServiceUnavailable indicates that the external service is unavailable.
	ErrServiceUnavailable = errors.New("service unavailable")

	// ErrTimeout indicates that an operation timed out.
	ErrTimeout = errors.New("operation timed out")

	// ErrInvalidResponse indicates that the service returned an invalid
	// response.
	ErrInvalidResponse = errors.New("invalid response")

	// ErrAuthenticationFailed indicates that authentication with the
	// service failed.
	ErrAuthenticationFailed = errors.New("authentication failed")

	// ErrCacheCorrupted indicates that cached data is corrupted or invalid.
	ErrCacheCorrupted = errors.New("cache corrupted")

	// ErrConfigNotFound indicates that required configuration is missing.
	ErrConfigNotFound = errors.New("configuration not found")
)

// LLMError represents an error from an LLM provider.
// It includes details about the model, operation, and any rate limit
// information.
type LLMError struct {
	// Model is the identifier of the LLM model that generated the error.
	Model string

	// Operation is the name of the operation that failed.
	Operation string

	// Err is the underlying error that occurred.
	Err error

	// TokensUsed is the number of tokens consumed before the error occurred.
	TokensUsed int

	// RetryAfter indicates how long to wait before retrying, if applicable.
	RetryAfter *time.Duration
}

// Error implements the error interface for LLMError.
func (e *LLMError) Error() string {
	msg := fmt.Sprintf("LLM error: model=%s, operation=%s, err=%v", e.Model, e.Operation, e.Err)
	if e.TokensUsed > 0 {
		msg += fmt.Sprintf(", tokens_used=%d", e.TokensUsed)
	}
	if e.RetryAfter != nil {
		msg += fmt.Sprintf(", retry_after=%v", *e.RetryAfter)
	}
	return msg
}

// Unwrap returns the underlying error.
func (e *LLMError) Unwrap() error {
	return e.Err
}

// IsRetryable returns true if the error is temporary and the operation
// can be retried.
func (e *LLMError) IsRetryable() bool {
	// Only network/service-level errors are retryable; logic errors are not
	return errors.Is(e.Err, ErrRateLimited) ||
		errors.Is(e.Err, ErrServiceUnavailable) ||
		errors.Is(e.Err, ErrTimeout)
}

// NewLLMError creates a new LLMError with the given details.
func NewLLMError(model, operation string, err error) *LLMError {
	return &LLMError{
		Model:     model,
		Operation: operation,
		Err:       err,
	}
}

// CacheError represents an error from cache operations.
// It includes the key and operation that failed.
type CacheError struct {
	// Key is the cache key that was involved in the failed operation.
	Key string

	// Operation is the name of the cache operation that failed.
	Operation string

	// Err is the underlying error that caused the cache operation to fail.
	Err error
}

// Error implements the error interface for CacheError.
func (e *CacheError) Error() string {
	return fmt.Sprintf("cache error: operation=%s, key=%s, err=%v", e.Operation, e.Key, e.Err)
}

// Unwrap returns the underlying error.
func (e *CacheError) Unwrap() error { return e.Err }

// NewCacheError creates a new CacheError with the given details.
func NewCacheError(key, operation string, err error) *CacheError {
	return &CacheError{
		Key:       key,
		Operation: operation,
		Err:       err,
	}
}

// MetricsError represents an error from metrics collection operations.
type MetricsError struct {
	// Metric is the name of the metric that was being collected when the
	// error occurred.
	Metric string

	// Operation is the name of the metrics operation that failed.
	Operation string

	// Err is the underlying error that caused the metrics operation to fail.
	Err error
}

// Error implements the error interface for MetricsError.
func (e *MetricsError) Error() string {
	return fmt.Sprintf("metrics error: operation=%s, metric=%s, err=%v", e.Operation, e.Metric, e.Err)
}

// Unwrap returns the underlying error.
func (e *MetricsError) Unwrap() error { return e.Err }

// NewMetricsError creates a new MetricsError with the given details.
func NewMetricsError(metric, operation string, err error) *MetricsError {
	return &MetricsError{
		Metric:    metric,
		Operation: operation,
		Err:       err,
	}
}

// ConfigError represents an error from configuration operations.
type ConfigError struct {
	// ConfigKey is the configuration key that was involved in the failed
	// operation.
	ConfigKey string

	// Err is the underlying error that caused the configuration operation
	// to fail.
	Err error
}

// Error implements the error interface for ConfigError.
func (e *ConfigError) Error() string {
	return fmt.Sprintf("config error: key=%s, err=%v", e.ConfigKey, e.Err)
}

// Unwrap returns the underlying error.
func (e *ConfigError) Unwrap() error { return e.Err }

// NewConfigError creates a new ConfigError with the given details.
func NewConfigError(key string, err error) *ConfigError {
	return &ConfigError{
		ConfigKey: key,
		Err:       err,
	}
}
