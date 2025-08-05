// Package llm provides a standardized interface for interacting with various
// large language model (LLM) providers. It abstracts provider-specific APIs
// and offers a unified client for making chat completion requests. The package
// also includes features like middleware support for cross-cutting concerns
// such as caching, rate limiting, and circuit breaking.
package llm

import (
	"context"
	"errors"
	"fmt"
)

// Common errors returned by the LLM client and providers.
var (
	// ErrEmptyAPIKey indicates that an API key was required but not provided.
	ErrEmptyAPIKey = errors.New("API key cannot be empty")
	// ErrEmptyResponse indicates that the provider's API returned an empty or nil response body.
	ErrEmptyResponse = errors.New("empty response from API")
	// ErrNoResponseChoice indicates that the provider's response contained no valid choices.
	ErrNoResponseChoice = errors.New("no response choices returned")
	// ErrInvalidModel indicates that the requested model is not valid or accessible.
	ErrInvalidModel = errors.New("invalid or inaccessible model")
)

// ErrorType represents the category of an error returned by an LLM provider.
// It helps classify errors for standardized handling, such as determining retryability.
type ErrorType int

const (
	// ErrorTypeUnknown indicates an error of an undetermined category.
	ErrorTypeUnknown ErrorType = iota
	// ErrorTypeAuthentication indicates a problem with authentication or authorization (e.g., invalid API key).
	ErrorTypeAuthentication
	// ErrorTypeRateLimit indicates that a rate limit has been exceeded.
	ErrorTypeRateLimit
	// ErrorTypeBadRequest indicates a malformed request or invalid parameters.
	ErrorTypeBadRequest
	// ErrorTypeNotFound indicates that a requested resource (e.g., a model) could not be found.
	ErrorTypeNotFound
	// ErrorTypeServerError indicates a problem on the provider's end.
	ErrorTypeServerError
	// ErrorTypeContentPolicy indicates that the request was blocked by a content policy.
	ErrorTypeContentPolicy
	// ErrorTypeNetwork indicates a client-side network problem.
	ErrorTypeNetwork
	// ErrorTypeTimeout indicates that the request timed out.
	ErrorTypeTimeout
)

// ProviderError represents a structured error from an LLM provider.
// It normalizes provider-specific errors into a common format,
// including a classified error type and relevant metadata.
type ProviderError struct {
	// Type classifies the error into a standard category.
	Type ErrorType
	// Provider identifies the name of the LLM provider that produced the error.
	Provider string
	// StatusCode holds the HTTP status code from the provider's response, if applicable.
	StatusCode int
	// Message contains the user-facing error message from the provider.
	Message string
	// WrappedError holds the original underlying error, allowing for error chaining.
	WrappedError error
}

// Error returns a string representation of the ProviderError,
// satisfying the standard error interface.
func (e *ProviderError) Error() string {
	base := fmt.Sprintf("%s error", e.Provider)
	if e.StatusCode > 0 {
		base += fmt.Sprintf(" (HTTP %d)", e.StatusCode)
	}

	typeStr := e.typeString()
	if typeStr != "" {
		base += fmt.Sprintf(" [%s]", typeStr)
	}

	if e.Message != "" {
		base += ": " + e.Message
	}

	if e.WrappedError != nil {
		base += fmt.Sprintf(": %v", e.WrappedError)
	}

	return base
}

// Unwrap returns the underlying wrapped error, allowing for error inspection
// with functions like errors.Is and errors.As.
func (e *ProviderError) Unwrap() error {
	return e.WrappedError
}

// IsRetryable determines whether a request that failed with this error
// should be retried. It returns true for transient issues like rate limits
// and server-side errors.
func (e *ProviderError) IsRetryable() bool {
	switch e.Type {
	case ErrorTypeRateLimit, ErrorTypeServerError, ErrorTypeNetwork, ErrorTypeTimeout:
		return true
	default:
		return false
	}
}

// typeString returns a human-readable error type.
func (e *ProviderError) typeString() string {
	switch e.Type {
	case ErrorTypeAuthentication:
		return "authentication"
	case ErrorTypeRateLimit:
		return "rate_limit"
	case ErrorTypeBadRequest:
		return "bad_request"
	case ErrorTypeNotFound:
		return "not_found"
	case ErrorTypeServerError:
		return "server_error"
	case ErrorTypeContentPolicy:
		return "content_policy"
	case ErrorTypeNetwork:
		return "network"
	case ErrorTypeTimeout:
		return "timeout"
	default:
		return ""
	}
}

// NewProviderError creates a new ProviderError.
// This constructor is used to build standardized errors from provider-specific responses.
func NewProviderError(provider string, errType ErrorType, statusCode int, message string, wrapped error) *ProviderError {
	return &ProviderError{
		Type:         errType,
		Provider:     provider,
		StatusCode:   statusCode,
		Message:      message,
		WrappedError: wrapped,
	}
}

// ErrorClassifier standardizes provider-specific errors into ProviderError instances.
// It uses context such as HTTP status codes to determine the appropriate ErrorType.
type ErrorClassifier struct {
	// Provider is the name of the LLM provider for which this classifier works.
	Provider string
}

// ClassifyHTTPError creates a ProviderError by classifying an error based on its HTTP status code.
func (ec *ErrorClassifier) ClassifyHTTPError(statusCode int, message string, err error) *ProviderError {
	var errType ErrorType
	var userMessage string

	switch statusCode {
	case 401, 403:
		errType = ErrorTypeAuthentication
		userMessage = fmt.Sprintf("%s authentication failed", ec.Provider)
	case 429:
		errType = ErrorTypeRateLimit
		userMessage = fmt.Sprintf("%s rate limit exceeded", ec.Provider)
	case 400:
		errType = ErrorTypeBadRequest
		userMessage = message
	case 404:
		errType = ErrorTypeNotFound
		userMessage = message
	case 500, 502, 503, 504:
		errType = ErrorTypeServerError
		userMessage = message
	default:
		if statusCode >= 400 && statusCode < 500 {
			errType = ErrorTypeBadRequest
		} else if statusCode >= 500 {
			errType = ErrorTypeServerError
		} else {
			errType = ErrorTypeUnknown
		}
		userMessage = message
	}

	return NewProviderError(ec.Provider, errType, statusCode, userMessage, err)
}

// ClassifyContextError creates a ProviderError by classifying a context-related error,
// such as context.DeadlineExceeded or context.Canceled.
func (ec *ErrorClassifier) ClassifyContextError(err error) *ProviderError {
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		return NewProviderError(ec.Provider, ErrorTypeNetwork, 0, "context deadline exceeded", err)
	case errors.Is(err, context.Canceled):
		return NewProviderError(ec.Provider, ErrorTypeNetwork, 0, "request canceled", err)
	default:
		return NewProviderError(ec.Provider, ErrorTypeUnknown, 0, "", err)
	}
}
