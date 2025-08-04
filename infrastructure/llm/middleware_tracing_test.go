package llm

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestTracingMiddleware_PassesThroughSuccessfulRequests tests that the tracing
// middleware correctly passes through successful requests.
func TestTracingMiddleware_PassesThroughSuccessfulRequests(t *testing.T) {
	mock := NewMockCoreLLM()
	middleware := TracingMiddleware("test-service")
	wrapped := middleware(mock)

	ctx := context.Background()
	response, tokensIn, tokensOut, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.NoError(t, err, "request should succeed")
	assert.Equal(t, "test response", response, "response should match")
	assert.Equal(t, 10, tokensIn, "input tokens should match")
	assert.Equal(t, 20, tokensOut, "output tokens should match")
	assert.Equal(t, 1, mock.GetCallCount(), "should call underlying implementation once")
}

// TestTracingMiddleware_PassesThroughFailedRequests tests that the tracing
// middleware correctly passes through failed requests.
func TestTracingMiddleware_PassesThroughFailedRequests(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.Error = errors.New("service error")
	middleware := TracingMiddleware("test-service")
	wrapped := middleware(mock)

	ctx := context.Background()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.Error(t, err, "request should fail")
	assert.Equal(t, "service error", err.Error(), "should return original error")
	assert.Equal(t, 1, mock.GetCallCount(), "should call underlying implementation once")
}

// TestTracingMiddleware_PassesThroughModelMethods tests that the tracing middleware
// correctly passes through calls to the underlying CoreLLM's methods.
func TestTracingMiddleware_PassesThroughModelMethods(t *testing.T) {
	mock := NewMockCoreLLM()
	middleware := TracingMiddleware("test-service")
	wrapped := middleware(mock)

	assert.Equal(t, "test-model", wrapped.GetModel(), "should pass through GetModel")

	wrapped.SetModel("new-model")
	assert.Equal(t, "new-model", wrapped.GetModel(), "should pass through SetModel")
	assert.Equal(t, "new-model", mock.GetModel(), "should update underlying mock")
}

// TestTracingMiddleware_PreservesContextAndOptions tests that the tracing
// middleware preserves the context and options passed to the DoRequest method.
func TestTracingMiddleware_PreservesContextAndOptions(t *testing.T) {
	mock := NewMockCoreLLM()
	middleware := TracingMiddleware("test-service")
	wrapped := middleware(mock)

	ctx := context.WithValue(context.Background(), testContextKey, "test-value")
	opts := map[string]any{"temperature": 0.7, "max_tokens": 100}
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", opts)

	require.NoError(t, err, "request should succeed")
	assert.Equal(t, "test prompt", mock.LastPrompt, "prompt should be preserved")
	assert.Equal(t, opts, mock.LastOpts, "options should be preserved")
	assert.Equal(t, "test-value", mock.LastContext.Value(testContextKey),
		"context value should be preserved")
}

// TestTracingMiddleware_HandlesContextCancellation tests that the tracing
// middleware correctly handles context cancellation.
func TestTracingMiddleware_HandlesContextCancellation(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.ResponseDelay = 100 * time.Millisecond
	middleware := TracingMiddleware("test-service")
	wrapped := middleware(mock)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.Error(t, err, "request should be cancelled")
	assert.Equal(t, context.Canceled, err, "should return context cancelled error")
}

// TestTracingMiddleware_HandlesCircuitBreakerErrors tests that the tracing
// middleware correctly handles errors from the circuit breaker.
func TestTracingMiddleware_HandlesCircuitBreakerErrors(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.Error = ErrCircuitOpen
	middleware := TracingMiddleware("test-service")
	wrapped := middleware(mock)

	ctx := context.Background()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.Error(t, err, "request should fail")
	assert.Equal(t, ErrCircuitOpen, err, "should return circuit breaker error")
	assert.Equal(t, 1, mock.GetCallCount(), "should call underlying implementation once")
}

// TestTracingMiddleware_WorksWithDifferentServiceNames tests that the tracing
// middleware works correctly with various service names.
func TestTracingMiddleware_WorksWithDifferentServiceNames(t *testing.T) {
	serviceNames := []string{
		"llm-service",
		"ai-gateway",
		"",
		"service-with-dashes",
		"ServiceWithCaps",
	}

	for _, serviceName := range serviceNames {
		t.Run(serviceName, func(t *testing.T) {
			mock := NewMockCoreLLM()
			middleware := TracingMiddleware(serviceName)
			wrapped := middleware(mock)

			ctx := context.Background()
			response, tokensIn, tokensOut, err := wrapped.DoRequest(ctx, "test prompt", nil)

			require.NoError(t, err, "request should succeed")
			assert.Equal(t, "test response", response, "response should match")
			assert.Equal(t, 10, tokensIn, "input tokens should match")
			assert.Equal(t, 20, tokensOut, "output tokens should match")
		})
	}
}

// TestTracingMiddleware_PreservesTokenCounts tests that the tracing middleware
// correctly preserves the token counts from the underlying CoreLLM.
func TestTracingMiddleware_PreservesTokenCounts(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.TokensIn = 150
	mock.TokensOut = 75
	middleware := TracingMiddleware("test-service")
	wrapped := middleware(mock)

	ctx := context.Background()
	response, tokensIn, tokensOut, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.NoError(t, err, "request should succeed")
	assert.Equal(t, "test response", response, "response should match")
	assert.Equal(t, 150, tokensIn, "input tokens should be preserved")
	assert.Equal(t, 75, tokensOut, "output tokens should be preserved")
}

// TestTracingMiddleware_HandlesEmptyPrompt tests that the tracing middleware
// correctly handles an empty prompt.
func TestTracingMiddleware_HandlesEmptyPrompt(t *testing.T) {
	mock := NewMockCoreLLM()
	middleware := TracingMiddleware("test-service")
	wrapped := middleware(mock)

	ctx := context.Background()
	response, tokensIn, tokensOut, err := wrapped.DoRequest(ctx, "", nil)

	require.NoError(t, err, "request should succeed")
	assert.Equal(t, "test response", response, "response should match")
	assert.Equal(t, 10, tokensIn, "input tokens should match")
	assert.Equal(t, 20, tokensOut, "output tokens should match")
	assert.Equal(t, "", mock.LastPrompt, "empty prompt should be preserved")
}

// TestTracingMiddleware_HandlesNilOptions tests that the tracing middleware
// correctly handles nil options.
func TestTracingMiddleware_HandlesNilOptions(t *testing.T) {
	mock := NewMockCoreLLM()
	middleware := TracingMiddleware("test-service")
	wrapped := middleware(mock)

	ctx := context.Background()
	response, tokensIn, tokensOut, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.NoError(t, err, "request should succeed")
	assert.Equal(t, "test response", response, "response should match")
	assert.Equal(t, 10, tokensIn, "input tokens should match")
	assert.Equal(t, 20, tokensOut, "output tokens should match")
	assert.Nil(t, mock.LastOpts, "nil options should be preserved")
}

// TestTracingMiddleware_WorksInChain tests that the tracing middleware works
// correctly when chained with other middlewares.
func TestTracingMiddleware_WorksInChain(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.ResponseDelay = 10 * time.Millisecond

	timeout := TimeoutMiddleware(100 * time.Millisecond)
	tracing := TracingMiddleware("test-service")

	wrapped := tracing(timeout(mock))

	ctx := context.Background()
	response, tokensIn, tokensOut, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.NoError(t, err, "request should succeed through middleware chain")
	assert.Equal(t, "test response", response, "response should match")
	assert.Equal(t, 10, tokensIn, "input tokens should match")
	assert.Equal(t, 20, tokensOut, "output tokens should match")
	assert.Equal(t, 1, mock.GetCallCount(), "should call underlying implementation once")
}
