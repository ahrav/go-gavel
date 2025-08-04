package llm

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTimeoutMiddleware_SucceedsWithinTimeout(t *testing.T) {
	// Given a mock that responds quickly
	mock := NewMockCoreLLM()
	mock.ResponseDelay = 10 * time.Millisecond
	timeout := 100 * time.Millisecond
	middleware := TimeoutMiddleware(timeout)
	wrapped := middleware(mock)

	// When making a request
	ctx := context.Background()
	response, tokensIn, tokensOut, err := wrapped.DoRequest(ctx, "test prompt", nil)

	// Then it should succeed
	require.NoError(t, err, "request should succeed within timeout")
	assert.Equal(t, "test response", response, "response should match")
	assert.Equal(t, 10, tokensIn, "input tokens should match")
	assert.Equal(t, 20, tokensOut, "output tokens should match")
	assert.Equal(t, 1, mock.GetCallCount(), "should call underlying implementation once")
}

func TestTimeoutMiddleware_FailsWhenExceedingTimeout(t *testing.T) {
	// Given a mock that responds slowly
	mock := NewMockCoreLLM()
	mock.ResponseDelay = 200 * time.Millisecond
	timeout := 50 * time.Millisecond
	middleware := TimeoutMiddleware(timeout)
	wrapped := middleware(mock)

	// When making a request
	ctx := context.Background()
	start := time.Now()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
	duration := time.Since(start)

	// Then it should timeout
	require.Error(t, err, "request should timeout")
	assert.True(t, errors.Is(err, context.DeadlineExceeded),
		"error should be deadline exceeded: %v", err)
	assert.Equal(t, 1, mock.GetCallCount(), "should call underlying implementation once")

	// Should timeout close to the configured timeout
	assert.Greater(t, duration, timeout, "should timeout after configured duration")
	assert.Less(t, duration, timeout+50*time.Millisecond, "should not wait much longer than timeout")
}

func TestTimeoutMiddleware_RespectsExistingContextTimeout(t *testing.T) {
	// Given a mock that responds slowly
	mock := NewMockCoreLLM()
	mock.ResponseDelay = 200 * time.Millisecond
	middlewareTimeout := 300 * time.Millisecond
	middleware := TimeoutMiddleware(middlewareTimeout)
	wrapped := middleware(mock)

	// When making a request with a shorter context timeout
	ctxTimeout := 50 * time.Millisecond
	ctx, cancel := context.WithTimeout(context.Background(), ctxTimeout)
	defer cancel()

	start := time.Now()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
	duration := time.Since(start)

	// Then it should respect the shorter context timeout
	require.Error(t, err, "request should timeout")
	assert.True(t, errors.Is(err, context.DeadlineExceeded),
		"error should be deadline exceeded: %v", err)

	// Should timeout close to the context timeout, not the middleware timeout
	assert.Greater(t, duration, ctxTimeout, "should timeout after context duration")
	assert.Less(t, duration, ctxTimeout+50*time.Millisecond, "should not wait much longer than context timeout")
	assert.Less(t, duration, middlewareTimeout, "should timeout before middleware timeout")
}

func TestTimeoutMiddleware_HandlesImmediateError(t *testing.T) {
	// Given a mock that fails immediately
	mock := NewMockCoreLLM()
	mock.Error = errors.New("immediate error")
	timeout := 100 * time.Millisecond
	middleware := TimeoutMiddleware(timeout)
	wrapped := middleware(mock)

	// When making a request
	ctx := context.Background()
	start := time.Now()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
	duration := time.Since(start)

	// Then it should fail immediately with the original error
	require.Error(t, err, "request should fail")
	assert.Equal(t, "immediate error", err.Error(), "should return original error")
	assert.Equal(t, 1, mock.GetCallCount(), "should call underlying implementation once")

	// Should not wait for timeout
	assert.Less(t, duration, 50*time.Millisecond, "should fail immediately")
}

func TestTimeoutMiddleware_PassesThroughModelMethods(t *testing.T) {
	// Given a wrapped mock
	mock := NewMockCoreLLM()
	timeout := 100 * time.Millisecond
	middleware := TimeoutMiddleware(timeout)
	wrapped := middleware(mock)

	// When calling model methods
	assert.Equal(t, "test-model", wrapped.GetModel(), "should pass through GetModel")

	wrapped.SetModel("new-model")
	assert.Equal(t, "new-model", wrapped.GetModel(), "should pass through SetModel")
	assert.Equal(t, "new-model", mock.GetModel(), "should update underlying mock")
}

func TestTimeoutMiddleware_PreservesContextValues(t *testing.T) {
	// Given a mock that succeeds
	mock := NewMockCoreLLM()
	timeout := 100 * time.Millisecond
	middleware := TimeoutMiddleware(timeout)
	wrapped := middleware(mock)

	// When making a request with context values
	ctx := context.WithValue(context.Background(), testContextKey, "test-value")
	opts := map[string]any{"temperature": 0.7}
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", opts)

	// Then context values should be preserved
	require.NoError(t, err, "request should succeed")
	assert.Equal(t, "test prompt", mock.LastPrompt, "prompt should be preserved")
	assert.Equal(t, opts, mock.LastOpts, "options should be preserved")
	assert.Equal(t, "test-value", mock.LastContext.Value(testContextKey),
		"context value should be preserved")
}

func TestTimeoutMiddleware_HandlesContextCancellation(t *testing.T) {
	// Given a mock that responds slowly
	mock := NewMockCoreLLM()
	mock.ResponseDelay = 200 * time.Millisecond
	timeout := 300 * time.Millisecond // Longer than response delay
	middleware := TimeoutMiddleware(timeout)
	wrapped := middleware(mock)

	// When making a request and cancelling context early
	ctx, cancel := context.WithCancel(context.Background())

	// Cancel after 50ms
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
	duration := time.Since(start)

	// Then it should fail with cancellation error
	require.Error(t, err, "request should be cancelled")
	assert.True(t, errors.Is(err, context.Canceled),
		"error should be context cancelled: %v", err)

	// Should be cancelled quickly
	assert.Greater(t, duration, 40*time.Millisecond, "should wait for cancellation")
	assert.Less(t, duration, 100*time.Millisecond, "should be cancelled quickly")
}

func TestTimeoutMiddleware_ZeroTimeout(t *testing.T) {
	// Given a middleware with zero timeout
	mock := NewMockCoreLLM()
	mock.ResponseDelay = 10 * time.Millisecond
	timeout := 0 * time.Millisecond
	middleware := TimeoutMiddleware(timeout)
	wrapped := middleware(mock)

	// When making a request
	ctx := context.Background()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	// Then it should timeout immediately
	require.Error(t, err, "request should timeout immediately")
	assert.True(t, errors.Is(err, context.DeadlineExceeded),
		"error should be deadline exceeded: %v", err)
}

func TestTimeoutMiddleware_VeryLongTimeout(t *testing.T) {
	// Given a middleware with very long timeout
	mock := NewMockCoreLLM()
	mock.ResponseDelay = 10 * time.Millisecond
	timeout := 10 * time.Second
	middleware := TimeoutMiddleware(timeout)
	wrapped := middleware(mock)

	// When making a request
	ctx := context.Background()
	start := time.Now()
	response, tokensIn, tokensOut, err := wrapped.DoRequest(ctx, "test prompt", nil)
	duration := time.Since(start)

	// Then it should succeed normally
	require.NoError(t, err, "request should succeed")
	assert.Equal(t, "test response", response, "response should match")
	assert.Equal(t, 10, tokensIn, "input tokens should match")
	assert.Equal(t, 20, tokensOut, "output tokens should match")

	// Should not wait for timeout
	assert.Less(t, duration, 100*time.Millisecond, "should not wait for long timeout")
}

func TestTimeoutMiddleware_MultipleSimultaneousRequests(t *testing.T) {
	// Given a middleware with timeout
	mock := NewMockCoreLLM()
	mock.ResponseDelay = 10 * time.Millisecond // Reduced delay
	timeout := 200 * time.Millisecond          // Increased timeout
	middleware := TimeoutMiddleware(timeout)
	wrapped := middleware(mock)

	// When making multiple simultaneous requests
	const numRequests = 3 // Reduced number
	errors := make(chan error, numRequests)

	for i := range numRequests {
		go func(i int) {
			ctx := context.Background()
			_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
			errors <- err
		}(i)
	}

	// Then all requests should succeed
	for i := range numRequests {
		select {
		case err := <-errors:
			assert.NoError(t, err, "request %d should succeed", i)
		case <-time.After(500 * time.Millisecond):
			t.Fatalf("request %d timed out", i)
		}
	}

	assert.Equal(t, numRequests, mock.GetCallCount(), "should handle all requests")
}
