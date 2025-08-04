package llm

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestTimeoutMiddleware_SucceedsWithinTimeout tests that the timeout middleware
// allows a request to succeed if it completes within the specified timeout.
func TestTimeoutMiddleware_SucceedsWithinTimeout(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.ResponseDelay = 10 * time.Millisecond
	timeout := 100 * time.Millisecond
	middleware := TimeoutMiddleware(timeout)
	wrapped := middleware(mock)

	ctx := context.Background()
	response, tokensIn, tokensOut, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.NoError(t, err, "request should succeed within timeout")
	assert.Equal(t, "test response", response, "response should match")
	assert.Equal(t, 10, tokensIn, "input tokens should match")
	assert.Equal(t, 20, tokensOut, "output tokens should match")
	assert.Equal(t, 1, mock.GetCallCount(), "should call underlying implementation once")
}

// TestTimeoutMiddleware_FailsWhenExceedingTimeout tests that the timeout middleware
// correctly times out a request that exceeds the specified timeout.
func TestTimeoutMiddleware_FailsWhenExceedingTimeout(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.ResponseDelay = 200 * time.Millisecond
	timeout := 50 * time.Millisecond
	middleware := TimeoutMiddleware(timeout)
	wrapped := middleware(mock)

	ctx := context.Background()
	start := time.Now()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
	duration := time.Since(start)

	require.Error(t, err, "request should timeout")
	assert.True(t, errors.Is(err, context.DeadlineExceeded),
		"error should be deadline exceeded: %v", err)
	assert.Equal(t, 1, mock.GetCallCount(), "should call underlying implementation once")

	assert.Greater(t, duration, timeout, "should timeout after configured duration")
	assert.Less(t, duration, timeout+50*time.Millisecond, "should not wait much longer than timeout")
}

// TestTimeoutMiddleware_RespectsExistingContextTimeout tests that the timeout
// middleware respects a shorter timeout defined in the request's context.
func TestTimeoutMiddleware_RespectsExistingContextTimeout(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.ResponseDelay = 200 * time.Millisecond
	middlewareTimeout := 300 * time.Millisecond
	middleware := TimeoutMiddleware(middlewareTimeout)
	wrapped := middleware(mock)

	ctxTimeout := 50 * time.Millisecond
	ctx, cancel := context.WithTimeout(context.Background(), ctxTimeout)
	defer cancel()

	start := time.Now()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
	duration := time.Since(start)

	require.Error(t, err, "request should timeout")
	assert.True(t, errors.Is(err, context.DeadlineExceeded),
		"error should be deadline exceeded: %v", err)

	assert.Greater(t, duration, ctxTimeout, "should timeout after context duration")
	assert.Less(t, duration, ctxTimeout+50*time.Millisecond, "should not wait much longer than context timeout")
	assert.Less(t, duration, middlewareTimeout, "should timeout before middleware timeout")
}

// TestTimeoutMiddleware_HandlesImmediateError tests that the timeout middleware
// correctly handles an immediate error from the underlying LLM without waiting
// for the timeout.
func TestTimeoutMiddleware_HandlesImmediateError(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.Error = errors.New("immediate error")
	timeout := 100 * time.Millisecond
	middleware := TimeoutMiddleware(timeout)
	wrapped := middleware(mock)

	ctx := context.Background()
	start := time.Now()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
	duration := time.Since(start)

	require.Error(t, err, "request should fail")
	assert.Equal(t, "immediate error", err.Error(), "should return original error")
	assert.Equal(t, 1, mock.GetCallCount(), "should call underlying implementation once")

	assert.Less(t, duration, 50*time.Millisecond, "should fail immediately")
}

// TestTimeoutMiddleware_PassesThroughModelMethods tests that the timeout middleware
// correctly passes through calls to the underlying CoreLLM's methods.
func TestTimeoutMiddleware_PassesThroughModelMethods(t *testing.T) {
	mock := NewMockCoreLLM()
	timeout := 100 * time.Millisecond
	middleware := TimeoutMiddleware(timeout)
	wrapped := middleware(mock)

	assert.Equal(t, "test-model", wrapped.GetModel(), "should pass through GetModel")

	wrapped.SetModel("new-model")
	assert.Equal(t, "new-model", wrapped.GetModel(), "should pass through SetModel")
	assert.Equal(t, "new-model", mock.GetModel(), "should update underlying mock")
}

// TestTimeoutMiddleware_PreservesContextValues tests that the timeout middleware
// preserves context values across the request.
func TestTimeoutMiddleware_PreservesContextValues(t *testing.T) {
	mock := NewMockCoreLLM()
	timeout := 100 * time.Millisecond
	middleware := TimeoutMiddleware(timeout)
	wrapped := middleware(mock)

	ctx := context.WithValue(context.Background(), testContextKey, "test-value")
	opts := map[string]any{"temperature": 0.7}
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", opts)

	require.NoError(t, err, "request should succeed")
	assert.Equal(t, "test prompt", mock.LastPrompt, "prompt should be preserved")
	assert.Equal(t, opts, mock.LastOpts, "options should be preserved")
	assert.Equal(t, "test-value", mock.LastContext.Value(testContextKey),
		"context value should be preserved")
}

// TestTimeoutMiddleware_HandlesContextCancellation tests that the timeout middleware
// correctly handles context cancellation.
func TestTimeoutMiddleware_HandlesContextCancellation(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.ResponseDelay = 200 * time.Millisecond
	timeout := 300 * time.Millisecond
	middleware := TimeoutMiddleware(timeout)
	wrapped := middleware(mock)

	ctx, cancel := context.WithCancel(context.Background())

	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
	duration := time.Since(start)

	require.Error(t, err, "request should be cancelled")
	assert.True(t, errors.Is(err, context.Canceled),
		"error should be context cancelled: %v", err)

	assert.Greater(t, duration, 40*time.Millisecond, "should wait for cancellation")
	assert.Less(t, duration, 100*time.Millisecond, "should be cancelled quickly")
}

// TestTimeoutMiddleware_ZeroTimeout tests the behavior of the timeout middleware
// when the timeout is set to zero.
func TestTimeoutMiddleware_ZeroTimeout(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.ResponseDelay = 10 * time.Millisecond
	timeout := 0 * time.Millisecond
	middleware := TimeoutMiddleware(timeout)
	wrapped := middleware(mock)

	ctx := context.Background()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.Error(t, err, "request should timeout immediately")
	assert.True(t, errors.Is(err, context.DeadlineExceeded),
		"error should be deadline exceeded: %v", err)
}

// TestTimeoutMiddleware_VeryLongTimeout tests that the timeout middleware does
// not unnecessarily delay a request when a very long timeout is set.
func TestTimeoutMiddleware_VeryLongTimeout(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.ResponseDelay = 10 * time.Millisecond
	timeout := 10 * time.Second
	middleware := TimeoutMiddleware(timeout)
	wrapped := middleware(mock)

	ctx := context.Background()
	start := time.Now()
	response, tokensIn, tokensOut, err := wrapped.DoRequest(ctx, "test prompt", nil)
	duration := time.Since(start)

	require.NoError(t, err, "request should succeed")
	assert.Equal(t, "test response", response, "response should match")
	assert.Equal(t, 10, tokensIn, "input tokens should match")
	assert.Equal(t, 20, tokensOut, "output tokens should match")

	assert.Less(t, duration, 100*time.Millisecond, "should not wait for long timeout")
}

// TestTimeoutMiddleware_MultipleSimultaneousRequests tests that the timeout
// middleware correctly handles multiple simultaneous requests.
func TestTimeoutMiddleware_MultipleSimultaneousRequests(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.ResponseDelay = 10 * time.Millisecond
	timeout := 200 * time.Millisecond
	middleware := TimeoutMiddleware(timeout)
	wrapped := middleware(mock)

	const numRequests = 3
	errors := make(chan error, numRequests)

	for i := range numRequests {
		go func(i int) {
			ctx := context.Background()
			_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
			errors <- err
		}(i)
	}

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
