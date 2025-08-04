package llm

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/time/rate"
)

// TestRateLimitMiddleware_AllowsRequestsWithinLimit tests that the rate limit middleware
// allows requests to pass through when they are within the defined rate limit.
func TestRateLimitMiddleware_AllowsRequestsWithinLimit(t *testing.T) {
	mock := NewMockCoreLLM()
	middleware := RateLimitMiddleware(rate.Limit(10), 1)
	wrapped := middleware(mock)

	ctx := context.Background()
	response, tokensIn, tokensOut, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.NoError(t, err, "request should succeed within rate limit")
	assert.Equal(t, "test response", response, "response should match")
	assert.Equal(t, 10, tokensIn, "input tokens should match")
	assert.Equal(t, 20, tokensOut, "output tokens should match")
	assert.Equal(t, 1, mock.GetCallCount(), "should call underlying implementation once")
}

// TestRateLimitMiddleware_DelaysRequestsExceedingRate tests that the rate limit middleware
// delays requests that exceed the defined rate limit.
func TestRateLimitMiddleware_DelaysRequestsExceedingRate(t *testing.T) {
	mock := NewMockCoreLLM()
	middleware := RateLimitMiddleware(rate.Limit(2), 1)
	wrapped := middleware(mock)

	ctx := context.Background()

	start := time.Now()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt 1", nil)
	firstDuration := time.Since(start)
	require.NoError(t, err, "first request should succeed immediately")
	assert.Less(t, firstDuration, 50*time.Millisecond, "first request should be immediate")

	start = time.Now()
	_, _, _, err = wrapped.DoRequest(ctx, "test prompt 2", nil)
	secondDuration := time.Since(start)
	require.NoError(t, err, "second request should succeed after delay")
	assert.Greater(t, secondDuration, 400*time.Millisecond, "second request should be delayed")
	assert.Less(t, secondDuration, 600*time.Millisecond, "delay should be reasonable")

	assert.Equal(t, 2, mock.GetCallCount(), "should call underlying implementation twice")
}

// TestRateLimitMiddleware_RespectsBurstLimit tests that the rate limit middleware
// correctly handles burst capacity, allowing a certain number of requests to
// exceed the rate limit before enforcing delays.
func TestRateLimitMiddleware_RespectsBurstLimit(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.ResponseDelay = 10 * time.Millisecond
	middleware := RateLimitMiddleware(rate.Limit(1), 3)
	wrapped := middleware(mock)

	ctx := context.Background()
	var durations []time.Duration

	for i := range 3 {
		start := time.Now()
		_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
		duration := time.Since(start)
		durations = append(durations, duration)
		require.NoError(t, err, "burst request %d should succeed", i+1)
	}

	for i, duration := range durations {
		assert.Less(t, duration, 100*time.Millisecond,
			"burst request %d should succeed quickly: %v", i+1, duration)
	}

	start := time.Now()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
	fourthDuration := time.Since(start)
	require.NoError(t, err, "fourth request should succeed after delay")
	assert.Greater(t, fourthDuration, 800*time.Millisecond, "fourth request should be delayed")

	assert.Equal(t, 4, mock.GetCallCount(), "should call underlying implementation 4 times")
}

// TestRateLimitMiddleware_RespectsContextCancellation tests that the rate limit
// middleware respects context cancellation and stops waiting for the rate limiter
// if the context is canceled.
func TestRateLimitMiddleware_RespectsContextCancellation(t *testing.T) {
	mock := NewMockCoreLLM()
	middleware := RateLimitMiddleware(rate.Limit(0.1), 1)
	wrapped := middleware(mock)

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	_, _, _, err := wrapped.DoRequest(context.Background(), "first", nil)
	require.NoError(t, err, "first request should succeed")

	_, _, _, err = wrapped.DoRequest(ctx, "second", nil)

	require.Error(t, err, "request should be cancelled")
	assert.True(t, errors.Is(err, context.DeadlineExceeded) || strings.Contains(err.Error(), "rate limit"),
		"error should be context or rate limit related: %v", err)

	assert.Equal(t, 1, mock.GetCallCount(), "should not call underlying implementation on cancelled request")
}

// TestRateLimitMiddleware_HandlesConcurrentRequests tests that the rate limit
// middleware correctly handles concurrent requests, ensuring that the overall
// rate limit is respected.
func TestRateLimitMiddleware_HandlesConcurrentRequests(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.ResponseDelay = 10 * time.Millisecond
	middleware := RateLimitMiddleware(rate.Limit(5), 2)
	wrapped := middleware(mock)

	const numGoroutines = 10
	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines)
	durations := make(chan time.Duration, numGoroutines)

	for i := range numGoroutines {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ctx := context.Background()
			start := time.Now()
			_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
			duration := time.Since(start)
			errors <- err
			durations <- duration
		}(i)
	}

	wg.Wait()
	close(errors)
	close(durations)

	var successCount int
	for err := range errors {
		if err == nil {
			successCount++
		} else {
			t.Errorf("unexpected error: %v", err)
		}
	}
	assert.Equal(t, numGoroutines, successCount, "all requests should succeed")

	var fastRequests, slowRequests int
	for duration := range durations {
		if duration < 100*time.Millisecond {
			fastRequests++
		} else {
			slowRequests++
		}
	}

	assert.Greater(t, slowRequests, 0, "some requests should be rate limited")
	assert.Equal(t, numGoroutines, mock.GetCallCount(), "should call underlying implementation for all requests")
}

// TestRateLimitMiddleware_PassesThroughModelMethods tests that the rate limit
// middleware correctly passes through calls to the underlying CoreLLM's methods.
func TestRateLimitMiddleware_PassesThroughModelMethods(t *testing.T) {
	mock := NewMockCoreLLM()
	middleware := RateLimitMiddleware(rate.Limit(10), 1)
	wrapped := middleware(mock)

	assert.Equal(t, "test-model", wrapped.GetModel(), "should pass through GetModel")

	wrapped.SetModel("new-model")
	assert.Equal(t, "new-model", wrapped.GetModel(), "should pass through SetModel")
	assert.Equal(t, "new-model", mock.GetModel(), "should update underlying mock")
}

// TestRateLimitMiddleware_PreservesContextAndOptions tests that the rate limit
// middleware preserves the context and options passed to the DoRequest method.
func TestRateLimitMiddleware_PreservesContextAndOptions(t *testing.T) {
	mock := NewMockCoreLLM()
	middleware := RateLimitMiddleware(rate.Limit(10), 1)
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

// TestRateLimitMiddleware_HandlesUnderlyingErrors tests that the rate limit
// middleware correctly propagates errors from the underlying CoreLLM.
func TestRateLimitMiddleware_HandlesUnderlyingErrors(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.Error = errors.New("underlying error")
	middleware := RateLimitMiddleware(rate.Limit(10), 1)
	wrapped := middleware(mock)

	ctx := context.Background()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.Error(t, err, "request should fail")
	assert.Equal(t, "underlying error", err.Error(), "should return underlying error")
	assert.Equal(t, 1, mock.GetCallCount(), "should call underlying implementation once")
}

// TestRateLimitMiddleware_ZeroRateLimit tests the behavior of the rate limit
// middleware when the rate limit is set to zero.
func TestRateLimitMiddleware_ZeroRateLimit(t *testing.T) {
	mock := NewMockCoreLLM()
	middleware := RateLimitMiddleware(rate.Limit(0), 0)
	wrapped := middleware(mock)

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.Error(t, err, "request should fail")
	assert.Contains(t, err.Error(), "rate limit", "should contain rate limit error")
	assert.Equal(t, 0, mock.GetCallCount(), "should not call underlying implementation")
}

// TestRateLimitMiddleware_HighBurstWithLowRate tests the behavior of the rate
// limit middleware with a high burst allowance but a low sustained rate.
func TestRateLimitMiddleware_HighBurstWithLowRate(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.ResponseDelay = 5 * time.Millisecond
	middleware := RateLimitMiddleware(rate.Limit(1), 10)
	wrapped := middleware(mock)

	ctx := context.Background()
	var fastRequests int

	for i := range 10 {
		start := time.Now()
		_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
		duration := time.Since(start)
		require.NoError(t, err, "burst request %d should succeed", i+1)

		if duration < 50*time.Millisecond {
			fastRequests++
		}
	}

	assert.Equal(t, 10, fastRequests, "all burst requests should be fast")
	assert.Equal(t, 10, mock.GetCallCount(), "should call underlying implementation 10 times")

	start := time.Now()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
	duration := time.Since(start)
	require.NoError(t, err, "additional request should succeed after delay")
	assert.Greater(t, duration, 900*time.Millisecond, "additional request should be delayed")
}
