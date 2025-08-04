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

func TestRateLimitMiddleware_AllowsRequestsWithinLimit(t *testing.T) {
	// Given a rate limiter that allows 10 requests per second
	mock := NewMockCoreLLM()
	middleware := RateLimitMiddleware(rate.Limit(10), 1)
	wrapped := middleware(mock)

	// When making a single request
	ctx := context.Background()
	response, tokensIn, tokensOut, err := wrapped.DoRequest(ctx, "test prompt", nil)

	// Then it should succeed immediately
	require.NoError(t, err, "request should succeed within rate limit")
	assert.Equal(t, "test response", response, "response should match")
	assert.Equal(t, 10, tokensIn, "input tokens should match")
	assert.Equal(t, 20, tokensOut, "output tokens should match")
	assert.Equal(t, 1, mock.GetCallCount(), "should call underlying implementation once")
}

func TestRateLimitMiddleware_DelaysRequestsExceedingRate(t *testing.T) {
	// Given a rate limiter that allows 2 requests per second with burst of 1
	mock := NewMockCoreLLM()
	middleware := RateLimitMiddleware(rate.Limit(2), 1)
	wrapped := middleware(mock)

	// When making multiple requests quickly
	ctx := context.Background()

	// First request should succeed immediately
	start := time.Now()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt 1", nil)
	firstDuration := time.Since(start)
	require.NoError(t, err, "first request should succeed immediately")
	assert.Less(t, firstDuration, 50*time.Millisecond, "first request should be immediate")

	// Second request should be delayed due to rate limiting
	start = time.Now()
	_, _, _, err = wrapped.DoRequest(ctx, "test prompt 2", nil)
	secondDuration := time.Since(start)
	require.NoError(t, err, "second request should succeed after delay")
	assert.Greater(t, secondDuration, 400*time.Millisecond, "second request should be delayed")
	assert.Less(t, secondDuration, 600*time.Millisecond, "delay should be reasonable")

	assert.Equal(t, 2, mock.GetCallCount(), "should call underlying implementation twice")
}

func TestRateLimitMiddleware_RespectsBurstLimit(t *testing.T) {
	// Given a rate limiter with burst capacity
	mock := NewMockCoreLLM()
	mock.ResponseDelay = 10 * time.Millisecond
	middleware := RateLimitMiddleware(rate.Limit(1), 3) // 1 per second, burst of 3
	wrapped := middleware(mock)

	// When making burst requests
	ctx := context.Background()
	var durations []time.Duration

	for i := range 3 {
		start := time.Now()
		_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
		duration := time.Since(start)
		durations = append(durations, duration)
		require.NoError(t, err, "burst request %d should succeed", i+1)
	}

	// Then first 3 requests should succeed quickly (within burst)
	for i, duration := range durations {
		assert.Less(t, duration, 100*time.Millisecond,
			"burst request %d should succeed quickly: %v", i+1, duration)
	}

	// Fourth request should be delayed
	start := time.Now()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
	fourthDuration := time.Since(start)
	require.NoError(t, err, "fourth request should succeed after delay")
	assert.Greater(t, fourthDuration, 800*time.Millisecond, "fourth request should be delayed")

	assert.Equal(t, 4, mock.GetCallCount(), "should call underlying implementation 4 times")
}

func TestRateLimitMiddleware_RespectsContextCancellation(t *testing.T) {
	// Given a very restrictive rate limiter
	mock := NewMockCoreLLM()
	middleware := RateLimitMiddleware(rate.Limit(0.1), 1) // Very slow: 1 per 10 seconds
	wrapped := middleware(mock)

	// When making a request with short timeout
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	// First request consumes the token
	_, _, _, err := wrapped.DoRequest(context.Background(), "first", nil)
	require.NoError(t, err, "first request should succeed")

	// Second request should be cancelled due to context timeout
	_, _, _, err = wrapped.DoRequest(ctx, "second", nil)

	require.Error(t, err, "request should be cancelled")
	assert.True(t, errors.Is(err, context.DeadlineExceeded) || strings.Contains(err.Error(), "rate limit"),
		"error should be context or rate limit related: %v", err)

	// Mock should only be called once (for the successful request)
	assert.Equal(t, 1, mock.GetCallCount(), "should not call underlying implementation on cancelled request")
}

func TestRateLimitMiddleware_HandlesConcurrentRequests(t *testing.T) {
	// Given a rate limiter with limited throughput
	mock := NewMockCoreLLM()
	mock.ResponseDelay = 10 * time.Millisecond
	middleware := RateLimitMiddleware(rate.Limit(5), 2) // 5 per second, burst of 2
	wrapped := middleware(mock)

	// When making concurrent requests
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

	// Then all requests should eventually succeed
	var successCount int
	for err := range errors {
		if err == nil {
			successCount++
		} else {
			t.Errorf("unexpected error: %v", err)
		}
	}
	assert.Equal(t, numGoroutines, successCount, "all requests should succeed")

	// Some requests should be delayed due to rate limiting
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

func TestRateLimitMiddleware_PassesThroughModelMethods(t *testing.T) {
	// Given a wrapped mock
	mock := NewMockCoreLLM()
	middleware := RateLimitMiddleware(rate.Limit(10), 1)
	wrapped := middleware(mock)

	// When calling model methods
	assert.Equal(t, "test-model", wrapped.GetModel(), "should pass through GetModel")

	wrapped.SetModel("new-model")
	assert.Equal(t, "new-model", wrapped.GetModel(), "should pass through SetModel")
	assert.Equal(t, "new-model", mock.GetModel(), "should update underlying mock")
}

func TestRateLimitMiddleware_PreservesContextAndOptions(t *testing.T) {
	// Given a rate-limited mock
	mock := NewMockCoreLLM()
	middleware := RateLimitMiddleware(rate.Limit(10), 1)
	wrapped := middleware(mock)

	// When making a request with context and options
	ctx := context.WithValue(context.Background(), testContextKey, "test-value")
	opts := map[string]any{"temperature": 0.7, "max_tokens": 100}
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", opts)

	// Then context and options should be preserved
	require.NoError(t, err, "request should succeed")
	assert.Equal(t, "test prompt", mock.LastPrompt, "prompt should be preserved")
	assert.Equal(t, opts, mock.LastOpts, "options should be preserved")
	assert.Equal(t, "test-value", mock.LastContext.Value(testContextKey),
		"context value should be preserved")
}

func TestRateLimitMiddleware_HandlesUnderlyingErrors(t *testing.T) {
	// Given a mock that fails
	mock := NewMockCoreLLM()
	mock.Error = errors.New("underlying error")
	middleware := RateLimitMiddleware(rate.Limit(10), 1)
	wrapped := middleware(mock)

	// When making a request
	ctx := context.Background()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	// Then it should return the underlying error
	require.Error(t, err, "request should fail")
	assert.Equal(t, "underlying error", err.Error(), "should return underlying error")
	assert.Equal(t, 1, mock.GetCallCount(), "should call underlying implementation once")
}

func TestRateLimitMiddleware_ZeroRateLimit(t *testing.T) {
	// Given a rate limiter with zero rate (no requests allowed)
	mock := NewMockCoreLLM()
	middleware := RateLimitMiddleware(rate.Limit(0), 0)
	wrapped := middleware(mock)

	// When making a request with short timeout
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	// Then it should fail due to rate limiting
	require.Error(t, err, "request should fail")
	assert.Contains(t, err.Error(), "rate limit", "should contain rate limit error")
	assert.Equal(t, 0, mock.GetCallCount(), "should not call underlying implementation")
}

func TestRateLimitMiddleware_HighBurstWithLowRate(t *testing.T) {
	// Given a rate limiter with high burst but low sustained rate
	mock := NewMockCoreLLM()
	mock.ResponseDelay = 5 * time.Millisecond
	middleware := RateLimitMiddleware(rate.Limit(1), 10) // 1 per second, burst of 10
	wrapped := middleware(mock)

	// When making burst requests
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

	// Then all burst requests should succeed quickly
	assert.Equal(t, 10, fastRequests, "all burst requests should be fast")
	assert.Equal(t, 10, mock.GetCallCount(), "should call underlying implementation 10 times")

	// Additional request should be delayed
	start := time.Now()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
	duration := time.Since(start)
	require.NoError(t, err, "additional request should succeed after delay")
	assert.Greater(t, duration, 900*time.Millisecond, "additional request should be delayed")
}
