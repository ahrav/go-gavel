package llm

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRetryMiddleware_SuccessOnFirstAttempt(t *testing.T) {
	// Given a mock that succeeds immediately
	mock := NewMockCoreLLM()
	middleware := RetryMiddleware(3, 100*time.Millisecond, 1*time.Second)
	wrapped := middleware(mock)

	// When making a request
	ctx := context.Background()
	response, tokensIn, tokensOut, err := wrapped.DoRequest(ctx, "test prompt", nil)

	// Then it should succeed without retries
	require.NoError(t, err, "request should succeed")
	assert.Equal(t, "test response", response, "response should match")
	assert.Equal(t, 10, tokensIn, "input tokens should match")
	assert.Equal(t, 20, tokensOut, "output tokens should match")
	assert.Equal(t, 1, mock.GetCallCount(), "should only call once on success")
}

func TestRetryMiddleware_RetriesOnTransientError(t *testing.T) {
	// Given a mock that fails twice then succeeds
	mock := NewMockCoreLLM()
	mock.FailUntilAttempt = 2
	middleware := RetryMiddleware(3, 10*time.Millisecond, 1*time.Second)
	wrapped := middleware(mock)

	// When making a request
	ctx := context.Background()
	response, tokensIn, tokensOut, err := wrapped.DoRequest(ctx, "test prompt", nil)

	// Then it should eventually succeed after retries
	require.NoError(t, err, "request should eventually succeed")
	assert.Equal(t, "test response", response, "response should match")
	assert.Equal(t, 10, tokensIn, "input tokens should match")
	assert.Equal(t, 20, tokensOut, "output tokens should match")
	assert.Equal(t, 3, mock.GetCallCount(), "should retry until success")
}

func TestRetryMiddleware_FailsAfterMaxRetries(t *testing.T) {
	// Given a mock that always fails
	mock := NewMockCoreLLM()
	mock.Error = errors.New("persistent error")
	middleware := RetryMiddleware(2, 10*time.Millisecond, 1*time.Second)
	wrapped := middleware(mock)

	// When making a request
	ctx := context.Background()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	// Then it should fail after exhausting retries
	require.Error(t, err, "request should fail")
	assert.Contains(t, err.Error(), "request failed after 3 attempts", "error should indicate retry exhaustion")
	assert.Contains(t, err.Error(), "persistent error", "error should contain original error")
	assert.Equal(t, 3, mock.GetCallCount(), "should attempt max retries + 1")
}

func TestRetryMiddleware_DoesNotRetryOnCircuitOpen(t *testing.T) {
	// Given a mock that returns circuit open error
	mock := NewMockCoreLLM()
	mock.Error = ErrCircuitOpen
	middleware := RetryMiddleware(3, 10*time.Millisecond, 1*time.Second)
	wrapped := middleware(mock)

	// When making a request
	ctx := context.Background()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	// Then it should fail without retries (but may try once before detecting circuit open)
	require.Error(t, err, "request should fail")
	assert.Contains(t, err.Error(), "circuit breaker is open", "should contain circuit open error")
	assert.LessOrEqual(t, mock.GetCallCount(), 2, "should not retry on circuit open")
}

func TestRetryMiddleware_RespectsContextCancellation(t *testing.T) {
	// Given a mock that always fails slowly
	mock := NewMockCoreLLM()
	mock.Error = errors.New("slow error")
	mock.ResponseDelay = 50 * time.Millisecond
	middleware := RetryMiddleware(5, 10*time.Millisecond, 1*time.Second)
	wrapped := middleware(mock)

	// When making a request with a short timeout
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	// Then it should fail with context error
	require.Error(t, err, "request should fail")
	assert.True(t, errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled),
		"error should be context related: %v", err)
	assert.Less(t, mock.GetCallCount(), 5, "should stop retrying on context cancellation")
}

func TestRetryMiddleware_ExponentialBackoff(t *testing.T) {
	// Given a mock that fails several times
	mock := NewMockCoreLLM()
	mock.FailUntilAttempt = 3
	baseDelay := 10 * time.Millisecond
	middleware := RetryMiddleware(5, baseDelay, 1*time.Second)
	wrapped := middleware(mock)

	// When making a request
	ctx := context.Background()
	start := time.Now()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
	duration := time.Since(start)

	// Then backoff delays should increase exponentially
	require.NoError(t, err, "request should eventually succeed")
	assert.Equal(t, 4, mock.GetCallCount(), "should make expected number of attempts")

	// Verify delays between calls increase
	delay1 := mock.GetTimeBetweenCalls(0, 1)
	delay2 := mock.GetTimeBetweenCalls(1, 2)
	delay3 := mock.GetTimeBetweenCalls(2, 3)

	require.NotNil(t, delay1, "should have delay between first retry")
	require.NotNil(t, delay2, "should have delay between second retry")
	require.NotNil(t, delay3, "should have delay between third retry")

	// Each delay should be larger than the previous (accounting for jitter)
	assert.Greater(t, delay2.Milliseconds(), delay1.Milliseconds()/2,
		"second delay should be larger than half of first delay (accounting for jitter)")
	assert.Greater(t, delay3.Milliseconds(), delay2.Milliseconds()/2,
		"third delay should be larger than half of second delay (accounting for jitter)")

	// Total duration should be reasonable
	assert.Less(t, duration, 500*time.Millisecond, "total duration should be reasonable")
}

func TestRetryMiddleware_RespectsMaxDelay(t *testing.T) {
	// Given a mock that fails many times with low max delay
	mock := NewMockCoreLLM()
	mock.FailUntilAttempt = 10
	maxDelay := 20 * time.Millisecond
	middleware := RetryMiddleware(15, 5*time.Millisecond, maxDelay)
	wrapped := middleware(mock)

	// When making a request
	ctx := context.Background()
	start := time.Now()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
	duration := time.Since(start)

	// Then delays should be capped at max delay
	require.NoError(t, err, "request should eventually succeed")

	// With 10 retries at max 20ms each (plus jitter), should be under 300ms
	assert.Less(t, duration, 300*time.Millisecond, "delays should be capped by max delay")
}

func TestRetryMiddleware_PassesThroughModelMethods(t *testing.T) {
	// Given a wrapped mock
	mock := NewMockCoreLLM()
	middleware := RetryMiddleware(3, 10*time.Millisecond, 1*time.Second)
	wrapped := middleware(mock)

	// When calling model methods
	assert.Equal(t, "test-model", wrapped.GetModel(), "should pass through GetModel")

	wrapped.SetModel("new-model")
	assert.Equal(t, "new-model", wrapped.GetModel(), "should pass through SetModel")
	assert.Equal(t, "new-model", mock.GetModel(), "should update underlying mock")
}

func TestRetryMiddleware_PreservesOptionsAndContext(t *testing.T) {
	// Given a mock that fails once
	mock := NewMockCoreLLM()
	mock.FailUntilAttempt = 1
	middleware := RetryMiddleware(3, 10*time.Millisecond, 1*time.Second)
	wrapped := middleware(mock)

	// When making a request with options
	ctx := context.WithValue(context.Background(), testContextKey, "test-value")
	opts := map[string]any{"temperature": 0.7, "max_tokens": 100}
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", opts)

	// Then context and options should be preserved across retries
	require.NoError(t, err, "request should succeed")
	assert.Equal(t, "test prompt", mock.LastPrompt, "prompt should be preserved")
	assert.Equal(t, opts, mock.LastOpts, "options should be preserved")

	// Verify context was passed through on all attempts
	for i, capturedCtx := range mock.Contexts {
		assert.Equal(t, "test-value", capturedCtx.Value(testContextKey),
			"context value should be preserved on attempt %d", i+1)
	}
}

func TestRetryMiddleware_CalculateDelayEdgeCases(t *testing.T) {
	// Given a retry middleware
	r := &retryLLM{
		baseDelay: 10 * time.Millisecond,
		maxDelay:  1 * time.Second,
	}

	// When calculating delay for various attempts
	tests := []struct {
		name    string
		attempt int
	}{
		{"negative attempt", -1},
		{"zero attempt", 0},
		{"normal attempt", 5},
		{"very large attempt", 50},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			delay := r.calculateDelay(tt.attempt)
			assert.Greater(t, delay, 0*time.Millisecond, "delay should be positive")
			assert.LessOrEqual(t, delay, r.maxDelay, "delay should not exceed max delay")
		})
	}
}
