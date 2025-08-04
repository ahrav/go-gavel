package llm

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestRetryMiddleware_SuccessOnFirstAttempt tests that the retry middleware does
// not interfere with a successful request.
func TestRetryMiddleware_SuccessOnFirstAttempt(t *testing.T) {
	mock := NewMockCoreLLM()
	middleware := RetryMiddleware(3, 100*time.Millisecond, 1*time.Second)
	wrapped := middleware(mock)

	ctx := context.Background()
	response, tokensIn, tokensOut, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.NoError(t, err, "request should succeed")
	assert.Equal(t, "test response", response, "response should match")
	assert.Equal(t, 10, tokensIn, "input tokens should match")
	assert.Equal(t, 20, tokensOut, "output tokens should match")
	assert.Equal(t, 1, mock.GetCallCount(), "should only call once on success")
}

// TestRetryMiddleware_RetriesOnTransientError tests that the retry middleware
// successfully retries a request that initially fails with a transient error.
func TestRetryMiddleware_RetriesOnTransientError(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.FailUntilAttempt = 2
	middleware := RetryMiddleware(3, 10*time.Millisecond, 1*time.Second)
	wrapped := middleware(mock)

	ctx := context.Background()
	response, tokensIn, tokensOut, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.NoError(t, err, "request should eventually succeed")
	assert.Equal(t, "test response", response, "response should match")
	assert.Equal(t, 10, tokensIn, "input tokens should match")
	assert.Equal(t, 20, tokensOut, "output tokens should match")
	assert.Equal(t, 3, mock.GetCallCount(), "should retry until success")
}

// TestRetryMiddleware_FailsAfterMaxRetries tests that the retry middleware
// gives up after the maximum number of retries has been exhausted.
func TestRetryMiddleware_FailsAfterMaxRetries(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.Error = errors.New("persistent error")
	middleware := RetryMiddleware(2, 10*time.Millisecond, 1*time.Second)
	wrapped := middleware(mock)

	ctx := context.Background()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.Error(t, err, "request should fail")
	assert.Contains(t, err.Error(), "request failed after 3 attempts", "error should indicate retry exhaustion")
	assert.Contains(t, err.Error(), "persistent error", "error should contain original error")
	assert.Equal(t, 3, mock.GetCallCount(), "should attempt max retries + 1")
}

// TestRetryMiddleware_DoesNotRetryOnCircuitOpen tests that the retry middleware
// does not attempt to retry a request if the circuit breaker is open.
func TestRetryMiddleware_DoesNotRetryOnCircuitOpen(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.Error = ErrCircuitOpen
	middleware := RetryMiddleware(3, 10*time.Millisecond, 1*time.Second)
	wrapped := middleware(mock)

	ctx := context.Background()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.Error(t, err, "request should fail")
	assert.Contains(t, err.Error(), "circuit breaker is open", "should contain circuit open error")
	assert.LessOrEqual(t, mock.GetCallCount(), 2, "should not retry on circuit open")
}

// TestRetryMiddleware_RespectsContextCancellation tests that the retry middleware
// respects context cancellation and stops retrying if the context is canceled.
func TestRetryMiddleware_RespectsContextCancellation(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.Error = errors.New("slow error")
	mock.ResponseDelay = 50 * time.Millisecond
	middleware := RetryMiddleware(5, 10*time.Millisecond, 1*time.Second)
	wrapped := middleware(mock)

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.Error(t, err, "request should fail")
	assert.True(t, errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled),
		"error should be context related: %v", err)
	assert.Less(t, mock.GetCallCount(), 5, "should stop retrying on context cancellation")
}

// TestRetryMiddleware_ExponentialBackoff tests that the retry middleware uses
// an exponential backoff strategy between retries.
func TestRetryMiddleware_ExponentialBackoff(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.FailUntilAttempt = 3
	baseDelay := 10 * time.Millisecond
	middleware := RetryMiddleware(5, baseDelay, 1*time.Second)
	wrapped := middleware(mock)

	ctx := context.Background()
	start := time.Now()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
	duration := time.Since(start)

	require.NoError(t, err, "request should eventually succeed")
	assert.Equal(t, 4, mock.GetCallCount(), "should make expected number of attempts")

	delay1 := mock.GetTimeBetweenCalls(0, 1)
	delay2 := mock.GetTimeBetweenCalls(1, 2)
	delay3 := mock.GetTimeBetweenCalls(2, 3)

	require.NotNil(t, delay1, "should have delay between first retry")
	require.NotNil(t, delay2, "should have delay between second retry")
	require.NotNil(t, delay3, "should have delay between third retry")

	assert.Greater(t, delay2.Milliseconds(), delay1.Milliseconds()/2,
		"second delay should be larger than half of first delay (accounting for jitter)")
	assert.Greater(t, delay3.Milliseconds(), delay2.Milliseconds()/2,
		"third delay should be larger than half of second delay (accounting for jitter)")

	assert.Less(t, duration, 500*time.Millisecond, "total duration should be reasonable")
}

// TestRetryMiddleware_RespectsMaxDelay tests that the retry middleware respects
// the maximum delay configured for backoff.
func TestRetryMiddleware_RespectsMaxDelay(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.FailUntilAttempt = 10
	maxDelay := 20 * time.Millisecond
	middleware := RetryMiddleware(15, 5*time.Millisecond, maxDelay)
	wrapped := middleware(mock)

	ctx := context.Background()
	start := time.Now()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
	duration := time.Since(start)

	require.NoError(t, err, "request should eventually succeed")

	assert.Less(t, duration, 300*time.Millisecond, "delays should be capped by max delay")
}

// TestRetryMiddleware_PassesThroughModelMethods tests that the retry middleware
// correctly passes through calls to the underlying CoreLLM's methods.
func TestRetryMiddleware_PassesThroughModelMethods(t *testing.T) {
	mock := NewMockCoreLLM()
	middleware := RetryMiddleware(3, 10*time.Millisecond, 1*time.Second)
	wrapped := middleware(mock)

	assert.Equal(t, "test-model", wrapped.GetModel(), "should pass through GetModel")

	wrapped.SetModel("new-model")
	assert.Equal(t, "new-model", wrapped.GetModel(), "should pass through SetModel")
	assert.Equal(t, "new-model", mock.GetModel(), "should update underlying mock")
}

// TestRetryMiddleware_PreservesOptionsAndContext tests that the retry middleware
// preserves the context and options across retries.
func TestRetryMiddleware_PreservesOptionsAndContext(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.FailUntilAttempt = 1
	middleware := RetryMiddleware(3, 10*time.Millisecond, 1*time.Second)
	wrapped := middleware(mock)

	ctx := context.WithValue(context.Background(), testContextKey, "test-value")
	opts := map[string]any{"temperature": 0.7, "max_tokens": 100}
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", opts)

	require.NoError(t, err, "request should succeed")
	assert.Equal(t, "test prompt", mock.LastPrompt, "prompt should be preserved")
	assert.Equal(t, opts, mock.LastOpts, "options should be preserved")

	for i, capturedCtx := range mock.Contexts {
		assert.Equal(t, "test-value", capturedCtx.Value(testContextKey),
			"context value should be preserved on attempt %d", i+1)
	}
}

// TestRetryMiddleware_CalculateDelayEdgeCases tests the delay calculation logic
// for edge cases, such as negative or zero attempt numbers.
func TestRetryMiddleware_CalculateDelayEdgeCases(t *testing.T) {
	r := &retryLLM{
		baseDelay: 10 * time.Millisecond,
		maxDelay:  1 * time.Second,
	}

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
