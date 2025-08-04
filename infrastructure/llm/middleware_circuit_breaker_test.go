package llm

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestCircuitBreakerMiddleware_AllowsRequestsWhenClosed tests that the circuit breaker
// allows requests to pass through when it is in the closed state.
func TestCircuitBreakerMiddleware_AllowsRequestsWhenClosed(t *testing.T) {
	mock := NewMockCoreLLM()
	middleware := CircuitBreakerMiddleware(3, 100*time.Millisecond)
	wrapped := middleware(mock)

	ctx := context.Background()
	response, tokensIn, tokensOut, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.NoError(t, err, "request should succeed when circuit is closed")
	assert.Equal(t, "test response", response, "response should match")
	assert.Equal(t, 10, tokensIn, "input tokens should match")
	assert.Equal(t, 20, tokensOut, "output tokens should match")
	assert.Equal(t, 1, mock.GetCallCount(), "should call underlying implementation once")
}

// TestCircuitBreakerMiddleware_OpensAfterMaxFailures tests that the circuit breaker
// opens after the configured maximum number of failures is reached.
func TestCircuitBreakerMiddleware_OpensAfterMaxFailures(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.Error = errors.New("service error")
	middleware := CircuitBreakerMiddleware(2, 100*time.Millisecond)
	wrapped := middleware(mock)

	ctx := context.Background()

	_, _, _, err1 := wrapped.DoRequest(ctx, "test 1", nil)
	_, _, _, err2 := wrapped.DoRequest(ctx, "test 2", nil)

	require.Error(t, err1, "first request should fail")
	require.Error(t, err2, "second request should fail")
	assert.Equal(t, "service error", err1.Error(), "should return original error")
	assert.Equal(t, "service error", err2.Error(), "should return original error")

	_, _, _, err3 := wrapped.DoRequest(ctx, "test 3", nil)

	require.Error(t, err3, "third request should fail")
	assert.Equal(t, ErrCircuitOpen, err3, "should return circuit open error")
	assert.Equal(t, 2, mock.GetCallCount(), "should not call underlying implementation when circuit is open")
}

// TestCircuitBreakerMiddleware_RemainsOpenDuringCooldown tests that the circuit breaker
// remains open and rejects requests during its cooldown period.
func TestCircuitBreakerMiddleware_RemainsOpenDuringCooldown(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.Error = errors.New("service error")
	cooldown := 100 * time.Millisecond
	middleware := CircuitBreakerMiddleware(1, cooldown)
	wrapped := middleware(mock)

	ctx := context.Background()

	_, _, _, err1 := wrapped.DoRequest(ctx, "test 1", nil)
	require.Error(t, err1, "first request should fail")

	_, _, _, err2 := wrapped.DoRequest(ctx, "test 2", nil)
	_, _, _, err3 := wrapped.DoRequest(ctx, "test 3", nil)

	assert.Equal(t, ErrCircuitOpen, err2, "should fail with circuit open during cooldown")
	assert.Equal(t, ErrCircuitOpen, err3, "should fail with circuit open during cooldown")
	assert.Equal(t, 1, mock.GetCallCount(), "should not call underlying implementation during cooldown")
}

// TestCircuitBreakerMiddleware_TransitionsToHalfOpenAfterCooldown tests that the circuit
// breaker transitions to the half-open state after the cooldown period expires.
func TestCircuitBreakerMiddleware_TransitionsToHalfOpenAfterCooldown(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.Error = errors.New("service error")
	cooldown := 50 * time.Millisecond
	middleware := CircuitBreakerMiddleware(1, cooldown)
	wrapped := middleware(mock)

	ctx := context.Background()

	_, _, _, err1 := wrapped.DoRequest(ctx, "test 1", nil)
	require.Error(t, err1, "first request should fail")

	time.Sleep(cooldown + 10*time.Millisecond)

	mock.Error = nil
	response, tokensIn, tokensOut, err2 := wrapped.DoRequest(ctx, "test 2", nil)

	require.NoError(t, err2, "request should succeed after cooldown")
	assert.Equal(t, "test response", response, "response should match")
	assert.Equal(t, 10, tokensIn, "input tokens should match")
	assert.Equal(t, 20, tokensOut, "output tokens should match")
	assert.Equal(t, 2, mock.GetCallCount(), "should call underlying implementation after cooldown")

	_, _, _, err3 := wrapped.DoRequest(ctx, "test 3", nil)
	require.NoError(t, err3, "subsequent request should succeed")
	assert.Equal(t, 3, mock.GetCallCount(), "should continue calling underlying implementation")
}

// TestCircuitBreakerMiddleware_ReopensOnFailureInHalfOpen tests that the circuit breaker
// re-opens if a request fails while it is in the half-open state.
func TestCircuitBreakerMiddleware_ReopensOnFailureInHalfOpen(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.Error = errors.New("service error")
	cooldown := 50 * time.Millisecond
	middleware := CircuitBreakerMiddleware(1, cooldown)
	wrapped := middleware(mock)

	ctx := context.Background()

	_, _, _, err1 := wrapped.DoRequest(ctx, "test 1", nil)
	require.Error(t, err1, "first request should fail")

	time.Sleep(cooldown + 10*time.Millisecond)

	_, _, _, err2 := wrapped.DoRequest(ctx, "test 2", nil)

	require.Error(t, err2, "request should fail in half-open state")
	assert.Equal(t, "service error", err2.Error(), "should return original error")

	_, _, _, err3 := wrapped.DoRequest(ctx, "test 3", nil)
	require.Error(t, err3, "subsequent request should fail")
	assert.Equal(t, ErrCircuitOpen, err3, "should fail with circuit open error")
	assert.Equal(t, 2, mock.GetCallCount(), "should not call underlying implementation when circuit reopens")
}

// TestCircuitBreakerMiddleware_ResetsFailureCountOnSuccess tests that the circuit breaker's
// failure count is reset after a successful request.
func TestCircuitBreakerMiddleware_ResetsFailureCountOnSuccess(t *testing.T) {
	mock := NewMockCoreLLM()
	middleware := CircuitBreakerMiddleware(3, 100*time.Millisecond)
	wrapped := middleware(mock)

	ctx := context.Background()

	mock.Error = errors.New("service error")
	_, _, _, err1 := wrapped.DoRequest(ctx, "test 1", nil)
	_, _, _, err2 := wrapped.DoRequest(ctx, "test 2", nil)
	require.Error(t, err1, "first request should fail")
	require.Error(t, err2, "second request should fail")

	mock.Error = nil
	_, _, _, err3 := wrapped.DoRequest(ctx, "test 3", nil)
	require.NoError(t, err3, "third request should succeed")

	mock.Error = errors.New("service error")
	_, _, _, err4 := wrapped.DoRequest(ctx, "test 4", nil)
	_, _, _, err5 := wrapped.DoRequest(ctx, "test 5", nil)

	require.Error(t, err4, "fourth request should fail")
	require.Error(t, err5, "fifth request should fail")
	assert.Equal(t, "service error", err4.Error(), "should return original error")
	assert.Equal(t, "service error", err5.Error(), "should return original error")

	_, _, _, err6 := wrapped.DoRequest(ctx, "test 6", nil)
	require.Error(t, err6, "sixth request should fail")
	assert.Equal(t, "service error", err6.Error(), "should still call underlying on 3rd failure")

	_, _, _, err7 := wrapped.DoRequest(ctx, "test 7", nil)
	require.Error(t, err7, "seventh request should fail")
	assert.Equal(t, ErrCircuitOpen, err7, "should get circuit open error after max failures reached")
	assert.Equal(t, 6, mock.GetCallCount(), "should call underlying until circuit opens")
}

// TestCircuitBreakerMiddleware_WithMetrics tests that the circuit breaker middleware
// correctly records metrics for its state changes and outcomes.
func TestCircuitBreakerMiddleware_WithMetrics(t *testing.T) {
	mock := NewMockCoreLLM()
	metrics := newMockCircuitBreakerMetrics()
	middleware := CircuitBreakerMiddlewareWithMetrics(2, 50*time.Millisecond, metrics)
	wrapped := middleware(mock)

	ctx := context.Background()

	_, _, _, err1 := wrapped.DoRequest(ctx, "test 1", nil)
	require.NoError(t, err1, "first request should succeed")

	assert.Equal(t, 1, metrics.successes, "should record success")
	assert.Contains(t, metrics.states, StateClosed, "should record closed state")

	mock.Error = errors.New("service error")
	_, _, _, err2 := wrapped.DoRequest(ctx, "test 2", nil)
	_, _, _, err3 := wrapped.DoRequest(ctx, "test 3", nil)
	require.Error(t, err2, "second request should fail")
	require.Error(t, err3, "third request should fail")

	assert.Equal(t, 2, metrics.failures, "should record failures")
	assert.Contains(t, metrics.states, StateOpen, "should record open state")

	_, _, _, err4 := wrapped.DoRequest(ctx, "test 4", nil)
	require.Error(t, err4, "fourth request should fail with circuit open")

	assert.Greater(t, metrics.trips, 0, "should record circuit trips")
}

// TestCircuitBreakerMiddleware_PassesThroughModelMethods tests that the middleware
// correctly passes through calls to the underlying CoreLLM's methods.
func TestCircuitBreakerMiddleware_PassesThroughModelMethods(t *testing.T) {
	mock := NewMockCoreLLM()
	middleware := CircuitBreakerMiddleware(3, 100*time.Millisecond)
	wrapped := middleware(mock)

	assert.Equal(t, "test-model", wrapped.GetModel(), "should pass through GetModel")

	wrapped.SetModel("new-model")
	assert.Equal(t, "new-model", wrapped.GetModel(), "should pass through SetModel")
	assert.Equal(t, "new-model", mock.GetModel(), "should update underlying mock")
}

// TestCircuitBreakerMiddleware_PreservesContextAndOptions tests that the middleware
// preserves the context and options passed to the DoRequest method.
func TestCircuitBreakerMiddleware_PreservesContextAndOptions(t *testing.T) {
	mock := NewMockCoreLLM()
	middleware := CircuitBreakerMiddleware(3, 100*time.Millisecond)
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

// TestCircuitBreakerMiddleware_ConcurrentRequests tests the behavior of the circuit
// breaker middleware under concurrent requests.
func TestCircuitBreakerMiddleware_ConcurrentRequests(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.AlternateErrors = true
	middleware := CircuitBreakerMiddleware(10, 100*time.Millisecond)
	wrapped := middleware(mock)

	const numGoroutines = 20
	var wg sync.WaitGroup
	successes := make(chan struct{}, numGoroutines)
	failures := make(chan struct{}, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ctx := context.Background()
			_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
			if err == nil {
				successes <- struct{}{}
			} else {
				failures <- struct{}{}
			}
		}(i)
	}

	wg.Wait()
	close(successes)
	close(failures)

	successCount := len(successes)
	failureCount := len(failures)

	assert.Greater(t, successCount, 0, "some requests should succeed")
	assert.Greater(t, failureCount, 0, "some requests should fail")
	assert.Equal(t, numGoroutines, successCount+failureCount, "all requests should complete")
	assert.Equal(t, numGoroutines, mock.GetCallCount(), "all requests should reach underlying implementation")
}

// TestCircuitBreaker_GetState tests the GetState method of the CircuitBreaker.
func TestCircuitBreaker_GetState(t *testing.T) {
	cb := NewCircuitBreaker(2, 100*time.Millisecond)

	state := cb.GetState()

	assert.Equal(t, StateClosed, state, "initial state should be closed")

	cb.Call(func() error { return errors.New("error 1") })
	cb.Call(func() error { return errors.New("error 2") })

	state = cb.GetState()
	assert.Equal(t, StateOpen, state, "state should be open after max failures")
}

// TestCircuitBreakerMiddleware_ZeroMaxFailures tests the circuit breaker's behavior
// when maxFailures is set to zero.
func TestCircuitBreakerMiddleware_ZeroMaxFailures(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.Error = errors.New("service error")
	middleware := CircuitBreakerMiddleware(0, 100*time.Millisecond)
	wrapped := middleware(mock)

	ctx := context.Background()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.Error(t, err, "request should fail")
	assert.Equal(t, "service error", err.Error(), "should get original error on first call")
	assert.Equal(t, 1, mock.GetCallCount(), "should call underlying implementation once")

	_, _, _, err2 := wrapped.DoRequest(ctx, "test prompt 2", nil)
	require.Error(t, err2, "second request should fail")
	assert.Equal(t, ErrCircuitOpen, err2, "should fail with circuit open")
	assert.Equal(t, 1, mock.GetCallCount(), "should not call underlying implementation again")
}

// TestCircuitBreakerMiddleware_VeryShortCooldown tests the circuit breaker's recovery
// with a very short cooldown period.
func TestCircuitBreakerMiddleware_VeryShortCooldown(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.Error = errors.New("service error")
	cooldown := 1 * time.Millisecond
	middleware := CircuitBreakerMiddleware(1, cooldown)
	wrapped := middleware(mock)

	ctx := context.Background()

	_, _, _, err1 := wrapped.DoRequest(ctx, "test 1", nil)
	require.Error(t, err1, "first request should fail")

	time.Sleep(cooldown + 1*time.Millisecond)

	mock.Error = nil
	_, _, _, err2 := wrapped.DoRequest(ctx, "test 2", nil)

	require.NoError(t, err2, "request should succeed after short cooldown")
	assert.Equal(t, 2, mock.GetCallCount(), "should call underlying implementation after recovery")
}
