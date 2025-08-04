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

func TestCircuitBreakerMiddleware_AllowsRequestsWhenClosed(t *testing.T) {
	// Given a circuit breaker that allows 3 failures
	mock := NewMockCoreLLM()
	middleware := CircuitBreakerMiddleware(3, 100*time.Millisecond)
	wrapped := middleware(mock)

	// When making a successful request
	ctx := context.Background()
	response, tokensIn, tokensOut, err := wrapped.DoRequest(ctx, "test prompt", nil)

	// Then it should succeed
	require.NoError(t, err, "request should succeed when circuit is closed")
	assert.Equal(t, "test response", response, "response should match")
	assert.Equal(t, 10, tokensIn, "input tokens should match")
	assert.Equal(t, 20, tokensOut, "output tokens should match")
	assert.Equal(t, 1, mock.GetCallCount(), "should call underlying implementation once")
}

func TestCircuitBreakerMiddleware_OpensAfterMaxFailures(t *testing.T) {
	// Given a circuit breaker that allows 2 failures and a mock that always fails
	mock := NewMockCoreLLM()
	mock.Error = errors.New("service error")
	middleware := CircuitBreakerMiddleware(2, 100*time.Millisecond)
	wrapped := middleware(mock)

	ctx := context.Background()

	// When making failing requests up to the threshold
	_, _, _, err1 := wrapped.DoRequest(ctx, "test 1", nil)
	_, _, _, err2 := wrapped.DoRequest(ctx, "test 2", nil)

	// Then both should fail with the original error
	require.Error(t, err1, "first request should fail")
	require.Error(t, err2, "second request should fail")
	assert.Equal(t, "service error", err1.Error(), "should return original error")
	assert.Equal(t, "service error", err2.Error(), "should return original error")

	// When making the third request (exceeds threshold)
	_, _, _, err3 := wrapped.DoRequest(ctx, "test 3", nil)

	// Then it should open the circuit and return circuit open error
	require.Error(t, err3, "third request should fail")
	assert.Equal(t, ErrCircuitOpen, err3, "should return circuit open error")
	assert.Equal(t, 2, mock.GetCallCount(), "should not call underlying implementation when circuit is open")
}

func TestCircuitBreakerMiddleware_RemainsOpenDuringCooldown(t *testing.T) {
	// Given a circuit breaker with short cooldown and a mock that always fails
	mock := NewMockCoreLLM()
	mock.Error = errors.New("service error")
	cooldown := 100 * time.Millisecond
	middleware := CircuitBreakerMiddleware(1, cooldown)
	wrapped := middleware(mock)

	ctx := context.Background()

	// When triggering the circuit to open
	_, _, _, err1 := wrapped.DoRequest(ctx, "test 1", nil)
	require.Error(t, err1, "first request should fail")

	// When making requests during cooldown period
	_, _, _, err2 := wrapped.DoRequest(ctx, "test 2", nil)
	_, _, _, err3 := wrapped.DoRequest(ctx, "test 3", nil)

	// Then all requests during cooldown should fail with circuit open error
	assert.Equal(t, ErrCircuitOpen, err2, "should fail with circuit open during cooldown")
	assert.Equal(t, ErrCircuitOpen, err3, "should fail with circuit open during cooldown")
	assert.Equal(t, 1, mock.GetCallCount(), "should not call underlying implementation during cooldown")
}

func TestCircuitBreakerMiddleware_TransitionsToHalfOpenAfterCooldown(t *testing.T) {
	// Given a circuit breaker with short cooldown
	mock := NewMockCoreLLM()
	mock.Error = errors.New("service error")
	cooldown := 50 * time.Millisecond
	middleware := CircuitBreakerMiddleware(1, cooldown)
	wrapped := middleware(mock)

	ctx := context.Background()

	// When triggering the circuit to open
	_, _, _, err1 := wrapped.DoRequest(ctx, "test 1", nil)
	require.Error(t, err1, "first request should fail")

	// Wait for cooldown period to expire
	time.Sleep(cooldown + 10*time.Millisecond)

	// When making a request after cooldown (now mock succeeds)
	mock.Error = nil // Fix the service
	response, tokensIn, tokensOut, err2 := wrapped.DoRequest(ctx, "test 2", nil)

	// Then it should succeed and close the circuit
	require.NoError(t, err2, "request should succeed after cooldown")
	assert.Equal(t, "test response", response, "response should match")
	assert.Equal(t, 10, tokensIn, "input tokens should match")
	assert.Equal(t, 20, tokensOut, "output tokens should match")
	assert.Equal(t, 2, mock.GetCallCount(), "should call underlying implementation after cooldown")

	// Subsequent requests should continue to succeed
	_, _, _, err3 := wrapped.DoRequest(ctx, "test 3", nil)
	require.NoError(t, err3, "subsequent request should succeed")
	assert.Equal(t, 3, mock.GetCallCount(), "should continue calling underlying implementation")
}

func TestCircuitBreakerMiddleware_ReopensOnFailureInHalfOpen(t *testing.T) {
	// Given a circuit breaker with short cooldown
	mock := NewMockCoreLLM()
	mock.Error = errors.New("service error")
	cooldown := 50 * time.Millisecond
	middleware := CircuitBreakerMiddleware(1, cooldown)
	wrapped := middleware(mock)

	ctx := context.Background()

	// When triggering the circuit to open
	_, _, _, err1 := wrapped.DoRequest(ctx, "test 1", nil)
	require.Error(t, err1, "first request should fail")

	// Wait for cooldown period to expire
	time.Sleep(cooldown + 10*time.Millisecond)

	// When making a request after cooldown but service still fails
	_, _, _, err2 := wrapped.DoRequest(ctx, "test 2", nil)

	// Then it should fail and reopen the circuit
	require.Error(t, err2, "request should fail in half-open state")
	assert.Equal(t, "service error", err2.Error(), "should return original error")

	// Subsequent request should immediately fail with circuit open
	_, _, _, err3 := wrapped.DoRequest(ctx, "test 3", nil)
	require.Error(t, err3, "subsequent request should fail")
	assert.Equal(t, ErrCircuitOpen, err3, "should fail with circuit open error")
	assert.Equal(t, 2, mock.GetCallCount(), "should not call underlying implementation when circuit reopens")
}

func TestCircuitBreakerMiddleware_ResetsFailureCountOnSuccess(t *testing.T) {
	// Given a circuit breaker that allows 3 failures
	mock := NewMockCoreLLM()
	middleware := CircuitBreakerMiddleware(3, 100*time.Millisecond)
	wrapped := middleware(mock)

	ctx := context.Background()

	// When alternating between failures and successes
	mock.Error = errors.New("service error")
	_, _, _, err1 := wrapped.DoRequest(ctx, "test 1", nil)
	_, _, _, err2 := wrapped.DoRequest(ctx, "test 2", nil)
	require.Error(t, err1, "first request should fail")
	require.Error(t, err2, "second request should fail")

	// Succeed once to reset failure count
	mock.Error = nil
	_, _, _, err3 := wrapped.DoRequest(ctx, "test 3", nil)
	require.NoError(t, err3, "third request should succeed")

	// Continue failing - should take 3 more failures to open (reset count to 0 after success)
	mock.Error = errors.New("service error")
	_, _, _, err4 := wrapped.DoRequest(ctx, "test 4", nil)
	_, _, _, err5 := wrapped.DoRequest(ctx, "test 5", nil)

	// Then these should call underlying service and return original error
	require.Error(t, err4, "fourth request should fail")
	require.Error(t, err5, "fifth request should fail")
	assert.Equal(t, "service error", err4.Error(), "should return original error")
	assert.Equal(t, "service error", err5.Error(), "should return original error")

	// Third failure should open circuit
	_, _, _, err6 := wrapped.DoRequest(ctx, "test 6", nil)
	require.Error(t, err6, "sixth request should fail")
	assert.Equal(t, "service error", err6.Error(), "should still call underlying on 3rd failure")

	// Fourth request should now get circuit open error
	_, _, _, err7 := wrapped.DoRequest(ctx, "test 7", nil)
	require.Error(t, err7, "seventh request should fail")
	assert.Equal(t, ErrCircuitOpen, err7, "should get circuit open error after max failures reached")
	assert.Equal(t, 6, mock.GetCallCount(), "should call underlying until circuit opens")
}

func TestCircuitBreakerMiddleware_WithMetrics(t *testing.T) {
	// Given a circuit breaker with metrics
	mock := NewMockCoreLLM()
	metrics := newMockCircuitBreakerMetrics()
	middleware := CircuitBreakerMiddlewareWithMetrics(2, 50*time.Millisecond, metrics)
	wrapped := middleware(mock)

	ctx := context.Background()

	// When making a successful request
	_, _, _, err1 := wrapped.DoRequest(ctx, "test 1", nil)
	require.NoError(t, err1, "first request should succeed")

	// Then success should be recorded
	assert.Equal(t, 1, metrics.successes, "should record success")
	assert.Contains(t, metrics.states, StateClosed, "should record closed state")

	// When making failing requests
	mock.Error = errors.New("service error")
	_, _, _, err2 := wrapped.DoRequest(ctx, "test 2", nil)
	_, _, _, err3 := wrapped.DoRequest(ctx, "test 3", nil)
	require.Error(t, err2, "second request should fail")
	require.Error(t, err3, "third request should fail")

	// Then failures and trip should be recorded
	assert.Equal(t, 2, metrics.failures, "should record failures")
	assert.Contains(t, metrics.states, StateOpen, "should record open state")

	// When making request with circuit open
	_, _, _, err4 := wrapped.DoRequest(ctx, "test 4", nil)
	require.Error(t, err4, "fourth request should fail with circuit open")

	// Then trip should be recorded
	assert.Greater(t, metrics.trips, 0, "should record circuit trips")
}

func TestCircuitBreakerMiddleware_PassesThroughModelMethods(t *testing.T) {
	// Given a wrapped mock
	mock := NewMockCoreLLM()
	middleware := CircuitBreakerMiddleware(3, 100*time.Millisecond)
	wrapped := middleware(mock)

	// When calling model methods
	assert.Equal(t, "test-model", wrapped.GetModel(), "should pass through GetModel")

	wrapped.SetModel("new-model")
	assert.Equal(t, "new-model", wrapped.GetModel(), "should pass through SetModel")
	assert.Equal(t, "new-model", mock.GetModel(), "should update underlying mock")
}

func TestCircuitBreakerMiddleware_PreservesContextAndOptions(t *testing.T) {
	// Given a circuit breaker
	mock := NewMockCoreLLM()
	middleware := CircuitBreakerMiddleware(3, 100*time.Millisecond)
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

func TestCircuitBreakerMiddleware_ConcurrentRequests(t *testing.T) {
	// Given a circuit breaker and mock that alternates between success and failure
	mock := NewMockCoreLLM()
	mock.AlternateErrors = true
	middleware := CircuitBreakerMiddleware(10, 100*time.Millisecond) // High threshold
	wrapped := middleware(mock)

	// When making concurrent requests
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

	// Then some requests should succeed and some should fail
	successCount := len(successes)
	failureCount := len(failures)

	assert.Greater(t, successCount, 0, "some requests should succeed")
	assert.Greater(t, failureCount, 0, "some requests should fail")
	assert.Equal(t, numGoroutines, successCount+failureCount, "all requests should complete")
	assert.Equal(t, numGoroutines, mock.GetCallCount(), "all requests should reach underlying implementation")
}

func TestCircuitBreaker_GetState(t *testing.T) {
	// Given a circuit breaker
	cb := NewCircuitBreaker(2, 100*time.Millisecond)

	// When checking initial state
	state := cb.GetState()

	// Then it should be closed
	assert.Equal(t, StateClosed, state, "initial state should be closed")

	// When causing failures to open circuit
	cb.Call(func() error { return errors.New("error 1") })
	cb.Call(func() error { return errors.New("error 2") })

	// Then state should be open
	state = cb.GetState()
	assert.Equal(t, StateOpen, state, "state should be open after max failures")
}

func TestCircuitBreakerMiddleware_ZeroMaxFailures(t *testing.T) {
	// Given a circuit breaker that allows 0 failures
	mock := NewMockCoreLLM()
	mock.Error = errors.New("service error")
	middleware := CircuitBreakerMiddleware(0, 100*time.Millisecond)
	wrapped := middleware(mock)

	// When making a request
	ctx := context.Background()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	// Then it should call once and then open the circuit
	require.Error(t, err, "request should fail")
	assert.Equal(t, "service error", err.Error(), "should get original error on first call")
	assert.Equal(t, 1, mock.GetCallCount(), "should call underlying implementation once")

	// Second request should hit open circuit
	_, _, _, err2 := wrapped.DoRequest(ctx, "test prompt 2", nil)
	require.Error(t, err2, "second request should fail")
	assert.Equal(t, ErrCircuitOpen, err2, "should fail with circuit open")
	assert.Equal(t, 1, mock.GetCallCount(), "should not call underlying implementation again")
}

func TestCircuitBreakerMiddleware_VeryShortCooldown(t *testing.T) {
	// Given a circuit breaker with very short cooldown
	mock := NewMockCoreLLM()
	mock.Error = errors.New("service error")
	cooldown := 1 * time.Millisecond
	middleware := CircuitBreakerMiddleware(1, cooldown)
	wrapped := middleware(mock)

	ctx := context.Background()

	// When triggering circuit to open
	_, _, _, err1 := wrapped.DoRequest(ctx, "test 1", nil)
	require.Error(t, err1, "first request should fail")

	// Wait for very short cooldown
	time.Sleep(cooldown + 1*time.Millisecond)

	// Fix service and make request
	mock.Error = nil
	_, _, _, err2 := wrapped.DoRequest(ctx, "test 2", nil)

	// Then circuit should recover quickly
	require.NoError(t, err2, "request should succeed after short cooldown")
	assert.Equal(t, 2, mock.GetCallCount(), "should call underlying implementation after recovery")
}
