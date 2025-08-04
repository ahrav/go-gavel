package llm

import (
	"context"
	"errors"
	"sync"
	"time"
)

// ErrCircuitOpen indicates that the circuit breaker rejected a request.
// This error is returned when the circuit is open and prevents
// requests from reaching the downstream service.
var ErrCircuitOpen = errors.New("circuit breaker is open")

// CircuitBreakerState represents the current state of a circuit breaker.
// The circuit breaker transitions between these states based on
// success and failure patterns to provide resilience.
type CircuitBreakerState int

// Circuit breaker states.
// These states control how the circuit breaker responds to requests
// and implements the circuit breaker pattern for resilience.
const (
	// StateClosed allows all requests to pass through normally.
	// This is the default state when the downstream service is healthy.
	StateClosed CircuitBreakerState = iota

	// StateOpen rejects all requests immediately to prevent cascading failures.
	// The circuit enters this state after too many consecutive failures.
	StateOpen

	// StateHalfOpen allows limited requests to test service recovery.
	// The circuit transitions to this state after the cooldown period expires.
	StateHalfOpen
)

// CircuitBreakerMetrics enables observability for circuit breaker behavior.
// Implementations can integrate with monitoring systems to track
// circuit breaker state changes, trips, and recovery patterns.
type CircuitBreakerMetrics interface {
	// RecordState updates the current circuit breaker state metric.
	RecordState(state CircuitBreakerState)

	// RecordTrip increments the circuit breaker trip counter.
	RecordTrip()

	// RecordSuccess increments the successful request counter.
	RecordSuccess()

	// RecordFailure increments the failed request counter.
	RecordFailure()
}

// CircuitBreaker implements the circuit breaker pattern for resilience.
// It tracks failure rates and automatically opens when failures exceed
// the threshold, then tests recovery through half-open states.
type CircuitBreaker struct {
	mu               sync.RWMutex
	state            CircuitBreakerState
	failureCount     int
	maxFailures      int
	cooldownDuration time.Duration
	lastFailure      time.Time
	metrics          CircuitBreakerMetrics
}

// NewCircuitBreaker creates a circuit breaker with the specified configuration.
// The circuit opens after maxFailures consecutive errors and stays open
// for cooldownDuration before testing recovery.
func NewCircuitBreaker(maxFailures int, cooldownDuration time.Duration) *CircuitBreaker {
	return &CircuitBreaker{
		state:            StateClosed,
		maxFailures:      maxFailures,
		cooldownDuration: cooldownDuration,
	}
}

// Call executes a function through the circuit breaker.
// If the circuit is open, this returns ErrCircuitOpen immediately.
// Otherwise, it executes the function and updates circuit state based on the result.
func (cb *CircuitBreaker) Call(fn func() error) error {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	switch cb.state {
	case StateOpen:
		if time.Since(cb.lastFailure) < cb.cooldownDuration {
			return ErrCircuitOpen
		}
		cb.state = StateHalfOpen
		fallthrough
	case StateHalfOpen:
		err := fn()
		if err != nil {
			cb.failureCount++
			cb.lastFailure = time.Now()
			cb.state = StateOpen
			return err
		}
		cb.failureCount = 0
		cb.state = StateClosed
		return nil
	case StateClosed:
		err := fn()
		if err != nil {
			cb.failureCount++
			cb.lastFailure = time.Now()
			if cb.failureCount >= cb.maxFailures {
				cb.state = StateOpen
			}
			return err
		}
		cb.failureCount = 0
		return nil
	}
	return nil
}

// GetState returns the current circuit breaker state.
// This is useful for monitoring and debugging circuit breaker behavior
// in operational environments.
func (cb *CircuitBreaker) GetState() CircuitBreakerState {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.state
}

// circuitBreakedLLM implements the circuit breaker pattern for resilience.
// When failures exceed the threshold, the circuit opens to prevent
// cascading failures and allows the downstream service to recover.
type circuitBreakedLLM struct {
	next    CoreLLM
	cb      *CircuitBreaker
	metrics CircuitBreakerMetrics
}

// CircuitBreakerMiddleware creates middleware that implements the circuit breaker pattern.
// The circuit opens after maxFailures consecutive errors and stays open
// for the cooldown duration before attempting recovery.
func CircuitBreakerMiddleware(maxFailures int, cooldown time.Duration) Middleware {
	return CircuitBreakerMiddlewareWithMetrics(maxFailures, cooldown, nil)
}

// CircuitBreakerMiddlewareWithMetrics creates circuit breaker middleware with metrics support.
// This allows monitoring of circuit breaker behavior in production systems.
func CircuitBreakerMiddlewareWithMetrics(maxFailures int, cooldown time.Duration, metrics CircuitBreakerMetrics) Middleware {
	cb := &CircuitBreaker{
		maxFailures:      maxFailures,
		cooldownDuration: cooldown,
		metrics:          metrics,
		state:            StateClosed,
	}

	return func(next CoreLLM) CoreLLM {
		return &circuitBreakedLLM{
			next:    next,
			cb:      cb,
			metrics: metrics,
		}
	}
}

// DoRequest executes the request through the circuit breaker.
// If the circuit is open, this fails immediately without calling
// the downstream service, providing fast failure response.
func (c *circuitBreakedLLM) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
	var response string
	var tokensIn, tokensOut int

	err := c.cb.Call(func() error {
		var err error
		response, tokensIn, tokensOut, err = c.next.DoRequest(ctx, prompt, opts)
		return err
	})

	if c.metrics != nil {
		switch err {
		case nil:
			c.metrics.RecordSuccess()
		case ErrCircuitOpen:
			c.metrics.RecordTrip()
		default:
			c.metrics.RecordFailure()
		}
		c.metrics.RecordState(c.cb.GetState())
	}

	return response, tokensIn, tokensOut, err
}

// GetModel returns the model name from the wrapped implementation.
func (c *circuitBreakedLLM) GetModel() string { return c.next.GetModel() }

// SetModel updates the model name in the wrapped implementation.
func (c *circuitBreakedLLM) SetModel(m string) { c.next.SetModel(m) }
