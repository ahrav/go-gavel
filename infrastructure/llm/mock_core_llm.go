package llm

import (
	"context"
	"sync"
	"time"
)

// MockCoreLLM provides a configurable mock implementation of CoreLLM for testing.
// It allows precise control over response behavior, timing, and error conditions
// to facilitate comprehensive middleware testing.
type MockCoreLLM struct {
	mu sync.Mutex

	// Response configuration
	Response      string
	TokensIn      int
	TokensOut     int
	Error         error
	Model         string
	ResponseDelay time.Duration

	// Behavior flags
	FailUntilAttempt int  // Fail for first N attempts, then succeed
	AlternateErrors  bool // Alternate between success and failure

	// Tracking
	CallCount      int
	LastPrompt     string
	LastOpts       map[string]any
	LastContext    context.Context
	Contexts       []context.Context // All contexts received
	CallTimestamps []time.Time
}

// NewMockCoreLLM creates a new mock CoreLLM with default successful behavior.
func NewMockCoreLLM() *MockCoreLLM {
	return &MockCoreLLM{
		Response:       "test response",
		TokensIn:       10,
		TokensOut:      20,
		Model:          "test-model",
		Contexts:       make([]context.Context, 0),
		CallTimestamps: make([]time.Time, 0),
	}
}

// DoRequest implements the CoreLLM interface with configurable behavior.
func (m *MockCoreLLM) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Track the call
	m.CallCount++
	m.LastPrompt = prompt
	m.LastOpts = opts
	m.LastContext = ctx
	m.Contexts = append(m.Contexts, ctx)
	m.CallTimestamps = append(m.CallTimestamps, time.Now())

	// Simulate response delay if configured
	if m.ResponseDelay > 0 {
		select {
		case <-time.After(m.ResponseDelay):
			// Normal delay completion
		case <-ctx.Done():
			// Context cancelled during delay
			return "", 0, 0, ctx.Err()
		}
	}

	// Handle failure behaviors
	if m.FailUntilAttempt > 0 && m.CallCount <= m.FailUntilAttempt {
		if m.Error != nil {
			return "", 0, 0, m.Error
		}
		return "", 0, 0, &testError{message: "simulated failure"}
	}

	if m.AlternateErrors && m.CallCount%2 == 0 {
		if m.Error != nil {
			return "", 0, 0, m.Error
		}
		return "", 0, 0, &testError{message: "alternating failure"}
	}

	// Return configured response
	if m.Error != nil {
		return "", 0, 0, m.Error
	}

	return m.Response, m.TokensIn, m.TokensOut, nil
}

// GetModel returns the configured model name.
func (m *MockCoreLLM) GetModel() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.Model
}

// SetModel updates the model name.
func (m *MockCoreLLM) SetModel(model string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.Model = model
}

// Reset clears all tracking data while preserving configuration.
func (m *MockCoreLLM) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.CallCount = 0
	m.LastPrompt = ""
	m.LastOpts = nil
	m.LastContext = nil
	m.Contexts = make([]context.Context, 0)
	m.CallTimestamps = make([]time.Time, 0)
}

// GetCallCount returns the number of times DoRequest was called.
func (m *MockCoreLLM) GetCallCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.CallCount
}

// GetTimeBetweenCalls calculates the duration between consecutive calls.
// Returns nil if fewer than 2 calls have been made.
func (m *MockCoreLLM) GetTimeBetweenCalls(call1, call2 int) *time.Duration {
	m.mu.Lock()
	defer m.mu.Unlock()

	if call1 < 0 || call2 < 0 || call1 >= len(m.CallTimestamps) || call2 >= len(m.CallTimestamps) {
		return nil
	}

	duration := m.CallTimestamps[call2].Sub(m.CallTimestamps[call1])
	return &duration
}

// testError provides a simple error type for testing.
type testError struct {
	message string
}

func (e *testError) Error() string {
	return e.message
}
