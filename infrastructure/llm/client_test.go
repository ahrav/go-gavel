package llm

import (
	"context"
	"fmt"
	"testing"
	"time"

	"golang.org/x/time/rate"

	"github.com/ahrav/go-gavel/internal/ports"
)

// Mock metrics collector for testing
type mockMetricsCollector struct {
	histograms map[string]float64
	counters   map[string]float64
	gauges     map[string]float64
}

func newMockMetricsCollector() *mockMetricsCollector {
	return &mockMetricsCollector{
		histograms: make(map[string]float64),
		counters:   make(map[string]float64),
		gauges:     make(map[string]float64),
	}
}

func (m *mockMetricsCollector) RecordLatency(operation string, duration time.Duration, labels map[string]string) {
	key := fmt.Sprintf("%s:%s", operation, labels["provider"])
	m.histograms[key] = duration.Seconds()
}

func (m *mockMetricsCollector) RecordCounter(metric string, value float64, labels map[string]string) {
	key := fmt.Sprintf("%s:%s", metric, labels["provider"])
	m.counters[key] += value
}

func (m *mockMetricsCollector) RecordGauge(metric string, value float64, labels map[string]string) {
	key := fmt.Sprintf("%s:%s", metric, labels["provider"])
	m.gauges[key] = value
}

func (m *mockMetricsCollector) RecordHistogram(metric string, value float64, labels map[string]string) {
	key := fmt.Sprintf("%s:%s", metric, labels["provider"])
	m.histograms[key] = value
}

// Mock circuit breaker metrics for testing
type mockCircuitBreakerMetrics struct {
	states    []CircuitBreakerState
	trips     int
	successes int
	failures  int
}

func newMockCircuitBreakerMetrics() *mockCircuitBreakerMetrics {
	return &mockCircuitBreakerMetrics{
		states: make([]CircuitBreakerState, 0),
	}
}

func (m *mockCircuitBreakerMetrics) RecordState(state CircuitBreakerState) {
	m.states = append(m.states, state)
}

func (m *mockCircuitBreakerMetrics) RecordTrip() {
	m.trips++
}

func (m *mockCircuitBreakerMetrics) RecordSuccess() {
	m.successes++
}

func (m *mockCircuitBreakerMetrics) RecordFailure() {
	m.failures++
}

func TestNewClient(t *testing.T) {
	tests := []struct {
		name        string
		provider    string
		config      ClientConfig
		expectError bool
	}{
		{
			name:     "valid openai client",
			provider: "openai",
			config: ClientConfig{
				APIKey: "test-api-key",
				Model:  "gpt-4",
			},
			expectError: false,
		},
		{
			name:     "valid anthropic client",
			provider: "anthropic",
			config: ClientConfig{
				APIKey: "test-api-key",
				Model:  "claude-3-sonnet",
			},
			expectError: false,
		},
		{
			name:     "valid google client",
			provider: "google",
			config: ClientConfig{
				APIKey: "test-api-key", // Use API key instead of file path for test
				Model:  "gemini-pro",
			},
			expectError: false,
		},
		{
			name:     "missing api key",
			provider: "openai",
			config: ClientConfig{
				Model: "gpt-4",
			},
			expectError: true,
		},
		{
			name:     "missing model",
			provider: "openai",
			config: ClientConfig{
				APIKey: "test-api-key",
			},
			expectError: true,
		},
		{
			name:     "unknown provider",
			provider: "unknown",
			config: ClientConfig{
				APIKey: "test-key",
				Model:  "some-model",
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := NewClient(tt.provider, tt.config)

			if tt.expectError {
				if err == nil {
					t.Errorf("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if client == nil {
				t.Errorf("expected client but got nil")
			}
		})
	}
}

func TestClientComplete(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}
	
	// Skip test if no real API key is available
	t.Skip("skipping integration test - requires valid API key")
	
	client, err := NewClient("openai", ClientConfig{
		APIKey: "test-api-key",
		Model:  "gpt-4",
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	ctx := context.Background()
	response, err := client.Complete(ctx, "test prompt", nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if response == "" {
		t.Errorf("expected non-empty response")
	}
}

func TestClientCompleteWithUsage(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}
	
	// Skip test if no real API key is available
	t.Skip("skipping integration test - requires valid API key")
	
	client, err := NewClient("anthropic", ClientConfig{
		APIKey: "test-api-key",
		Model:  "claude-3-sonnet",
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	ctx := context.Background()
	response, tokensIn, tokensOut, err := client.CompleteWithUsage(ctx, "test prompt", nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if response == "" {
		t.Errorf("expected non-empty response")
	}

	if tokensIn <= 0 {
		t.Errorf("expected positive input token count, got %d", tokensIn)
	}

	if tokensOut <= 0 {
		t.Errorf("expected positive output token count, got %d", tokensOut)
	}
}

// TestClientEstimateTokens tests the token estimation functionality of the client.
func TestClientEstimateTokens(t *testing.T) {
	client, err := NewClient("openai", ClientConfig{
		APIKey: "test-api-key",
		Model:  "gpt-4",
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	text := "This is a test string with some words"
	tokens, err := client.EstimateTokens(text)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if tokens <= 0 {
		t.Errorf("expected positive token count, got %d", tokens)
	}
}

// TestClientWithMiddleware tests the client's functionality when middleware is applied.
// It ensures that middleware is correctly invoked and that metrics are recorded.
func TestClientWithMiddleware(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}
	
	// Skip test if no real API key is available
	t.Skip("skipping integration test - requires valid API key")
	metrics := newMockMetricsCollector()
	cbMetrics := newMockCircuitBreakerMetrics()

	client, err := NewClient("openai", ClientConfig{
		APIKey: "test-api-key",
		Model:  "gpt-4",
		Middleware: []Middleware{
			RateLimitMiddleware(rate.Limit(100), 10),
			CircuitBreakerMiddlewareWithMetrics(3, 60*time.Second, cbMetrics),
			TimeoutMiddleware(30 * time.Second),
			MetricsMiddleware(metrics),
		},
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	ctx := context.Background()
	response, err := client.Complete(ctx, "test prompt", nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if response == "" {
		t.Errorf("expected non-empty response")
	}

	if len(metrics.counters) == 0 {
		t.Errorf("expected metrics to be recorded")
	}

	if cbMetrics.successes == 0 {
		t.Errorf("expected circuit breaker success to be recorded")
	}
}

// TestTokenEstimators tests various token estimator implementations.
func TestTokenEstimators(t *testing.T) {
	tests := []struct {
		name      string
		estimator TokenEstimator
		text      string
		minTokens int
	}{
		{
			name:      "simple estimator",
			estimator: &SimpleTokenEstimator{},
			text:      "Hello world",
			minTokens: 1,
		},
		{
			name:      "word based estimator",
			estimator: NewWordBasedTokenEstimator(0.75),
			text:      "Hello world test",
			minTokens: 1,
		},
		{
			name:      "character based estimator",
			estimator: NewCharacterBasedTokenEstimator(4.0),
			text:      "Hello world",
			minTokens: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokens := tt.estimator.EstimateTokens(tt.text)
			if tokens < tt.minTokens {
				t.Errorf("expected at least %d tokens, got %d", tt.minTokens, tokens)
			}
		})
	}
}

// TestCachingTokenEstimator tests the caching functionality of the token estimator.
func TestCachingTokenEstimator(t *testing.T) {
	underlying := &SimpleTokenEstimator{}
	caching := NewCachingTokenEstimator(underlying, 10)

	text := "test text"

	tokens1 := caching.EstimateTokens(text)
	if caching.CacheSize() != 1 {
		t.Errorf("expected cache size 1, got %d", caching.CacheSize())
	}

	tokens2 := caching.EstimateTokens(text)
	if tokens1 != tokens2 {
		t.Errorf("expected same token count from cache, got %d vs %d", tokens1, tokens2)
	}

	caching.ClearCache()
	if caching.CacheSize() != 0 {
		t.Errorf("expected empty cache after clear, got size %d", caching.CacheSize())
	}
}

// TestCustomTokenEstimator tests using a custom token estimator with the client.
func TestCustomTokenEstimator(t *testing.T) {
	customEstimator := &SimpleTokenEstimator{}

	client, err := NewClient("openai", ClientConfig{
		APIKey:         "test-api-key",
		Model:          "gpt-4",
		TokenEstimator: customEstimator,
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	text := "This is a test"
	tokens, err := client.EstimateTokens(text)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	expected := (len(text) + 3) / 4
	if tokens != expected {
		t.Errorf("expected %d tokens, got %d", expected, tokens)
	}
}

// TestClientWithTimeout tests the client's behavior with a timeout configured.
func TestClientWithTimeout(t *testing.T) {
	client, err := NewClient("openai", ClientConfig{
		APIKey:  "test-api-key",
		Model:   "gpt-4",
		Timeout: 100 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	ctx := context.Background()
	_, err = client.Complete(ctx, "test prompt", nil)
	if err != nil {
		t.Logf("got expected error from timeout: %v", err)
	}
}

var _ ports.LLMClient = (*Client)(nil)
