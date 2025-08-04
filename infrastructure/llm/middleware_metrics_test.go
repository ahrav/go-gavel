package llm

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockMetricsCollectorWithCapture wraps a mock to capture method calls
type mockMetricsCollectorWithCapture struct {
	*mockMetricsCollector
	onRecordHistogram func(metric string, value float64, labels map[string]string)
}

func (m *mockMetricsCollectorWithCapture) RecordHistogram(metric string, value float64, labels map[string]string) {
	if m.onRecordHistogram != nil {
		m.onRecordHistogram(metric, value, labels)
	}
	m.mockMetricsCollector.RecordHistogram(metric, value, labels)
}

func TestMetricsMiddleware_RecordsSuccessfulRequests(t *testing.T) {
	// Given a mock and metrics collector
	mock := NewMockCoreLLM()
	mock.Model = "gpt-4"
	metrics := newMockMetricsCollector()
	middleware := MetricsMiddleware(metrics)
	wrapped := middleware(mock)

	// When making a successful request
	ctx := context.Background()
	response, tokensIn, tokensOut, err := wrapped.DoRequest(ctx, "test prompt", nil)

	// Then it should succeed and record metrics
	require.NoError(t, err, "request should succeed")
	assert.Equal(t, "test response", response, "response should match")
	assert.Equal(t, 10, tokensIn, "input tokens should match")
	assert.Equal(t, 20, tokensOut, "output tokens should match")

	// Verify latency histogram was recorded
	latencyKey := "llm_latency_seconds:openai"
	assert.Contains(t, metrics.histograms, latencyKey, "should record latency histogram")
	assert.Greater(t, metrics.histograms[latencyKey], 0.0, "latency should be positive")

	// Verify request counter was recorded
	requestKey := "llm_requests_total:openai"
	assert.Equal(t, 1.0, metrics.counters[requestKey], "should record request counter")

	// Verify token counters were recorded (mock accumulates all token calls)
	inputTokenKey := "llm_tokens_total:openai"
	assert.Equal(t, 30.0, metrics.counters[inputTokenKey], "should record total tokens (input + output)")
}

func TestMetricsMiddleware_RecordsFailedRequests(t *testing.T) {
	// Given a mock that fails
	mock := NewMockCoreLLM()
	mock.Model = "claude-3-sonnet"
	mock.Error = errors.New("service error")
	metrics := newMockMetricsCollector()
	middleware := MetricsMiddleware(metrics)
	wrapped := middleware(mock)

	// When making a failed request
	ctx := context.Background()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	// Then it should fail and record error metrics
	require.Error(t, err, "request should fail")
	assert.Equal(t, "service error", err.Error(), "should return original error")

	// Verify latency histogram was recorded
	latencyKey := "llm_latency_seconds:anthropic"
	assert.Contains(t, metrics.histograms, latencyKey, "should record latency histogram")

	// Verify request counter was recorded with error status
	requestKey := "llm_requests_total:anthropic"
	assert.Equal(t, 1.0, metrics.counters[requestKey], "should record request counter")

	// Verify no token counters were recorded for failed requests
	tokenKey := "llm_tokens_total:anthropic"
	assert.NotContains(t, metrics.counters, tokenKey, "should not record tokens for failed requests")
}

func TestMetricsMiddleware_RecordsCircuitOpenErrors(t *testing.T) {
	// Given a mock that returns circuit open error
	mock := NewMockCoreLLM()
	mock.Model = "gpt-3.5-turbo"
	mock.Error = ErrCircuitOpen
	metrics := newMockMetricsCollector()
	middleware := MetricsMiddleware(metrics)
	wrapped := middleware(mock)

	// When making a request that hits circuit breaker
	ctx := context.Background()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	// Then it should fail with circuit open error
	require.Error(t, err, "request should fail")
	assert.Equal(t, ErrCircuitOpen, err, "should return circuit open error")

	// Verify metrics were recorded with circuit_open status
	latencyKey := "llm_latency_seconds:openai"
	assert.Contains(t, metrics.histograms, latencyKey, "should record latency histogram")

	requestKey := "llm_requests_total:openai"
	assert.Equal(t, 1.0, metrics.counters[requestKey], "should record request counter")
}

func TestMetricsMiddleware_RecordsTimeoutErrors(t *testing.T) {
	// Given a mock with slow response and short context timeout
	mock := NewMockCoreLLM()
	mock.Model = "gemini-pro"
	mock.ResponseDelay = 200 * time.Millisecond
	metrics := newMockMetricsCollector()
	middleware := MetricsMiddleware(metrics)
	wrapped := middleware(mock)

	// When making a request with timeout context
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	// Then it should timeout
	require.Error(t, err, "request should timeout")

	// Verify metrics were recorded with timeout status
	latencyKey := "llm_latency_seconds:google"
	assert.Contains(t, metrics.histograms, latencyKey, "should record latency histogram")

	requestKey := "llm_requests_total:google"
	assert.Equal(t, 1.0, metrics.counters[requestKey], "should record request counter")
}

func TestMetricsMiddleware_ExtractsProviderFromModel(t *testing.T) {
	tests := []struct {
		model            string
		expectedProvider string
	}{
		{"gpt-4", "openai"},
		{"gpt-3.5-turbo", "openai"},
		{"claude-3-sonnet", "anthropic"},
		{"claude-3-haiku", "anthropic"},
		{"gemini-pro", "google"},
		{"gemini-1.5-flash", "google"},
		{"custom-model", "unknown"},
		{"", "unknown"},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			// Given a mock with specific model
			mock := NewMockCoreLLM()
			mock.Model = tt.model
			metrics := newMockMetricsCollector()
			middleware := MetricsMiddleware(metrics)
			wrapped := middleware(mock)

			// When making a request
			ctx := context.Background()
			_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

			// Then provider should be correctly extracted
			require.NoError(t, err, "request should succeed")

			expectedKey := "llm_latency_seconds:" + tt.expectedProvider
			assert.Contains(t, metrics.histograms, expectedKey,
				"should record metrics with correct provider: %s", tt.expectedProvider)
		})
	}
}

func TestMetricsMiddleware_RecordsTokenCountsSeparately(t *testing.T) {
	// Given a mock with specific token counts
	mock := NewMockCoreLLM()
	mock.Model = "gpt-4"
	mock.TokensIn = 150
	mock.TokensOut = 75
	metrics := newMockMetricsCollector()
	middleware := MetricsMiddleware(metrics)
	wrapped := middleware(mock)

	// When making a successful request
	ctx := context.Background()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	// Then both token types should be recorded
	require.NoError(t, err, "request should succeed")

	// The mock collector implementation accumulates values
	tokenKey := "llm_tokens_total:openai"
	assert.Contains(t, metrics.counters, tokenKey, "should record token metrics")
	assert.Equal(t, 225.0, metrics.counters[tokenKey], "should accumulate input + output tokens")
}

func TestMetricsMiddleware_PassesThroughModelMethods(t *testing.T) {
	// Given a wrapped mock
	mock := NewMockCoreLLM()
	metrics := newMockMetricsCollector()
	middleware := MetricsMiddleware(metrics)
	wrapped := middleware(mock)

	// When calling model methods
	assert.Equal(t, "test-model", wrapped.GetModel(), "should pass through GetModel")

	wrapped.SetModel("new-model")
	assert.Equal(t, "new-model", wrapped.GetModel(), "should pass through SetModel")
	assert.Equal(t, "new-model", mock.GetModel(), "should update underlying mock")
}

func TestMetricsMiddleware_PreservesContextAndOptions(t *testing.T) {
	// Given a metrics middleware
	mock := NewMockCoreLLM()
	metrics := newMockMetricsCollector()
	middleware := MetricsMiddleware(metrics)
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

func TestMetricsMiddleware_RecordsLatencyAccurately(t *testing.T) {
	// Given a mock with controlled delay
	mock := NewMockCoreLLM()
	mock.Model = "gpt-4"
	mock.ResponseDelay = 100 * time.Millisecond
	metrics := newMockMetricsCollector()
	middleware := MetricsMiddleware(metrics)
	wrapped := middleware(mock)

	// When making a request
	ctx := context.Background()
	start := time.Now()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
	actualDuration := time.Since(start)

	// Then latency should be recorded accurately
	require.NoError(t, err, "request should succeed")

	latencyKey := "llm_latency_seconds:openai"
	recordedLatency := metrics.histograms[latencyKey]

	// Recorded latency should be close to actual duration
	assert.Greater(t, recordedLatency, 0.08, "recorded latency should be at least 80ms")
	assert.Less(t, recordedLatency, actualDuration.Seconds()+0.01,
		"recorded latency should not exceed actual duration by much")
}

func TestMetricsMiddleware_HandlesMultipleRequestsCorrectly(t *testing.T) {
	// Given a metrics middleware
	mock := NewMockCoreLLM()
	mock.Model = "claude-3-sonnet"
	metrics := newMockMetricsCollector()
	middleware := MetricsMiddleware(metrics)
	wrapped := middleware(mock)

	ctx := context.Background()

	// When making multiple successful requests
	for i := range 3 {
		_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
		require.NoError(t, err, "request %d should succeed", i+1)
	}

	// Then counters should accumulate
	requestKey := "llm_requests_total:anthropic"
	assert.Equal(t, 3.0, metrics.counters[requestKey], "should accumulate request counter")

	// When making a failed request
	mock.Error = errors.New("service error")
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
	require.Error(t, err, "last request should fail")

	// Then total counter should include failed request
	assert.Equal(t, 4.0, metrics.counters[requestKey], "should include failed request in counter")
}

func TestMetricsMiddleware_NilMetricsCollector(t *testing.T) {
	// Given a middleware with nil metrics collector
	mock := NewMockCoreLLM()
	middleware := MetricsMiddleware(nil)
	wrapped := middleware(mock)

	// When making a request
	ctx := context.Background()
	response, tokensIn, tokensOut, err := wrapped.DoRequest(ctx, "test prompt", nil)

	// Then it should succeed without panicking
	require.NoError(t, err, "request should succeed")
	assert.Equal(t, "test response", response, "response should match")
	assert.Equal(t, 10, tokensIn, "input tokens should match")
	assert.Equal(t, 20, tokensOut, "output tokens should match")
	assert.Equal(t, 1, mock.GetCallCount(), "should call underlying implementation")
}

func TestMetricsMiddleware_IncludesModelInLabels(t *testing.T) {
	// Given a mock with a specific model and custom metrics collector
	mock := NewMockCoreLLM()
	mock.Model = "gpt-4-turbo"

	var capturedLabels map[string]string
	customMetrics := &mockMetricsCollectorWithCapture{
		mockMetricsCollector: newMockMetricsCollector(),
		onRecordHistogram: func(metric string, value float64, labels map[string]string) {
			capturedLabels = labels
		},
	}
	middleware := MetricsMiddleware(customMetrics)
	wrapped := middleware(mock)

	// When making a request
	ctx := context.Background()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	// Then labels should include model information
	require.NoError(t, err, "request should succeed")
	require.NotNil(t, capturedLabels, "should capture labels")
	assert.Equal(t, "gpt-4-turbo", capturedLabels["model"], "should include model in labels")
	assert.Equal(t, "openai", capturedLabels["provider"], "should include provider in labels")
	assert.Equal(t, "success", capturedLabels["status"], "should include status in labels")
}
