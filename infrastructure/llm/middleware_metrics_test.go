package llm

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockMetricsCollectorWithCapture wraps a mock to capture method calls for testing.
type mockMetricsCollectorWithCapture struct {
	*mockMetricsCollector
	onRecordHistogram func(metric string, value float64, labels map[string]string)
}

// RecordHistogram captures the histogram recording for inspection in tests.
func (m *mockMetricsCollectorWithCapture) RecordHistogram(metric string, value float64, labels map[string]string) {
	if m.onRecordHistogram != nil {
		m.onRecordHistogram(metric, value, labels)
	}
	m.mockMetricsCollector.RecordHistogram(metric, value, labels)
}

// TestMetricsMiddleware_RecordsSuccessfulRequests tests that the metrics middleware
// correctly records metrics for successful LLM requests.
func TestMetricsMiddleware_RecordsSuccessfulRequests(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.Model = "gpt-4"
	metrics := newMockMetricsCollector()
	middleware := MetricsMiddleware(metrics)
	wrapped := middleware(mock)

	ctx := context.Background()
	response, tokensIn, tokensOut, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.NoError(t, err, "request should succeed")
	assert.Equal(t, "test response", response, "response should match")
	assert.Equal(t, 10, tokensIn, "input tokens should match")
	assert.Equal(t, 20, tokensOut, "output tokens should match")

	latencyKey := "llm_latency_seconds:openai"
	assert.Contains(t, metrics.histograms, latencyKey, "should record latency histogram")
	assert.Greater(t, metrics.histograms[latencyKey], 0.0, "latency should be positive")

	requestKey := "llm_requests_total:openai"
	assert.Equal(t, 1.0, metrics.counters[requestKey], "should record request counter")

	inputTokenKey := "llm_tokens_total:openai"
	assert.Equal(t, 30.0, metrics.counters[inputTokenKey], "should record total tokens (input + output)")
}

// TestMetricsMiddleware_RecordsFailedRequests tests that the metrics middleware
// correctly records metrics for failed LLM requests.
func TestMetricsMiddleware_RecordsFailedRequests(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.Model = "claude-3-sonnet"
	mock.Error = errors.New("service error")
	metrics := newMockMetricsCollector()
	middleware := MetricsMiddleware(metrics)
	wrapped := middleware(mock)

	ctx := context.Background()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.Error(t, err, "request should fail")
	assert.Equal(t, "service error", err.Error(), "should return original error")

	latencyKey := "llm_latency_seconds:anthropic"
	assert.Contains(t, metrics.histograms, latencyKey, "should record latency histogram")

	requestKey := "llm_requests_total:anthropic"
	assert.Equal(t, 1.0, metrics.counters[requestKey], "should record request counter")

	tokenKey := "llm_tokens_total:anthropic"
	assert.NotContains(t, metrics.counters, tokenKey, "should not record tokens for failed requests")
}

// TestMetricsMiddleware_RecordsCircuitOpenErrors tests that the metrics middleware
// correctly records metrics when a request is rejected by the circuit breaker.
func TestMetricsMiddleware_RecordsCircuitOpenErrors(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.Model = "gpt-3.5-turbo"
	mock.Error = ErrCircuitOpen
	metrics := newMockMetricsCollector()
	middleware := MetricsMiddleware(metrics)
	wrapped := middleware(mock)

	ctx := context.Background()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.Error(t, err, "request should fail")
	assert.Equal(t, ErrCircuitOpen, err, "should return circuit open error")

	latencyKey := "llm_latency_seconds:openai"
	assert.Contains(t, metrics.histograms, latencyKey, "should record latency histogram")

	requestKey := "llm_requests_total:openai"
	assert.Equal(t, 1.0, metrics.counters[requestKey], "should record request counter")
}

// TestMetricsMiddleware_RecordsTimeoutErrors tests that the metrics middleware
// correctly records metrics when a request times out.
func TestMetricsMiddleware_RecordsTimeoutErrors(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.Model = "gemini-pro"
	mock.ResponseDelay = 200 * time.Millisecond
	metrics := newMockMetricsCollector()
	middleware := MetricsMiddleware(metrics)
	wrapped := middleware(mock)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.Error(t, err, "request should timeout")

	latencyKey := "llm_latency_seconds:google"
	assert.Contains(t, metrics.histograms, latencyKey, "should record latency histogram")

	requestKey := "llm_requests_total:google"
	assert.Equal(t, 1.0, metrics.counters[requestKey], "should record request counter")
}

// TestMetricsMiddleware_ExtractsProviderFromModel tests that the metrics middleware
// correctly extracts the provider name from the model identifier.
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
			mock := NewMockCoreLLM()
			mock.Model = tt.model
			metrics := newMockMetricsCollector()
			middleware := MetricsMiddleware(metrics)
			wrapped := middleware(mock)

			ctx := context.Background()
			_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

			require.NoError(t, err, "request should succeed")

			expectedKey := "llm_latency_seconds:" + tt.expectedProvider
			assert.Contains(t, metrics.histograms, expectedKey,
				"should record metrics with correct provider: %s", tt.expectedProvider)
		})
	}
}

// TestMetricsMiddleware_RecordsTokenCountsSeparately tests that the metrics middleware
// records both input and output token counts.
func TestMetricsMiddleware_RecordsTokenCountsSeparately(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.Model = "gpt-4"
	mock.TokensIn = 150
	mock.TokensOut = 75
	metrics := newMockMetricsCollector()
	middleware := MetricsMiddleware(metrics)
	wrapped := middleware(mock)

	ctx := context.Background()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.NoError(t, err, "request should succeed")

	tokenKey := "llm_tokens_total:openai"
	assert.Contains(t, metrics.counters, tokenKey, "should record token metrics")
	assert.Equal(t, 225.0, metrics.counters[tokenKey], "should accumulate input + output tokens")
}

// TestMetricsMiddleware_PassesThroughModelMethods tests that the metrics middleware
// correctly passes through calls to the underlying CoreLLM's methods.
func TestMetricsMiddleware_PassesThroughModelMethods(t *testing.T) {
	mock := NewMockCoreLLM()
	metrics := newMockMetricsCollector()
	middleware := MetricsMiddleware(metrics)
	wrapped := middleware(mock)

	assert.Equal(t, "test-model", wrapped.GetModel(), "should pass through GetModel")

	wrapped.SetModel("new-model")
	assert.Equal(t, "new-model", wrapped.GetModel(), "should pass through SetModel")
	assert.Equal(t, "new-model", mock.GetModel(), "should update underlying mock")
}

// TestMetricsMiddleware_PreservesContextAndOptions tests that the metrics middleware
// preserves the context and options passed to the DoRequest method.
func TestMetricsMiddleware_PreservesContextAndOptions(t *testing.T) {
	mock := NewMockCoreLLM()
	metrics := newMockMetricsCollector()
	middleware := MetricsMiddleware(metrics)
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

// TestMetricsMiddleware_RecordsLatencyAccurately tests that the metrics middleware
// accurately records the latency of LLM requests.
func TestMetricsMiddleware_RecordsLatencyAccurately(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.Model = "gpt-4"
	mock.ResponseDelay = 100 * time.Millisecond
	metrics := newMockMetricsCollector()
	middleware := MetricsMiddleware(metrics)
	wrapped := middleware(mock)

	ctx := context.Background()
	start := time.Now()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
	actualDuration := time.Since(start)

	require.NoError(t, err, "request should succeed")

	latencyKey := "llm_latency_seconds:openai"
	recordedLatency := metrics.histograms[latencyKey]

	assert.Greater(t, recordedLatency, 0.08, "recorded latency should be at least 80ms")
	assert.Less(t, recordedLatency, actualDuration.Seconds()+0.01,
		"recorded latency should not exceed actual duration by much")
}

// TestMetricsMiddleware_HandlesMultipleRequestsCorrectly tests that the metrics middleware
// correctly handles and accumulates metrics over multiple requests.
func TestMetricsMiddleware_HandlesMultipleRequestsCorrectly(t *testing.T) {
	mock := NewMockCoreLLM()
	mock.Model = "claude-3-sonnet"
	metrics := newMockMetricsCollector()
	middleware := MetricsMiddleware(metrics)
	wrapped := middleware(mock)

	ctx := context.Background()

	for i := range 3 {
		_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
		require.NoError(t, err, "request %d should succeed", i+1)
	}

	requestKey := "llm_requests_total:anthropic"
	assert.Equal(t, 3.0, metrics.counters[requestKey], "should accumulate request counter")

	mock.Error = errors.New("service error")
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)
	require.Error(t, err, "last request should fail")

	assert.Equal(t, 4.0, metrics.counters[requestKey], "should include failed request in counter")
}

// TestMetricsMiddleware_NilMetricsCollector tests that the metrics middleware
// operates without panicking when the metrics collector is nil.
func TestMetricsMiddleware_NilMetricsCollector(t *testing.T) {
	mock := NewMockCoreLLM()
	middleware := MetricsMiddleware(nil)
	wrapped := middleware(mock)

	ctx := context.Background()
	response, tokensIn, tokensOut, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.NoError(t, err, "request should succeed")
	assert.Equal(t, "test response", response, "response should match")
	assert.Equal(t, 10, tokensIn, "input tokens should match")
	assert.Equal(t, 20, tokensOut, "output tokens should match")
	assert.Equal(t, 1, mock.GetCallCount(), "should call underlying implementation")
}

// TestMetricsMiddleware_IncludesModelInLabels tests that the metrics middleware
// includes the model name in the labels of recorded metrics.
func TestMetricsMiddleware_IncludesModelInLabels(t *testing.T) {
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

	ctx := context.Background()
	_, _, _, err := wrapped.DoRequest(ctx, "test prompt", nil)

	require.NoError(t, err, "request should succeed")
	require.NotNil(t, capturedLabels, "should capture labels")
	assert.Equal(t, "gpt-4-turbo", capturedLabels["model"], "should include model in labels")
	assert.Equal(t, "openai", capturedLabels["provider"], "should include provider in labels")
	assert.Equal(t, "success", capturedLabels["status"], "should include status in labels")
}
