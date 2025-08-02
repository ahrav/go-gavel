package llm

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/ahrav/go-gavel/internal/ports"
)

// Compile-time check to ensure mockLLMClient implements ports.LLMClient
var _ ports.LLMClient = (*mockLLMClient)(nil)

// mockLLMClient is a test double for ports.LLMClient that allows us to simulate
// various error conditions and success scenarios for testing the retry logic.
type mockLLMClient struct {
	// completeCalls tracks how many times Complete() was called
	completeCalls int
	// completeWithUsageCalls tracks how many times CompleteWithUsage() was called
	completeWithUsageCalls int
	// completeResponses contains the responses to return from Complete() calls
	completeResponses []completeResponse
	// completeWithUsageResponses contains the responses to return from CompleteWithUsage() calls
	completeWithUsageResponses []completeWithUsageResponse
	// model is the model name to return from GetModel()
	model string
	// estimateTokensResponse is the response to return from EstimateTokens()
	estimateTokensResponse int
	// estimateTokensError is the error to return from EstimateTokens()
	estimateTokensError error
}

type completeResponse struct {
	response string
	err      error
}

type completeWithUsageResponse struct {
	response  string
	tokensIn  int
	tokensOut int
	err       error
}

func (m *mockLLMClient) Complete(ctx context.Context, prompt string, options map[string]any) (string, error) {
	if m.completeCalls >= len(m.completeResponses) {
		return "", errors.New("no more responses configured")
	}
	resp := m.completeResponses[m.completeCalls]
	m.completeCalls++
	return resp.response, resp.err
}

func (m *mockLLMClient) CompleteWithUsage(ctx context.Context, prompt string, options map[string]any) (string, int, int, error) {
	if m.completeWithUsageCalls >= len(m.completeWithUsageResponses) {
		return "", 0, 0, errors.New("no more responses configured")
	}
	resp := m.completeWithUsageResponses[m.completeWithUsageCalls]
	m.completeWithUsageCalls++
	return resp.response, resp.tokensIn, resp.tokensOut, resp.err
}

func (m *mockLLMClient) EstimateTokens(text string) (int, error) {
	return m.estimateTokensResponse, m.estimateTokensError
}

func (m *mockLLMClient) GetModel() string {
	return m.model
}

func TestNewRetryingLLMClient(t *testing.T) {
	mockClient := &mockLLMClient{model: "test-model"}
	config := RetryConfig{
		MaxAttempts:   2,
		BaseDelay:     100 * time.Millisecond,
		MaxDelay:      1 * time.Second,
		JitterPercent: 0.1,
	}

	retryClient := NewRetryingLLMClient(mockClient, config)

	if retryClient == nil {
		t.Fatal("NewRetryingLLMClient returned nil")
	}
	if retryClient.client != mockClient {
		t.Error("RetryingLLMClient should wrap the provided client")
	}
	if retryClient.config != config {
		t.Error("RetryingLLMClient should use the provided config")
	}
}

func TestDefaultRetryConfig(t *testing.T) {
	config := DefaultRetryConfig()

	if config.MaxAttempts != DefaultMaxAttempts {
		t.Errorf("Expected MaxAttempts %d, got %d", DefaultMaxAttempts, config.MaxAttempts)
	}
	if config.BaseDelay != DefaultBaseDelay {
		t.Errorf("Expected BaseDelay %v, got %v", DefaultBaseDelay, config.BaseDelay)
	}
	if config.MaxDelay != DefaultMaxDelay {
		t.Errorf("Expected MaxDelay %v, got %v", DefaultMaxDelay, config.MaxDelay)
	}
	if config.JitterPercent != DefaultJitterPercent {
		t.Errorf("Expected JitterPercent %f, got %f", DefaultJitterPercent, config.JitterPercent)
	}
}

func TestComplete_Success(t *testing.T) {
	mockClient := &mockLLMClient{
		model: "test-model",
		completeResponses: []completeResponse{
			{response: "test response", err: nil},
		},
	}

	retryClient := NewRetryingLLMClient(mockClient, DefaultRetryConfig())
	response, err := retryClient.Complete(context.Background(), "test prompt", nil)

	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if response != "test response" {
		t.Errorf("Expected 'test response', got '%s'", response)
	}
	if mockClient.completeCalls != 1 {
		t.Errorf("Expected 1 call to Complete, got %d", mockClient.completeCalls)
	}
}

func TestComplete_RetryableError(t *testing.T) {
	mockClient := &mockLLMClient{
		model: "test-model",
		completeResponses: []completeResponse{
			{response: "", err: errors.New("rate limit exceeded")},
			{response: "", err: errors.New("timeout")},
			{response: "success after retries", err: nil},
		},
	}

	config := RetryConfig{
		MaxAttempts:   2,
		BaseDelay:     1 * time.Millisecond, // Very short delay for testing
		MaxDelay:      10 * time.Millisecond,
		JitterPercent: 0.0, // No jitter for predictable testing
	}

	retryClient := NewRetryingLLMClient(mockClient, config)
	response, err := retryClient.Complete(context.Background(), "test prompt", nil)

	if err != nil {
		t.Fatalf("Expected no error after retries, got %v", err)
	}
	if response != "success after retries" {
		t.Errorf("Expected 'success after retries', got '%s'", response)
	}
	if mockClient.completeCalls != 3 {
		t.Errorf("Expected 3 calls to Complete (initial + 2 retries), got %d", mockClient.completeCalls)
	}
}

func TestComplete_NonRetryableError(t *testing.T) {
	mockClient := &mockLLMClient{
		model: "test-model",
		completeResponses: []completeResponse{
			{response: "", err: errors.New("invalid request")},
		},
	}

	retryClient := NewRetryingLLMClient(mockClient, DefaultRetryConfig())
	response, err := retryClient.Complete(context.Background(), "test prompt", nil)

	if err == nil {
		t.Fatal("Expected error for non-retryable error")
	}
	if response != "" {
		t.Errorf("Expected empty response, got '%s'", response)
	}
	if mockClient.completeCalls != 1 {
		t.Errorf("Expected 1 call to Complete (no retries for non-retryable error), got %d", mockClient.completeCalls)
	}
}

func TestComplete_ExceedsMaxAttempts(t *testing.T) {
	mockClient := &mockLLMClient{
		model: "test-model",
		completeResponses: []completeResponse{
			{response: "", err: errors.New("rate limit exceeded")},
			{response: "", err: errors.New("rate limit exceeded")},
			{response: "", err: errors.New("rate limit exceeded")},
			{response: "", err: errors.New("rate limit exceeded")},
		},
	}

	config := RetryConfig{
		MaxAttempts:   2,
		BaseDelay:     1 * time.Millisecond,
		MaxDelay:      10 * time.Millisecond,
		JitterPercent: 0.0,
	}

	retryClient := NewRetryingLLMClient(mockClient, config)
	response, err := retryClient.Complete(context.Background(), "test prompt", nil)

	if err == nil {
		t.Fatal("Expected error after exceeding max attempts")
	}
	if !strings.Contains(err.Error(), "LLM call failed after 3 attempts") {
		t.Errorf("Expected error message about max attempts, got: %v", err)
	}
	if response != "" {
		t.Errorf("Expected empty response, got '%s'", response)
	}
	if mockClient.completeCalls != 3 {
		t.Errorf("Expected 3 calls to Complete (initial + 2 retries), got %d", mockClient.completeCalls)
	}
}

func TestComplete_ContextCancellation(t *testing.T) {
	mockClient := &mockLLMClient{
		model: "test-model",
		completeResponses: []completeResponse{
			{response: "", err: errors.New("rate limit exceeded")},
			{response: "", err: errors.New("rate limit exceeded")},
		},
	}

	config := RetryConfig{
		MaxAttempts:   2,
		BaseDelay:     100 * time.Millisecond,
		MaxDelay:      1 * time.Second,
		JitterPercent: 0.0,
	}

	retryClient := NewRetryingLLMClient(mockClient, config)

	// Create a context that will be cancelled after a short delay
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()

	response, err := retryClient.Complete(ctx, "test prompt", nil)

	if err == nil {
		t.Fatal("Expected error due to context cancellation")
	}
	if !strings.Contains(err.Error(), "context cancelled during retry") {
		t.Errorf("Expected context cancellation error, got: %v", err)
	}
	if response != "" {
		t.Errorf("Expected empty response, got '%s'", response)
	}
}

func TestCompleteWithUsage_Success(t *testing.T) {
	mockClient := &mockLLMClient{
		model: "test-model",
		completeWithUsageResponses: []completeWithUsageResponse{
			{response: "test response", tokensIn: 10, tokensOut: 20, err: nil},
		},
	}

	retryClient := NewRetryingLLMClient(mockClient, DefaultRetryConfig())
	response, tokensIn, tokensOut, err := retryClient.CompleteWithUsage(context.Background(), "test prompt", nil)

	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if response != "test response" {
		t.Errorf("Expected 'test response', got '%s'", response)
	}
	if tokensIn != 10 {
		t.Errorf("Expected tokensIn 10, got %d", tokensIn)
	}
	if tokensOut != 20 {
		t.Errorf("Expected tokensOut 20, got %d", tokensOut)
	}
	if mockClient.completeWithUsageCalls != 1 {
		t.Errorf("Expected 1 call to CompleteWithUsage, got %d", mockClient.completeWithUsageCalls)
	}
}

func TestCompleteWithUsage_Retry(t *testing.T) {
	mockClient := &mockLLMClient{
		model: "test-model",
		completeWithUsageResponses: []completeWithUsageResponse{
			{response: "", tokensIn: 0, tokensOut: 0, err: errors.New("rate limit exceeded")},
			{response: "success", tokensIn: 15, tokensOut: 25, err: nil},
		},
	}

	config := RetryConfig{
		MaxAttempts:   1,
		BaseDelay:     1 * time.Millisecond,
		MaxDelay:      10 * time.Millisecond,
		JitterPercent: 0.0,
	}

	retryClient := NewRetryingLLMClient(mockClient, config)
	response, tokensIn, tokensOut, err := retryClient.CompleteWithUsage(context.Background(), "test prompt", nil)

	if err != nil {
		t.Fatalf("Expected no error after retry, got %v", err)
	}
	if response != "success" {
		t.Errorf("Expected 'success', got '%s'", response)
	}
	if tokensIn != 15 {
		t.Errorf("Expected tokensIn 15, got %d", tokensIn)
	}
	if tokensOut != 25 {
		t.Errorf("Expected tokensOut 25, got %d", tokensOut)
	}
	if mockClient.completeWithUsageCalls != 2 {
		t.Errorf("Expected 2 calls to CompleteWithUsage, got %d", mockClient.completeWithUsageCalls)
	}
}

func TestEstimateTokens(t *testing.T) {
	mockClient := &mockLLMClient{
		model:                  "test-model",
		estimateTokensResponse: 42,
		estimateTokensError:    nil,
	}

	retryClient := NewRetryingLLMClient(mockClient, DefaultRetryConfig())
	tokens, err := retryClient.EstimateTokens("test text")

	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if tokens != 42 {
		t.Errorf("Expected 42 tokens, got %d", tokens)
	}
}

func TestGetModel(t *testing.T) {
	mockClient := &mockLLMClient{model: "test-model-123"}

	retryClient := NewRetryingLLMClient(mockClient, DefaultRetryConfig())
	model := retryClient.GetModel()

	if model != "test-model-123" {
		t.Errorf("Expected 'test-model-123', got '%s'", model)
	}
}

func TestIsRetryableError(t *testing.T) {
	retryClient := NewRetryingLLMClient(&mockLLMClient{}, DefaultRetryConfig())

	testCases := []struct {
		name        string
		err         error
		shouldRetry bool
	}{
		{
			name:        "nil error",
			err:         nil,
			shouldRetry: false,
		},
		{
			name:        "rate limit error",
			err:         errors.New("rate limit exceeded"),
			shouldRetry: true,
		},
		{
			name:        "too many requests error",
			err:         errors.New("too many requests"),
			shouldRetry: true,
		},
		{
			name:        "timeout error",
			err:         errors.New("request timeout"),
			shouldRetry: true,
		},
		{
			name:        "connection refused error",
			err:         errors.New("connection refused"),
			shouldRetry: true,
		},
		{
			name:        "network error",
			err:         errors.New("network error occurred"),
			shouldRetry: true,
		},
		{
			name:        "internal server error",
			err:         errors.New("internal server error"),
			shouldRetry: true,
		},
		{
			name:        "bad gateway error",
			err:         errors.New("bad gateway"),
			shouldRetry: true,
		},
		{
			name:        "gateway timeout error",
			err:         errors.New("gateway timeout"),
			shouldRetry: true,
		},
		{
			name:        "service unavailable error",
			err:         errors.New("service unavailable"),
			shouldRetry: true,
		},
		{
			name:        "temporary failure error",
			err:         errors.New("temporary failure"),
			shouldRetry: true,
		},
		{
			name:        "connection reset error",
			err:         errors.New("connection reset by peer"),
			shouldRetry: true,
		},
		{
			name:        "case insensitive matching",
			err:         errors.New("RATE LIMIT EXCEEDED"),
			shouldRetry: true,
		},
		{
			name:        "non-retryable error",
			err:         errors.New("invalid request format"),
			shouldRetry: false,
		},
		{
			name:        "authentication error",
			err:         errors.New("unauthorized access"),
			shouldRetry: false,
		},
		{
			name:        "validation error",
			err:         errors.New("invalid parameter"),
			shouldRetry: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := retryClient.isRetryableError(tc.err)
			if result != tc.shouldRetry {
				t.Errorf("Expected isRetryableError(%v) = %v, got %v", tc.err, tc.shouldRetry, result)
			}
		})
	}
}

func TestCalculateRetryDelay(t *testing.T) {
	config := RetryConfig{
		BaseDelay:     100 * time.Millisecond,
		MaxDelay:      1 * time.Second,
		JitterPercent: 0.0, // No jitter for predictable testing
	}
	retryClient := NewRetryingLLMClient(&mockLLMClient{}, config)

	testCases := []struct {
		name           string
		attempt        int
		expectedDelay  time.Duration
		maxExpectedCap time.Duration
	}{
		{
			name:          "first retry",
			attempt:       0,
			expectedDelay: 100 * time.Millisecond,
		},
		{
			name:          "second retry",
			attempt:       1,
			expectedDelay: 200 * time.Millisecond,
		},
		{
			name:          "third retry",
			attempt:       2,
			expectedDelay: 400 * time.Millisecond,
		},
		{
			name:          "fourth retry",
			attempt:       3,
			expectedDelay: 800 * time.Millisecond,
		},
		{
			name:           "capped at max delay",
			attempt:        4,
			maxExpectedCap: 1 * time.Second,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			delay := retryClient.calculateRetryDelay(tc.attempt)
			if tc.maxExpectedCap > 0 {
				if delay > tc.maxExpectedCap {
					t.Errorf("Expected delay to be capped at %v, got %v", tc.maxExpectedCap, delay)
				}
			} else {
				if delay != tc.expectedDelay {
					t.Errorf("Expected delay %v, got %v", tc.expectedDelay, delay)
				}
			}
		})
	}
}

func TestCalculateRetryDelay_WithJitter(t *testing.T) {
	config := RetryConfig{
		BaseDelay:     100 * time.Millisecond,
		MaxDelay:      1 * time.Second,
		JitterPercent: 0.1,
	}
	retryClient := NewRetryingLLMClient(&mockLLMClient{}, config)

	// Test that jitter produces different delays within expected range
	attempt := 1
	expectedBase := 200 * time.Millisecond
	jitterRange := int64(float64(expectedBase) * 0.1)

	delays := make([]time.Duration, 10)
	for i := 0; i < 10; i++ {
		delays[i] = retryClient.calculateRetryDelay(attempt)
	}

	// Check that delays are within expected range
	for i, delay := range delays {
		minDelay := expectedBase - time.Duration(jitterRange)
		maxDelay := expectedBase + time.Duration(jitterRange)
		if delay < minDelay || delay > maxDelay {
			t.Errorf("Delay %d (%v) is outside expected range [%v, %v]", i, delay, minDelay, maxDelay)
		}
	}
}

func TestCalculateRetryDelay_NegativeDelayHandling(t *testing.T) {
	// Test edge case where jitter could theoretically make delay negative
	config := RetryConfig{
		BaseDelay:     1 * time.Nanosecond, // Very small base delay
		MaxDelay:      1 * time.Second,
		JitterPercent: 1.0, // 100% jitter
	}
	retryClient := NewRetryingLLMClient(&mockLLMClient{}, config)

	delay := retryClient.calculateRetryDelay(0)

	// Should never return negative delay
	if delay < 0 {
		t.Errorf("Expected non-negative delay, got %v", delay)
	}

	// Should return at least the base delay when negative is detected
	if delay < config.BaseDelay {
		t.Errorf("Expected delay to be at least base delay %v, got %v", config.BaseDelay, delay)
	}
}

// Benchmark tests to ensure retry logic doesn't introduce significant overhead
func BenchmarkComplete_NoRetry(b *testing.B) {
	mockClient := &mockLLMClient{
		model:             "test-model",
		completeResponses: make([]completeResponse, b.N),
	}
	for i := 0; i < b.N; i++ {
		mockClient.completeResponses[i] = completeResponse{response: "success", err: nil}
	}

	retryClient := NewRetryingLLMClient(mockClient, DefaultRetryConfig())
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := retryClient.Complete(ctx, "test prompt", nil)
		if err != nil {
			b.Fatalf("Unexpected error: %v", err)
		}
	}
}

func BenchmarkCompleteWithUsage_NoRetry(b *testing.B) {
	mockClient := &mockLLMClient{
		model:                      "test-model",
		completeWithUsageResponses: make([]completeWithUsageResponse, b.N),
	}
	for i := 0; i < b.N; i++ {
		mockClient.completeWithUsageResponses[i] = completeWithUsageResponse{
			response: "success", tokensIn: 10, tokensOut: 20, err: nil,
		}
	}

	retryClient := NewRetryingLLMClient(mockClient, DefaultRetryConfig())
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _, err := retryClient.CompleteWithUsage(ctx, "test prompt", nil)
		if err != nil {
			b.Fatalf("Unexpected error: %v", err)
		}
	}
}
