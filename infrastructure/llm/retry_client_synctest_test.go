//go:build goexperiment.synctest

// Package llm provides synctest-based tests for the retry client.
// These tests require GOEXPERIMENT=synctest to run and demonstrate
// how synctest can make timing-based tests faster and more reliable.
package llm

import (
	"context"
	"errors"
	"strings"
	"testing"
	"testing/synctest"
	"time"
)

// TestComplete_RetryWithSynctest demonstrates using synctest for controlled timing.
// This test is much faster than the traditional retry test because it doesn't
// actually wait for real time to pass.
func TestComplete_RetryWithSynctest(t *testing.T) {
	synctest.Run(func() {
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
			BaseDelay:     1 * time.Second, // Can use realistic delays with synctest
			MaxDelay:      10 * time.Second,
			JitterPercent: 0.0,
		}

		retryClient := NewRetryingLLMClient(mockClient, config)

		start := time.Now()
		response, err := retryClient.Complete(context.Background(), "test prompt", nil)
		elapsed := time.Since(start)

		if err != nil {
			t.Fatalf("Expected no error after retries, got %v", err)
		}
		if response != "success after retries" {
			t.Errorf("Expected 'success after retries', got '%s'", response)
		}
		if mockClient.completeCalls != 3 {
			t.Errorf("Expected 3 calls to Complete, got %d", mockClient.completeCalls)
		}

		// With synctest, we can verify that the appropriate time passed
		// without actually waiting for it
		expectedDelay := 1*time.Second + 2*time.Second // BaseDelay * (2^0 + 2^1)
		if elapsed < expectedDelay {
			t.Errorf("Expected at least %v elapsed time, got %v", expectedDelay, elapsed)
		}

		t.Logf("Synctest completed in %v (simulated %v)", elapsed, expectedDelay)
	})
}

// TestComplete_ContextCancellationWithSynctest shows how synctest handles context cancellation.
func TestComplete_ContextCancellationWithSynctest(t *testing.T) {
	synctest.Run(func() {
		mockClient := &mockLLMClient{
			model: "test-model",
			completeResponses: []completeResponse{
				{response: "", err: errors.New("rate limit exceeded")},
				{response: "", err: errors.New("rate limit exceeded")},
			},
		}

		config := RetryConfig{
			MaxAttempts:   2,
			BaseDelay:     1 * time.Second,
			MaxDelay:      10 * time.Second,
			JitterPercent: 0.0,
		}

		retryClient := NewRetryingLLMClient(mockClient, config)

		// Create a context that will be cancelled after a delay
		ctx, cancel := context.WithCancel(context.Background())

		// Cancel the context after a short delay in a separate goroutine
		go func() {
			time.Sleep(500 * time.Millisecond)
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

		// The first call should have happened, but retries should be cancelled
		if mockClient.completeCalls != 1 {
			t.Errorf("Expected 1 call to Complete before cancellation, got %d", mockClient.completeCalls)
		}
	})
}

// TestCalculateRetryDelay_TimingWithSynctest verifies delay calculations with controlled time.
func TestCalculateRetryDelay_TimingWithSynctest(t *testing.T) {
	synctest.Run(func() {
		config := RetryConfig{
			BaseDelay:     100 * time.Millisecond,
			MaxDelay:      1 * time.Second,
			JitterPercent: 0.0, // No jitter for predictable testing
		}
		retryClient := NewRetryingLLMClient(&mockLLMClient{}, config)

		testCases := []struct {
			name          string
			attempt       int
			expectedDelay time.Duration
		}{
			{"first retry", 0, 100 * time.Millisecond},
			{"second retry", 1, 200 * time.Millisecond},
			{"third retry", 2, 400 * time.Millisecond},
			{"fourth retry", 3, 800 * time.Millisecond},
			{"capped at max", 4, 1 * time.Second},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				start := time.Now()

				// Simulate waiting for the calculated delay
				delay := retryClient.calculateRetryDelay(tc.attempt)
				time.Sleep(delay)

				elapsed := time.Since(start)

				if delay != tc.expectedDelay {
					t.Errorf("Expected delay %v, got %v", tc.expectedDelay, delay)
				}

				// With synctest, the elapsed time should match the delay exactly
				if elapsed != delay {
					t.Errorf("Expected elapsed time %v to match delay %v", elapsed, delay)
				}
			})
		}
	})
}

// TestCompleteWithUsage_RetryWithSynctest demonstrates CompleteWithUsage with synctest.
func TestCompleteWithUsage_RetryWithSynctest(t *testing.T) {
	synctest.Run(func() {
		mockClient := &mockLLMClient{
			model: "test-model",
			completeWithUsageResponses: []completeWithUsageResponse{
				{response: "", tokensIn: 0, tokensOut: 0, err: errors.New("rate limit exceeded")},
				{response: "success", tokensIn: 15, tokensOut: 25, err: nil},
			},
		}

		config := RetryConfig{
			MaxAttempts:   1,
			BaseDelay:     500 * time.Millisecond,
			MaxDelay:      2 * time.Second,
			JitterPercent: 0.0,
		}

		retryClient := NewRetryingLLMClient(mockClient, config)

		start := time.Now()
		response, tokensIn, tokensOut, err := retryClient.CompleteWithUsage(context.Background(), "test prompt", nil)
		elapsed := time.Since(start)

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

		// Should have waited for the base delay
		expectedDelay := 500 * time.Millisecond
		if elapsed < expectedDelay {
			t.Errorf("Expected at least %v elapsed time, got %v", expectedDelay, elapsed)
		}

		t.Logf("CompleteWithUsage synctest completed in %v", elapsed)
	})
}
