package testutils

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestCatastrophicTimeouts tests timeout behavior under various conditions.
func TestCatastrophicTimeouts(t *testing.T) {
	dataset := GenerateSampleBenchmarkDataset(10, 42)

	testCases := []struct {
		name          string
		timeout       time.Duration
		ctxTimeout    time.Duration
		expectError   bool
		errorContains string
	}{
		{
			name:          "Short timeout",
			timeout:       10 * time.Millisecond,
			ctxTimeout:    1 * time.Second,
			expectError:   true,
			errorContains: "timeout",
		},
		{
			name:          "Context cancellation before timeout",
			timeout:       1 * time.Second,
			ctxTimeout:    10 * time.Millisecond,
			expectError:   true,
			errorContains: "context",
		},
		{
			name:          "No timeout",
			timeout:       0, // No timeout set
			ctxTimeout:    1 * time.Second,
			expectError:   false,
			errorContains: "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client := NewBenchmarkMockLLMClient("timeout-test", dataset, ConservativeJudge)

			if tc.timeout > 0 {
				client.SimulateTimeout(tc.timeout)
			}

			ctx, cancel := context.WithTimeout(context.Background(), tc.ctxTimeout)
			defer cancel()

			prompt := "Rate this answer: Question: Test? Answer: Test answer"
			start := time.Now()
			response, err := client.Complete(ctx, prompt, nil)
			elapsed := time.Since(start)

			if tc.expectError {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.errorContains)
				assert.Empty(t, response)

				// Verify timeout was respected
				if tc.timeout > 0 && tc.timeout < tc.ctxTimeout {
					assert.GreaterOrEqual(t, elapsed, tc.timeout)
					assert.Less(t, elapsed, tc.timeout+50*time.Millisecond) // Allow some slack
				}
			} else {
				require.NoError(t, err)
				assert.NotEmpty(t, response)

				// Verify response is valid JSON
				var result map[string]any
				err = json.Unmarshal([]byte(response), &result)
				require.NoError(t, err)
			}
		})
	}
}

// TestPartialAndMalformedResponses tests handling of corrupted responses.
func TestPartialAndMalformedResponses(t *testing.T) {
	dataset := GenerateSampleBenchmarkDataset(10, 123)
	ctx := context.Background()

	t.Run("Partial Response", func(t *testing.T) {
		client := NewBenchmarkMockLLMClient("partial-test", dataset, AnalyticalJudge)
		client.SimulatePartialResponse()

		prompt := "Rate this answer: Question: What is 2+2? Answer: 4"
		response, err := client.Complete(ctx, prompt, nil)

		// Should not error, but response will be incomplete
		require.NoError(t, err)
		assert.NotEmpty(t, response)

		// Response should be truncated
		assert.True(t, strings.HasSuffix(response, "...") || len(response) < 20)

		// Should not be valid JSON
		var result map[string]any
		err = json.Unmarshal([]byte(response), &result)
		assert.Error(t, err, "Expected invalid JSON due to truncation")
	})

	t.Run("Malformed JSON Response", func(t *testing.T) {
		client := NewBenchmarkMockLLMClient("malformed-test", dataset, ComprehensiveJudge)
		client.SimulateMalformedJSON()

		prompt := "Rate this answer: Question: What is 2+2? Answer: 4"
		response, err := client.Complete(ctx, prompt, nil)

		// Should not error at Complete level
		require.NoError(t, err)
		assert.NotEmpty(t, response)

		// Response should contain corrupted JSON
		assert.Contains(t, response, `"scor`) // Corrupted "score" field

		// Should fail JSON parsing
		var result map[string]any
		err = json.Unmarshal([]byte(response), &result)
		assert.Error(t, err, "Expected JSON parse error due to corruption")
	})
}

// TestNetworkFailures tests various network failure scenarios.
func TestNetworkFailures(t *testing.T) {
	dataset := GenerateSampleBenchmarkDataset(100, 456)
	ctx := context.Background()

	testCases := []struct {
		name        string
		failureRate float64
		requests    int
		minFailures int
		maxFailures int
	}{
		{
			name:        "10% failure rate",
			failureRate: 0.1,
			requests:    100,
			minFailures: 5,  // At least 5%
			maxFailures: 20, // At most 20%
		},
		{
			name:        "50% failure rate",
			failureRate: 0.5,
			requests:    100,
			minFailures: 40, // At least 40%
			maxFailures: 60, // At most 60%
		},
		{
			name:        "90% failure rate",
			failureRate: 0.9,
			requests:    100,
			minFailures: 85, // At least 85%
			maxFailures: 95, // At most 95%
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client := NewBenchmarkMockLLMClient("network-test", dataset, BiasedJudge)
			client.SimulateNetworkFailure(tc.failureRate)

			failureCount := 0
			successCount := 0

			for i := 0; i < tc.requests; i++ {
				question := dataset.Questions[i%len(dataset.Questions)]
				prompt := fmt.Sprintf(
					"Rate this answer: Question: %s Answer: %s",
					question.Question,
					question.Answers[0].Content,
				)

				_, err := client.Complete(ctx, prompt, nil)
				if err != nil {
					assert.Contains(t, err.Error(), "network error")
					failureCount++
				} else {
					successCount++
				}
			}

			t.Logf("Network failures: %d/%d (%.1f%%)",
				failureCount, tc.requests, float64(failureCount)/float64(tc.requests)*100)

			// Verify failure rate is within expected range
			assert.GreaterOrEqual(t, failureCount, tc.minFailures,
				"Expected at least %d failures, got %d", tc.minFailures, failureCount)
			assert.LessOrEqual(t, failureCount, tc.maxFailures,
				"Expected at most %d failures, got %d", tc.maxFailures, failureCount)
		})
	}
}

// TestRateLimiting tests rate limiting behavior.
func TestRateLimiting(t *testing.T) {
	dataset := GenerateSampleBenchmarkDataset(20, 789)
	ctx := context.Background()

	testCases := []struct {
		name          string
		limitAfter    int
		totalRequests int
		expectedPass  int
	}{
		{
			name:          "Rate limit after 5 requests",
			limitAfter:    5,
			totalRequests: 10,
			expectedPass:  5,
		},
		{
			name:          "Rate limit after 1 request",
			limitAfter:    1,
			totalRequests: 5,
			expectedPass:  1,
		},
		{
			name:          "No rate limit",
			limitAfter:    0, // 0 means no limit
			totalRequests: 10,
			expectedPass:  10,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client := NewBenchmarkMockLLMClient("ratelimit-test", dataset, ConservativeJudge)

			if tc.limitAfter > 0 {
				client.SimulateRateLimiting(tc.limitAfter)
			}

			successCount := 0
			rateLimitCount := 0

			for i := 0; i < tc.totalRequests; i++ {
				prompt := fmt.Sprintf("Rate answer %d: Question: Test? Answer: Test", i)
				_, err := client.Complete(ctx, prompt, nil)

				if err == nil {
					successCount++
				} else if strings.Contains(err.Error(), "rate limit") {
					rateLimitCount++
				} else {
					t.Errorf("Unexpected error: %v", err)
				}
			}

			assert.Equal(t, tc.expectedPass, successCount,
				"Expected %d successful requests, got %d", tc.expectedPass, successCount)

			if tc.limitAfter > 0 {
				expectedRateLimited := tc.totalRequests - tc.expectedPass
				assert.Equal(t, expectedRateLimited, rateLimitCount,
					"Expected %d rate limited requests, got %d", expectedRateLimited, rateLimitCount)
			}
		})
	}
}

// TestCombinedFailures tests multiple failure modes simultaneously.
func TestCombinedFailures(t *testing.T) {
	dataset := GenerateSampleBenchmarkDataset(50, 999)
	ctx := context.Background()

	t.Run("Timeout + Network Failure", func(t *testing.T) {
		client := NewBenchmarkMockLLMClient("combined-test-1", dataset, AnalyticalJudge)
		client.SimulateTimeout(50 * time.Millisecond)
		client.SimulateNetworkFailure(0.3) // 30% network failure

		errors := make(map[string]int)

		for i := 0; i < 20; i++ {
			prompt := "Rate this answer: Question: Test? Answer: Test"
			_, err := client.Complete(ctx, prompt, nil)

			if err != nil {
				if strings.Contains(err.Error(), "timeout") {
					errors["timeout"]++
				} else if strings.Contains(err.Error(), "network") {
					errors["network"]++
				} else {
					errors["other"]++
				}
			}
		}

		// Should see both types of errors
		t.Logf("Errors: %+v", errors)
		assert.Greater(t, errors["timeout"]+errors["network"], 0, "Expected some failures")
	})

	t.Run("Rate Limit + Partial Response", func(t *testing.T) {
		client := NewBenchmarkMockLLMClient("combined-test-2", dataset, ComprehensiveJudge)
		client.SimulateRateLimiting(3)
		client.SimulatePartialResponse()

		validJSONCount := 0
		rateLimitCount := 0
		partialCount := 0

		for i := 0; i < 6; i++ {
			prompt := "Rate this answer: Question: Test? Answer: Test"
			response, err := client.Complete(ctx, prompt, nil)

			if err != nil && strings.Contains(err.Error(), "rate limit") {
				rateLimitCount++
			} else if err == nil {
				// Check if response is partial
				var result map[string]any
				if jsonErr := json.Unmarshal([]byte(response), &result); jsonErr != nil {
					partialCount++
				} else {
					validJSONCount++
				}
			}
		}

		assert.Equal(t, 3, rateLimitCount, "Expected 3 rate limited requests")
		assert.Equal(t, 3, partialCount, "Expected 3 partial responses")
		assert.Equal(t, 0, validJSONCount, "Expected no valid JSON due to partial responses")
	})
}

// TestFailureRecovery tests that failures can be recovered from.
func TestFailureRecovery(t *testing.T) {
	dataset := GenerateSampleBenchmarkDataset(10, 555)
	ctx := context.Background()

	client := NewBenchmarkMockLLMClient("recovery-test", dataset, ConservativeJudge)
	prompt := "Rate this answer: Question: What is 2+2? Answer: 4"

	// First, verify normal operation
	response, err := client.Complete(ctx, prompt, nil)
	require.NoError(t, err)
	assert.NotEmpty(t, response)

	// Enable failures
	client.SimulateTimeout(10 * time.Millisecond)
	client.SimulateNetworkFailure(1.0) // 100% failure
	client.SimulateMalformedJSON()

	// Verify failures are active
	_, err = client.Complete(ctx, prompt, nil)
	assert.Error(t, err)

	// Reset failures
	client.ResetFailureSimulation()

	// Verify normal operation is restored
	response, err = client.Complete(ctx, prompt, nil)
	require.NoError(t, err)
	assert.NotEmpty(t, response)

	// Verify valid JSON
	var result map[string]any
	err = json.Unmarshal([]byte(response), &result)
	require.NoError(t, err)
	assert.Contains(t, result, "score")
	assert.Contains(t, result, "confidence")
}

// TestAdversarialInputsWithFailures tests adversarial inputs combined with failures.
func TestAdversarialInputsWithFailures(t *testing.T) {
	adversarialDataset := GenerateAdversarialDataset()
	ctx := context.Background()

	t.Run("Adversarial with Network Failures", func(t *testing.T) {
		client := NewBenchmarkMockLLMClient("adversarial-failure", adversarialDataset, BiasedJudge)
		client.SimulateNetworkFailure(0.2) // 20% failure rate

		successCount := 0
		failureCount := 0

		for _, question := range AdversarialQuestions[:5] { // Test first 5 adversarial questions
			for _, answer := range question.Answers {
				prompt := fmt.Sprintf(
					"Rate this answer: Question: %s Answer: %s",
					question.Question,
					answer.Content,
				)

				response, err := client.Complete(ctx, prompt, nil)
				if err != nil {
					failureCount++
					assert.Contains(t, err.Error(), "network")
				} else {
					successCount++
					// Even with adversarial input, valid responses should be returned
					assert.NotEmpty(t, response)
				}
			}
		}

		t.Logf("Adversarial tests: %d succeeded, %d failed", successCount, failureCount)
		assert.Greater(t, successCount, 0, "Expected some successful adversarial tests")
		assert.Greater(t, failureCount, 0, "Expected some network failures")
	})

	t.Run("Extremely Long Input with Timeout", func(t *testing.T) {
		client := NewBenchmarkMockLLMClient("long-input-timeout", adversarialDataset, AnalyticalJudge)
		client.SimulateTimeout(100 * time.Millisecond)

		// Find the extremely long input test case
		var longQuestion BenchmarkQuestion
		for _, q := range AdversarialQuestions {
			if q.ID == "adv3" { // The extremely long input test
				longQuestion = q
				break
			}
		}

		prompt := fmt.Sprintf(
			"Rate this answer: Question: %s Answer: %s",
			longQuestion.Question,           // This is 10,000 'A's
			longQuestion.Answers[1].Content, // This is 5,000 'B's
		)

		start := time.Now()
		_, err := client.Complete(ctx, prompt, nil)
		elapsed := time.Since(start)

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "timeout")
		assert.GreaterOrEqual(t, elapsed, 100*time.Millisecond)
	})
}
