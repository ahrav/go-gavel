package testutils

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestBenchmarkMockLLMClientThreadSafety verifies that the client is thread-safe.
func TestBenchmarkMockLLMClientThreadSafety(t *testing.T) {
	// Create a test dataset
	dataset := GenerateSampleBenchmarkDataset(100, 42)

	// Create clients with different personalities
	clients := map[string]*BenchmarkMockLLMClient{
		"conservative":  NewBenchmarkMockLLMClient("test-conservative", dataset, ConservativeJudge),
		"comprehensive": NewBenchmarkMockLLMClient("test-comprehensive", dataset, ComprehensiveJudge),
		"analytical":    NewBenchmarkMockLLMClient("test-analytical", dataset, AnalyticalJudge),
		"biased":        NewBiasedBenchmarkMockLLMClient("test-biased", dataset, 0.3),
	}

	ctx := context.Background()

	// Test concurrent access to each client
	for name, client := range clients {
		t.Run(name, func(t *testing.T) {
			const numGoroutines = 50
			const requestsPerGoroutine = 20

			var wg sync.WaitGroup
			errors := make(chan error, numGoroutines*requestsPerGoroutine)

			// Launch concurrent goroutines
			for i := range numGoroutines {
				wg.Add(1)
				go func(goroutineID int) {
					defer wg.Done()

					for j := range requestsPerGoroutine {
						// Use different questions to ensure variety
						questionIdx := (goroutineID*requestsPerGoroutine + j) % len(dataset.Questions)
						question := dataset.Questions[questionIdx]

						// Create a scoring prompt
						prompt := fmt.Sprintf(
							"Rate this answer to the question on a scale from 0.0 to 1.0:\nQuestion: %s\nAnswer: %s\n",
							question.Question,
							question.Answers[j%len(question.Answers)].Content,
						)

						// Execute the request
						response, err := client.Complete(ctx, prompt, nil)
						if err != nil {
							errors <- fmt.Errorf("goroutine %d request %d: %w", goroutineID, j, err)
							continue
						}

						// Validate the response
						var result map[string]any
						if err := json.Unmarshal([]byte(response), &result); err != nil {
							errors <- fmt.Errorf("goroutine %d request %d: invalid JSON: %w", goroutineID, j, err)
							continue
						}

						// Check required fields
						if _, ok := result["score"]; !ok {
							errors <- fmt.Errorf("goroutine %d request %d: missing score field", goroutineID, j)
						}
						if _, ok := result["confidence"]; !ok {
							errors <- fmt.Errorf("goroutine %d request %d: missing confidence field", goroutineID, j)
						}
					}
				}(i)
			}

			// Wait for all goroutines to complete
			wg.Wait()
			close(errors)

			// Check for errors
			var errorCount int
			for err := range errors {
				t.Errorf("Concurrent access error: %v", err)
				errorCount++
				if errorCount > 10 {
					t.Fatal("Too many errors, stopping test")
				}
			}

			assert.Equal(t, 0, errorCount, "Expected no errors during concurrent access")
		})
	}
}

// TestBenchmarkMockLLMClientRaceDetection uses Go's race detector to find race conditions.
func TestBenchmarkMockLLMClientRaceDetection(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping race detection test in short mode")
	}

	dataset := GenerateSampleBenchmarkDataset(50, 12345)
	client := NewBenchmarkMockLLMClient("race-test", dataset, ComprehensiveJudge)

	ctx := context.Background()

	// Perform operations that might cause races
	var wg sync.WaitGroup

	// Concurrent reads and writes to various fields
	operations := []func(){
		// Complete requests
		func() {
			prompt := "Rate this answer: Question: Test? Answer: Test answer"
			_, _ = client.Complete(ctx, prompt, nil)
		},
		// Modify configuration
		func() {
			config := DefaultJudgeConfig()
			config.NoiseFactor = 0.15
			client.config = config
		},
		// Set ensemble config
		func() {
			ensembleConfig := DefaultEnsembleConfig()
			client.SetEnsembleConfig(&ensembleConfig)
		},
		// Simulate failures
		func() {
			client.SimulateTimeout(100 * time.Millisecond)
			client.ResetFailureSimulation()
		},
		// Shared error state
		func() {
			sharedState := make(map[string]float64)
			sharedState["q1"] = 0.1
			client.SetSharedErrorState(sharedState)
		},
	}

	// Run operations concurrently
	const iterations = 100
	for _, op := range operations {
		wg.Add(1)
		go func(operation func()) {
			defer wg.Done()
			for range iterations {
				operation()
			}
		}(op)
	}

	wg.Wait()

	// If we get here without race detector complaints, the test passes
	t.Log("No race conditions detected")
}

// TestCorrelatedErrorsConcurrency tests that correlated errors work correctly with concurrent access.
func TestCorrelatedErrorsConcurrency(t *testing.T) {
	dataset := GenerateSampleBenchmarkDataset(100, 99999)

	// Create ensemble config with correlation
	ensembleConfig := EnsembleConfig{
		ErrorCorrelation: 0.7,
		SharedErrorSeed:  42,
	}

	// Create correlated ensemble
	mocks := CreateCorrelatedBenchmarkEnsembleMocks(dataset, ensembleConfig)

	ctx := context.Background()
	var wg sync.WaitGroup
	results := make(map[string][]float64)
	var mu sync.Mutex

	// Each judge evaluates the same questions concurrently
	for judgeName, judge := range mocks {
		wg.Add(1)
		go func(name string, client *BenchmarkMockLLMClient) {
			defer wg.Done()

			scores := make([]float64, 0, 10)

			// Evaluate first 10 questions
			for i := 0; i < 10 && i < len(dataset.Questions); i++ {
				question := dataset.Questions[i]
				prompt := fmt.Sprintf(
					"Rate this answer: Question: %s Answer: %s",
					question.Question,
					question.Answers[0].Content,
				)

				response, err := client.Complete(ctx, prompt, nil)
				require.NoError(t, err)

				var result map[string]any
				err = json.Unmarshal([]byte(response), &result)
				require.NoError(t, err)

				score := result["score"].(float64)
				scores = append(scores, score)
			}

			mu.Lock()
			results[name] = scores
			mu.Unlock()
		}(judgeName, judge)
	}

	wg.Wait()

	// Verify that scores show correlation
	// With 0.7 correlation, judges should often make similar errors
	correlatedCount := 0
	totalComparisons := 0

	judges := []string{"conservative", "comprehensive", "analytical", "biased"}
	for i := range 10 {
		for j := 0; j < len(judges)-1; j++ {
			for k := j + 1; k < len(judges); k++ {
				score1 := results[judges[j]][i]
				score2 := results[judges[k]][i]

				diff := abs(score1 - score2)
				if diff < 0.2 { // Similar scores
					correlatedCount++
				}
				totalComparisons++
			}
		}
	}

	correlationRate := float64(correlatedCount) / float64(totalComparisons)
	t.Logf("Correlation rate: %.2f (expected around %.2f)", correlationRate, ensembleConfig.ErrorCorrelation)

	// With 70% correlation, we expect at least 50% of comparisons to be similar
	assert.Greater(t, correlationRate, 0.5, "Expected judges to show correlated errors")
}

// TestCatastrophicFailuresConcurrency tests failure scenarios under concurrent load.
func TestCatastrophicFailuresConcurrency(t *testing.T) {
	dataset := GenerateSampleBenchmarkDataset(50, 11111)

	testCases := []struct {
		name          string
		setupFailure  func(*BenchmarkMockLLMClient)
		expectedError string
	}{
		{
			name: "Timeout",
			setupFailure: func(c *BenchmarkMockLLMClient) {
				c.SimulateTimeout(50 * time.Millisecond)
			},
			expectedError: "timeout",
		},
		{
			name: "Network Failure",
			setupFailure: func(c *BenchmarkMockLLMClient) {
				c.SimulateNetworkFailure(0.5) // 50% failure rate
			},
			expectedError: "network error",
		},
		{
			name: "Rate Limiting",
			setupFailure: func(c *BenchmarkMockLLMClient) {
				c.SimulateRateLimiting(10)
			},
			expectedError: "rate limit",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client := NewBenchmarkMockLLMClient("failure-test", dataset, AnalyticalJudge)
			tc.setupFailure(client)

			ctx := context.Background()
			var wg sync.WaitGroup
			errorCount := 0
			var errorMu sync.Mutex

			// Launch concurrent requests
			const numGoroutines = 20
			const requestsPerGoroutine = 5

			for range numGoroutines {
				wg.Add(1)
				go func() {
					defer wg.Done()

					for range requestsPerGoroutine {
						prompt := "Rate this answer: Question: Test? Answer: Test"
						_, err := client.Complete(ctx, prompt, nil)

						if err != nil {
							errorMu.Lock()
							errorCount++
							errorMu.Unlock()

							// Verify error message
							assert.Contains(t, err.Error(), tc.expectedError)
						}
					}
				}()
			}

			wg.Wait()

			// For network failures and rate limiting, we expect some errors
			if tc.name != "Timeout" {
				assert.Greater(t, errorCount, 0, "Expected some failures for %s", tc.name)
			} else {
				// For timeout, all requests should fail
				assert.Equal(t, numGoroutines*requestsPerGoroutine, errorCount)
			}
		})
	}
}

// Benchmark for concurrent performance
func BenchmarkConcurrentComplete(b *testing.B) {
	dataset := GenerateSampleBenchmarkDataset(100, 54321)
	client := NewBenchmarkMockLLMClient("benchmark", dataset, ComprehensiveJudge)

	ctx := context.Background()
	prompt := "Rate this answer: Question: What is 2+2? Answer: 4"

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, err := client.Complete(ctx, prompt, nil)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}
