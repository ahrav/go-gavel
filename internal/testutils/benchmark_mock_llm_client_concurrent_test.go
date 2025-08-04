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

// TestBenchmarkMockLLMClientThreadSafety verifies that the mock LLM client is thread-safe.
// It launches multiple goroutines to make concurrent requests to the client and checks for errors.
func TestBenchmarkMockLLMClientThreadSafety(t *testing.T) {
	dataset := GenerateSampleBenchmarkDataset(100, 42)

	clients := map[string]*BenchmarkMockLLMClient{
		"conservative":  NewBenchmarkMockLLMClient("test-conservative", dataset, ConservativeJudge),
		"comprehensive": NewBenchmarkMockLLMClient("test-comprehensive", dataset, ComprehensiveJudge),
		"analytical":    NewBenchmarkMockLLMClient("test-analytical", dataset, AnalyticalJudge),
		"biased":        NewBiasedBenchmarkMockLLMClient("test-biased", dataset, 0.3),
	}

	ctx := context.Background()

	for name, client := range clients {
		t.Run(name, func(t *testing.T) {
			const numGoroutines = 50
			const requestsPerGoroutine = 20

			var wg sync.WaitGroup
			errors := make(chan error, numGoroutines*requestsPerGoroutine)

			for i := range numGoroutines {
				wg.Add(1)
				go func(goroutineID int) {
					defer wg.Done()

					for j := range requestsPerGoroutine {
						// Use a variety of questions to avoid cache hits.
						questionIdx := (goroutineID*requestsPerGoroutine + j) % len(dataset.Questions)
						question := dataset.Questions[questionIdx]

						prompt := fmt.Sprintf(
							"Rate this answer to the question on a scale from 0.0 to 1.0:\nQuestion: %s\nAnswer: %s\n",
							question.Question,
							question.Answers[j%len(question.Answers)].Content,
						)

						response, err := client.Complete(ctx, prompt, nil)
						if err != nil {
							errors <- fmt.Errorf("goroutine %d request %d: %w", goroutineID, j, err)
							continue
						}

						var result map[string]any
						if err := json.Unmarshal([]byte(response), &result); err != nil {
							errors <- fmt.Errorf("goroutine %d request %d: invalid JSON: %w", goroutineID, j, err)
							continue
						}

						if _, ok := result["score"]; !ok {
							errors <- fmt.Errorf("goroutine %d request %d: missing score field", goroutineID, j)
						}
						if _, ok := result["confidence"]; !ok {
							errors <- fmt.Errorf("goroutine %d request %d: missing confidence field", goroutineID, j)
						}
					}
				}(i)
			}

			wg.Wait()
			close(errors)

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

// TestBenchmarkMockLLMClientRaceDetection uses Go's race detector to identify potential race conditions.
// It runs various client operations concurrently to stress test thread safety.
func TestBenchmarkMockLLMClientRaceDetection(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping race detection test in short mode")
	}

	dataset := GenerateSampleBenchmarkDataset(50, 12345)
	client := NewBenchmarkMockLLMClient("race-test", dataset, ComprehensiveJudge)

	ctx := context.Background()

	var wg sync.WaitGroup

	operations := []func(){
		func() {
			prompt := "Rate this answer: Question: Test? Answer: Test answer"
			_, _ = client.Complete(ctx, prompt, nil)
		},
		func() {
			config := DefaultJudgeConfig()
			config.NoiseFactor = 0.15
			client.SetConfig(config)
		},
		func() {
			ensembleConfig := DefaultEnsembleConfig()
			client.SetEnsembleConfig(&ensembleConfig)
		},
		func() {
			client.SimulateTimeout(100 * time.Millisecond)
			client.ResetFailureSimulation()
		},
		func() {
			sharedState := make(map[string]float64)
			sharedState["q1"] = 0.1
			client.SetSharedErrorState(sharedState)
		},
	}

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

	// If the race detector does not report any issues, the test passes.
	t.Log("No race conditions detected")
}

// TestCorrelatedErrorsConcurrency verifies that correlated errors behave as expected under concurrent access.
// It ensures that judges in an ensemble produce similar errors when configured with a high correlation factor.
func TestCorrelatedErrorsConcurrency(t *testing.T) {
	dataset := GenerateSampleBenchmarkDataset(100, 99999)

	ensembleConfig := EnsembleConfig{
		ErrorCorrelation: 0.7,
		SharedErrorSeed:  42,
	}

	mocks := CreateCorrelatedBenchmarkEnsembleMocks(dataset, ensembleConfig)

	ctx := context.Background()
	var wg sync.WaitGroup
	results := make(map[string][]float64)
	var mu sync.Mutex

	for judgeName, judge := range mocks {
		wg.Add(1)
		go func(name string, client *BenchmarkMockLLMClient) {
			defer wg.Done()

			scores := make([]float64, 0, 10)

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

	// With a high error correlation, judges should produce similar error patterns.
	correlatedCount := 0
	totalComparisons := 0

	judges := []string{"conservative", "comprehensive", "analytical", "biased"}
	for i := range 10 {
		for j := 0; j < len(judges)-1; j++ {
			for k := j + 1; k < len(judges); k++ {
				score1 := results[judges[j]][i]
				score2 := results[judges[k]][i]

				diff := abs(score1 - score2)
				if diff < 0.2 {
					correlatedCount++
				}
				totalComparisons++
			}
		}
	}

	correlationRate := float64(correlatedCount) / float64(totalComparisons)
	t.Logf("Correlation rate: %.2f (expected around %.2f)", correlationRate, ensembleConfig.ErrorCorrelation)

	// A 70% correlation should result in at least 50% of comparisons being similar.
	assert.Greater(t, correlationRate, 0.5, "Expected judges to show correlated errors")
}

// TestCatastrophicFailuresConcurrency tests how the mock client handles catastrophic failures under concurrent load.
// It covers timeouts, network failures, and rate limiting.
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

			// Network failures and rate limiting should produce some errors, but not all.
			if tc.name != "Timeout" {
				assert.Greater(t, errorCount, 0, "Expected some failures for %s", tc.name)
			} else {
				// A timeout should cause all requests to fail.
				assert.Equal(t, numGoroutines*requestsPerGoroutine, errorCount)
			}
		})
	}
}

// BenchmarkConcurrentComplete benchmarks the performance of the mock client under concurrent load.
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
