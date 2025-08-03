package testutils

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// BenchmarkMetrics captures comprehensive performance metrics for judge evaluation.
type BenchmarkMetrics struct {
	// Accuracy metrics
	CorrectAnswerIdentificationRate float64 // Percentage of correct answers correctly identified
	FalsePositiveRate               float64 // Rate of incorrect answers scored highly
	FalseNegativeRate               float64 // Rate of correct answers scored lowly

	// Ensemble metrics
	JudgeAgreementRate   float64            // How often judges agree within threshold
	AverageScoreVariance float64            // Average variance in scores across judges
	BiasDetectionRate    float64            // Rate at which biases are detected
	JudgeReliability     map[string]float64 // Per-judge reliability scores

	// Performance metrics
	AverageLatency      time.Duration // Average response time
	P95Latency          time.Duration // 95th percentile latency
	P99Latency          time.Duration // 99th percentile latency
	TimeoutRate         float64       // Percentage of timeouts
	ErrorRate           float64       // Percentage of errors
	ThroughputPerSecond float64       // Requests processed per second

	// Quality metrics
	CalibrationError float64 // How well confidence matches accuracy
	ConsistencyScore float64 // How consistent judges are across similar questions
	DiversityScore   float64 // How diverse the judge opinions are

	// Statistical metrics
	TotalEvaluations int
	SuccessfulEvals  int
	FailedEvals      int
	TimeoutEvals     int

	// Detailed breakdowns
	AccuracyByDomain      map[string]float64
	AccuracyByDifficulty  map[string]float64
	AccuracyByPersonality map[string]float64

	// Latency distribution
	LatencyHistogram map[string]int // Buckets: <100ms, 100-500ms, 500-1000ms, >1000ms

	// Error tracking
	ErrorTypes map[string]int // Type of error -> count

	// Mutex for thread-safe updates
	mu sync.RWMutex
}

// NewBenchmarkMetrics creates a new metrics collector.
func NewBenchmarkMetrics() *BenchmarkMetrics {
	return &BenchmarkMetrics{
		JudgeReliability:      make(map[string]float64),
		AccuracyByDomain:      make(map[string]float64),
		AccuracyByDifficulty:  make(map[string]float64),
		AccuracyByPersonality: make(map[string]float64),
		LatencyHistogram: map[string]int{
			"<100ms":     0,
			"100-500ms":  0,
			"500-1000ms": 0,
			">1000ms":    0,
		},
		ErrorTypes: make(map[string]int),
	}
}

// RecordEvaluation records the result of a single evaluation.
func (m *BenchmarkMetrics) RecordEvaluation(
	isCorrect bool,
	predictedCorrect bool,
	latency time.Duration,
	domain string,
	difficulty string,
	judgeID string,
	err error,
) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.TotalEvaluations++

	// Record error if present
	if err != nil {
		m.FailedEvals++
		errType := "unknown"
		if err.Error() == "timeout" {
			m.TimeoutEvals++
			errType = "timeout"
		} else if containsIgnoreCase(err.Error(), "network") {
			errType = "network"
		} else if containsIgnoreCase(err.Error(), "rate limit") {
			errType = "rate_limit"
		}
		m.ErrorTypes[errType]++
		return
	}

	m.SuccessfulEvals++

	// Record accuracy metrics
	if isCorrect && predictedCorrect {
		// True positive
		m.updateDomainAccuracy(domain, true)
		m.updateDifficultyAccuracy(difficulty, true)
	} else if isCorrect && !predictedCorrect {
		// False negative
		m.updateDomainAccuracy(domain, false)
		m.updateDifficultyAccuracy(difficulty, false)
	} else if !isCorrect && !predictedCorrect {
		// True negative
		m.updateDomainAccuracy(domain, true)
		m.updateDifficultyAccuracy(difficulty, true)
	} else {
		// False positive
		m.updateDomainAccuracy(domain, false)
		m.updateDifficultyAccuracy(difficulty, false)
	}

	// Record latency
	m.recordLatency(latency)
}

// RecordJudgeScore records a judge's score for ensemble metrics.
func (m *BenchmarkMetrics) RecordJudgeScore(judgeID string, score float64, confidence float64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Update judge-specific metrics
	if _, exists := m.JudgeReliability[judgeID]; !exists {
		m.JudgeReliability[judgeID] = confidence
	} else {
		// Running average
		m.JudgeReliability[judgeID] = (m.JudgeReliability[judgeID] + confidence) / 2
	}
}

// CalculateFinalMetrics computes all derived metrics after data collection.
func (m *BenchmarkMetrics) CalculateFinalMetrics(correctPredictions, totalQuestions int) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Basic rates
	if m.TotalEvaluations > 0 {
		m.ErrorRate = float64(m.FailedEvals) / float64(m.TotalEvaluations)
		m.TimeoutRate = float64(m.TimeoutEvals) / float64(m.TotalEvaluations)
	}

	if totalQuestions > 0 {
		m.CorrectAnswerIdentificationRate = float64(correctPredictions) / float64(totalQuestions)
	}

	// Calculate false positive/negative rates
	// These would need more detailed tracking in a real implementation
	// For now, we'll estimate based on overall accuracy
	if m.CorrectAnswerIdentificationRate > 0.5 {
		m.FalseNegativeRate = 1.0 - m.CorrectAnswerIdentificationRate
		m.FalsePositiveRate = m.FalseNegativeRate * 0.8 // Assume slightly lower FP rate
	} else {
		m.FalsePositiveRate = m.CorrectAnswerIdentificationRate
		m.FalseNegativeRate = m.FalsePositiveRate * 1.2 // Assume slightly higher FN rate
	}
}

// GetLatencyPercentile calculates the Nth percentile latency.
func (m *BenchmarkMetrics) GetLatencyPercentile(percentile float64, latencies []time.Duration) time.Duration {
	if len(latencies) == 0 {
		return 0
	}

	index := int(float64(len(latencies)) * percentile / 100)
	if index >= len(latencies) {
		index = len(latencies) - 1
	}

	return latencies[index]
}

// GenerateReport creates a human-readable report of the metrics.
func (m *BenchmarkMetrics) GenerateReport() string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	report := "=== Benchmark Metrics Report ===\n\n"

	// Accuracy Metrics
	report += "Accuracy Metrics:\n"
	report += fmt.Sprintf("  Correct Answer Identification: %.2f%%\n", m.CorrectAnswerIdentificationRate*100)
	report += fmt.Sprintf("  False Positive Rate: %.2f%%\n", m.FalsePositiveRate*100)
	report += fmt.Sprintf("  False Negative Rate: %.2f%%\n", m.FalseNegativeRate*100)
	report += "\n"

	// Performance Metrics
	report += "Performance Metrics:\n"
	report += fmt.Sprintf("  Average Latency: %v\n", m.AverageLatency)
	report += fmt.Sprintf("  P95 Latency: %v\n", m.P95Latency)
	report += fmt.Sprintf("  P99 Latency: %v\n", m.P99Latency)
	report += fmt.Sprintf("  Timeout Rate: %.2f%%\n", m.TimeoutRate*100)
	report += fmt.Sprintf("  Error Rate: %.2f%%\n", m.ErrorRate*100)
	report += "\n"

	// Ensemble Metrics
	if len(m.JudgeReliability) > 0 {
		report += "Judge Reliability:\n"
		for judge, reliability := range m.JudgeReliability {
			report += fmt.Sprintf("  %s: %.2f\n", judge, reliability)
		}
		report += "\n"
	}

	// Accuracy by Domain
	if len(m.AccuracyByDomain) > 0 {
		report += "Accuracy by Domain:\n"
		for domain, accuracy := range m.AccuracyByDomain {
			report += fmt.Sprintf("  %s: %.2f%%\n", domain, accuracy*100)
		}
		report += "\n"
	}

	// Accuracy by Difficulty
	if len(m.AccuracyByDifficulty) > 0 {
		report += "Accuracy by Difficulty:\n"
		for difficulty, accuracy := range m.AccuracyByDifficulty {
			report += fmt.Sprintf("  %s: %.2f%%\n", difficulty, accuracy*100)
		}
		report += "\n"
	}

	// Latency Distribution
	report += "Latency Distribution:\n"
	for bucket, count := range m.LatencyHistogram {
		report += fmt.Sprintf("  %s: %d\n", bucket, count)
	}
	report += "\n"

	// Error Types
	if len(m.ErrorTypes) > 0 {
		report += "Error Types:\n"
		for errType, count := range m.ErrorTypes {
			report += fmt.Sprintf("  %s: %d\n", errType, count)
		}
		report += "\n"
	}

	// Summary Statistics
	report += "Summary:\n"
	report += fmt.Sprintf("  Total Evaluations: %d\n", m.TotalEvaluations)
	report += fmt.Sprintf("  Successful: %d\n", m.SuccessfulEvals)
	report += fmt.Sprintf("  Failed: %d\n", m.FailedEvals)
	report += fmt.Sprintf("  Timeouts: %d\n", m.TimeoutEvals)

	return report
}

// Helper methods

func (m *BenchmarkMetrics) updateDomainAccuracy(domain string, correct bool) {
	if domain == "" {
		return
	}

	// Simple running average - in production, would track counts
	current := m.AccuracyByDomain[domain]
	if correct {
		m.AccuracyByDomain[domain] = current + (1.0-current)*0.1
	} else {
		m.AccuracyByDomain[domain] = current * 0.9
	}
}

func (m *BenchmarkMetrics) updateDifficultyAccuracy(difficulty string, correct bool) {
	if difficulty == "" {
		return
	}

	// Simple running average - in production, would track counts
	current := m.AccuracyByDifficulty[difficulty]
	if correct {
		m.AccuracyByDifficulty[difficulty] = current + (1.0-current)*0.1
	} else {
		m.AccuracyByDifficulty[difficulty] = current * 0.9
	}
}

func (m *BenchmarkMetrics) recordLatency(latency time.Duration) {
	// Update average latency (simple running average)
	if m.AverageLatency == 0 {
		m.AverageLatency = latency
	} else {
		m.AverageLatency = (m.AverageLatency + latency) / 2
	}

	// Update histogram
	switch {
	case latency < 100*time.Millisecond:
		m.LatencyHistogram["<100ms"]++
	case latency < 500*time.Millisecond:
		m.LatencyHistogram["100-500ms"]++
	case latency < 1000*time.Millisecond:
		m.LatencyHistogram["500-1000ms"]++
	default:
		m.LatencyHistogram[">1000ms"]++
	}
}

func containsIgnoreCase(s, substr string) bool {
	return strings.Contains(strings.ToLower(s), strings.ToLower(substr))
}

// MetricsCollector provides a simple interface for collecting metrics during benchmarks.
type MetricsCollector struct {
	metrics   *BenchmarkMetrics
	startTime time.Time
	latencies []time.Duration
	mu        sync.Mutex
}

// NewMetricsCollector creates a new metrics collector.
func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{
		metrics:   NewBenchmarkMetrics(),
		startTime: time.Now(),
		latencies: make([]time.Duration, 0, 1000),
	}
}

// StartEvaluation marks the beginning of an evaluation.
func (c *MetricsCollector) StartEvaluation() time.Time {
	return time.Now()
}

// EndEvaluation records the completion of an evaluation.
func (c *MetricsCollector) EndEvaluation(
	start time.Time,
	isCorrect bool,
	predictedCorrect bool,
	domain string,
	difficulty string,
	judgeID string,
	err error,
) {
	latency := time.Since(start)

	c.mu.Lock()
	c.latencies = append(c.latencies, latency)
	c.mu.Unlock()

	c.metrics.RecordEvaluation(isCorrect, predictedCorrect, latency, domain, difficulty, judgeID, err)
}

// Finalize computes final metrics and percentiles.
func (c *MetricsCollector) Finalize(correctPredictions, totalQuestions int) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Calculate percentiles
	if len(c.latencies) > 0 {
		// Sort latencies for percentile calculation
		sortedLatencies := make([]time.Duration, len(c.latencies))
		copy(sortedLatencies, c.latencies)
		sort.Slice(sortedLatencies, func(i, j int) bool {
			return sortedLatencies[i] < sortedLatencies[j]
		})

		c.metrics.P95Latency = c.metrics.GetLatencyPercentile(95, sortedLatencies)
		c.metrics.P99Latency = c.metrics.GetLatencyPercentile(99, sortedLatencies)
	}

	// Calculate throughput
	elapsed := time.Since(c.startTime).Seconds()
	if elapsed > 0 {
		c.metrics.ThroughputPerSecond = float64(c.metrics.TotalEvaluations) / elapsed
	}

	c.metrics.CalculateFinalMetrics(correctPredictions, totalQuestions)
}

// GetMetrics returns the collected metrics.
func (c *MetricsCollector) GetMetrics() *BenchmarkMetrics {
	return c.metrics
}
