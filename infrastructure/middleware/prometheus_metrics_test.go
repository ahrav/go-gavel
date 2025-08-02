package middleware

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ahrav/go-gavel/internal/ports"
)

// Global test instance to avoid duplicate metric registration issues
var testPrometheusMetrics *PrometheusMetrics

func init() {
	// Create a single instance for all tests to avoid registration conflicts
	testPrometheusMetrics = NewPrometheusMetrics()
}

func TestNewPrometheusMetrics(t *testing.T) {
	// Use the global test instance to avoid duplicate registrations
	pm := testPrometheusMetrics

	// Verify that the instance is not nil
	assert.NotNil(t, pm, "PrometheusMetrics instance should not be nil")

	// Verify that all metric fields are properly initialized
	assert.NotNil(t, pm.budgetTokensUsed, "budgetTokensUsed should be initialized")
	assert.NotNil(t, pm.budgetCallsUsed, "budgetCallsUsed should be initialized")
	assert.NotNil(t, pm.costPerEvaluation, "costPerEvaluation should be initialized")
	assert.NotNil(t, pm.executionLatency, "executionLatency should be initialized")
	assert.NotNil(t, pm.operationCounter, "operationCounter should be initialized")
	assert.NotNil(t, pm.systemGauges, "systemGauges should be initialized")

	// Verify that PrometheusMetrics implements MetricsCollector interface
	var _ ports.MetricsCollector = pm
}

func TestPrometheusMetrics_RecordLatency(t *testing.T) {
	pm := testPrometheusMetrics

	tests := []struct {
		name      string
		operation string
		duration  time.Duration
		labels    map[string]string
		wantUnit  string
	}{
		{
			name:      "record latency with unit label",
			operation: "test_operation",
			duration:  100 * time.Millisecond,
			labels:    map[string]string{"unit": "test-unit"},
			wantUnit:  "test-unit",
		},
		{
			name:      "record latency without unit label",
			operation: "another_operation",
			duration:  250 * time.Millisecond,
			labels:    map[string]string{"other": "value"},
			wantUnit:  "unknown",
		},
		{
			name:      "record latency with empty unit label",
			operation: "empty_unit_operation",
			duration:  50 * time.Millisecond,
			labels:    map[string]string{"unit": ""},
			wantUnit:  "unknown",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Record the latency
			pm.RecordLatency(tt.operation, tt.duration, tt.labels)

			// Verify that the metric was recorded
			// Note: In a real test environment, you might want to check the actual metric value
			// using prometheus testutil package, but that requires more complex setup
			assert.NotPanics(t, func() {
				pm.RecordLatency(tt.operation, tt.duration, tt.labels)
			}, "RecordLatency should not panic")
		})
	}
}

func TestPrometheusMetrics_RecordCounter(t *testing.T) {
	pm := testPrometheusMetrics

	tests := []struct {
		name       string
		metric     string
		value      float64
		labels     map[string]string
		shouldWork bool
	}{
		{
			name:   "record budget tokens used",
			metric: "budget_tokens_used",
			value:  100.0,
			labels: map[string]string{
				"graph_id":        "test-graph",
				"evaluation_type": "test-eval",
				"budget_limit":    "tokens_only",
				"unit":            "test-unit",
			},
			shouldWork: true,
		},
		{
			name:   "record budget calls used",
			metric: "budget_calls_used",
			value:  5.0,
			labels: map[string]string{
				"graph_id":        "test-graph",
				"evaluation_type": "test-eval",
				"budget_limit":    "calls_only",
				"unit":            "test-unit",
			},
			shouldWork: true,
		},
		{
			name:   "record budget exceeded",
			metric: "budget_exceeded_total",
			value:  1.0,
			labels: map[string]string{
				"limit_type": "tokens",
				"unit":       "test-unit",
			},
			shouldWork: true,
		},
		{
			name:   "record unknown metric",
			metric: "unknown_metric",
			value:  42.0,
			labels: map[string]string{
				"unit": "test-unit",
			},
			shouldWork: true,
		},
		{
			name:   "record with missing unit label",
			metric: "budget_tokens_used",
			value:  50.0,
			labels: map[string]string{
				"graph_id":        "test-graph",
				"evaluation_type": "test-eval",
				"budget_limit":    "tokens_only",
			},
			shouldWork: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.shouldWork {
				assert.NotPanics(t, func() {
					pm.RecordCounter(tt.metric, tt.value, tt.labels)
				}, "RecordCounter should not panic for valid inputs")
			}
		})
	}
}

func TestPrometheusMetrics_RecordGauge(t *testing.T) {
	pm := testPrometheusMetrics

	tests := []struct {
		name       string
		metric     string
		value      float64
		labels     map[string]string
		shouldWork bool
	}{
		{
			name:   "record cost per evaluation",
			metric: "cost_per_evaluation",
			value:  0.05,
			labels: map[string]string{
				"graph_id":        "test-graph",
				"evaluation_type": "test-eval",
				"budget_limit":    "tokens_and_calls",
				"unit":            "test-unit",
			},
			shouldWork: true,
		},
		{
			name:   "record budget tokens used gauge",
			metric: "budget_tokens_used",
			value:  150.0,
			labels: map[string]string{
				"unit": "test-unit",
			},
			shouldWork: true,
		},
		{
			name:   "record budget remaining tokens",
			metric: "budget_remaining_tokens",
			value:  850.0,
			labels: map[string]string{
				"unit": "test-unit",
			},
			shouldWork: true,
		},
		{
			name:   "record budget remaining calls",
			metric: "budget_remaining_calls",
			value:  15.0,
			labels: map[string]string{
				"unit": "test-unit",
			},
			shouldWork: true,
		},
		{
			name:   "record unknown gauge metric",
			metric: "unknown_gauge",
			value:  123.45,
			labels: map[string]string{
				"unit": "test-unit",
			},
			shouldWork: true,
		},
		{
			name:   "record with empty unit label",
			metric: "cost_per_evaluation",
			value:  0.03,
			labels: map[string]string{
				"graph_id":        "test-graph",
				"evaluation_type": "test-eval",
				"budget_limit":    "unlimited",
				"unit":            "",
			},
			shouldWork: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.shouldWork {
				assert.NotPanics(t, func() {
					pm.RecordGauge(tt.metric, tt.value, tt.labels)
				}, "RecordGauge should not panic for valid inputs")
			}
		})
	}
}

func TestPrometheusMetrics_RecordHistogram(t *testing.T) {
	pm := testPrometheusMetrics

	tests := []struct {
		name       string
		metric     string
		value      float64
		labels     map[string]string
		shouldWork bool
	}{
		{
			name:   "record histogram with unit",
			metric: "test_histogram",
			value:  0.123,
			labels: map[string]string{
				"unit": "test-unit",
			},
			shouldWork: true,
		},
		{
			name:   "record histogram without unit",
			metric: "another_histogram",
			value:  0.456,
			labels: map[string]string{
				"other": "value",
			},
			shouldWork: true,
		},
		{
			name:   "record histogram with empty unit",
			metric: "empty_unit_histogram",
			value:  0.789,
			labels: map[string]string{
				"unit": "",
			},
			shouldWork: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.shouldWork {
				assert.NotPanics(t, func() {
					pm.RecordHistogram(tt.metric, tt.value, tt.labels)
				}, "RecordHistogram should not panic for valid inputs")
			}
		})
	}
}

func TestPrometheusMetrics_LabelHandling(t *testing.T) {
	pm := testPrometheusMetrics

	tests := []struct {
		name         string
		labels       map[string]string
		expectedUnit string
	}{
		{
			name:         "nil labels map",
			labels:       nil,
			expectedUnit: "unknown",
		},
		{
			name:         "empty labels map",
			labels:       map[string]string{},
			expectedUnit: "unknown",
		},
		{
			name: "labels map with unit",
			labels: map[string]string{
				"unit": "test-unit",
			},
			expectedUnit: "test-unit",
		},
		{
			name: "labels map with empty unit",
			labels: map[string]string{
				"unit":  "",
				"other": "value",
			},
			expectedUnit: "unknown",
		},
		{
			name: "labels map without unit",
			labels: map[string]string{
				"graph_id": "test-graph",
				"other":    "value",
			},
			expectedUnit: "unknown",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test with RecordLatency to verify label handling
			assert.NotPanics(t, func() {
				pm.RecordLatency("test_operation", 100*time.Millisecond, tt.labels)
			}, "Should handle labels gracefully")

			// Test with RecordCounter to verify label handling
			assert.NotPanics(t, func() {
				pm.RecordCounter("test_counter", 1.0, tt.labels)
			}, "Should handle labels gracefully")

			// Test with RecordGauge to verify label handling
			assert.NotPanics(t, func() {
				pm.RecordGauge("test_gauge", 42.0, tt.labels)
			}, "Should handle labels gracefully")

			// Test with RecordHistogram to verify label handling
			assert.NotPanics(t, func() {
				pm.RecordHistogram("test_histogram", 0.5, tt.labels)
			}, "Should handle labels gracefully")
		})
	}
}

func TestPrometheusMetrics_InterfaceCompliance(t *testing.T) {
	// Verify that PrometheusMetrics implements the MetricsCollector interface
	var metrics ports.MetricsCollector = testPrometheusMetrics
	require.NotNil(t, metrics, "PrometheusMetrics should implement MetricsCollector interface")

	// Test that all interface methods can be called
	labels := map[string]string{"unit": "test-unit"}

	assert.NotPanics(t, func() {
		metrics.RecordLatency("test", 100*time.Millisecond, labels)
	}, "RecordLatency should be callable through interface")

	assert.NotPanics(t, func() {
		metrics.RecordCounter("test", 1.0, labels)
	}, "RecordCounter should be callable through interface")

	assert.NotPanics(t, func() {
		metrics.RecordGauge("test", 42.0, labels)
	}, "RecordGauge should be callable through interface")

	assert.NotPanics(t, func() {
		metrics.RecordHistogram("test", 0.5, labels)
	}, "RecordHistogram should be callable through interface")
}

func TestPrometheusMetrics_BudgetSpecificMetrics(t *testing.T) {
	pm := testPrometheusMetrics

	// Test budget-specific counter metrics
	budgetLabels := map[string]string{
		"graph_id":        "test-graph-123",
		"evaluation_type": "unit-test",
		"budget_limit":    "tokens_and_calls",
		"unit":            "budget-manager",
	}

	t.Run("budget tokens counter", func(t *testing.T) {
		assert.NotPanics(t, func() {
			pm.RecordCounter("budget_tokens_used", 500.0, budgetLabels)
		}, "Should record budget tokens counter without panic")
	})

	t.Run("budget calls counter", func(t *testing.T) {
		assert.NotPanics(t, func() {
			pm.RecordCounter("budget_calls_used", 10.0, budgetLabels)
		}, "Should record budget calls counter without panic")
	})

	t.Run("cost per evaluation gauge", func(t *testing.T) {
		assert.NotPanics(t, func() {
			pm.RecordGauge("cost_per_evaluation", 0.125, budgetLabels)
		}, "Should record cost per evaluation gauge without panic")
	})

	t.Run("budget exceeded counter", func(t *testing.T) {
		exceededLabels := map[string]string{
			"limit_type": "tokens",
			"unit":       "budget-manager",
		}
		assert.NotPanics(t, func() {
			pm.RecordCounter("budget_exceeded_total", 1.0, exceededLabels)
		}, "Should record budget exceeded counter without panic")
	})
}

func TestPrometheusMetrics_EdgeCases(t *testing.T) {
	pm := testPrometheusMetrics

	t.Run("zero duration latency", func(t *testing.T) {
		assert.NotPanics(t, func() {
			pm.RecordLatency("zero_duration", 0, map[string]string{"unit": "test"})
		}, "Should handle zero duration gracefully")
	})

	t.Run("negative counter value", func(t *testing.T) {
		// Prometheus counters cannot have negative values, so this should panic
		assert.Panics(t, func() {
			pm.RecordCounter("negative_counter", -1.0, map[string]string{"unit": "test"})
		}, "Prometheus counters should panic on negative values")
	})

	t.Run("very large gauge value", func(t *testing.T) {
		assert.NotPanics(t, func() {
			pm.RecordGauge("large_gauge", 1e9, map[string]string{"unit": "test"})
		}, "Should handle large gauge values gracefully")
	})

	t.Run("very small histogram value", func(t *testing.T) {
		assert.NotPanics(t, func() {
			pm.RecordHistogram("small_histogram", 1e-9, map[string]string{"unit": "test"})
		}, "Should handle very small histogram values gracefully")
	})

	t.Run("missing required labels", func(t *testing.T) {
		// Budget metrics should handle missing labels gracefully
		incompleteLabels := map[string]string{
			"graph_id": "test-graph",
			// Missing other required labels
		}
		assert.NotPanics(t, func() {
			pm.RecordCounter("budget_tokens_used", 100.0, incompleteLabels)
		}, "Should handle incomplete budget labels gracefully")
	})
}

// Benchmark tests to ensure performance is acceptable
func BenchmarkPrometheusMetrics_RecordLatency(b *testing.B) {
	pm := testPrometheusMetrics
	labels := map[string]string{"unit": "benchmark-test"}
	duration := 100 * time.Millisecond

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pm.RecordLatency("benchmark_operation", duration, labels)
	}
}

func BenchmarkPrometheusMetrics_RecordCounter(b *testing.B) {
	pm := testPrometheusMetrics
	labels := map[string]string{
		"graph_id":        "benchmark-graph",
		"evaluation_type": "benchmark",
		"budget_limit":    "tokens_only",
		"unit":            "benchmark-test",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pm.RecordCounter("budget_tokens_used", float64(i), labels)
	}
}

func BenchmarkPrometheusMetrics_RecordGauge(b *testing.B) {
	pm := testPrometheusMetrics
	labels := map[string]string{
		"graph_id":        "benchmark-graph",
		"evaluation_type": "benchmark",
		"budget_limit":    "tokens_only",
		"unit":            "benchmark-test",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pm.RecordGauge("cost_per_evaluation", float64(i)*0.001, labels)
	}
}
