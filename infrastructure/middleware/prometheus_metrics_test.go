// Package middleware_test contains the unit tests for the middleware package.
package middleware

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ahrav/go-gavel/internal/ports"
)

// testPrometheusMetrics provides a global instance to avoid duplicate metric
// registration issues across tests in the same package.
var testPrometheusMetrics *PrometheusMetrics

func init() {
	// Create a single PrometheusMetrics instance to be shared across all tests
	// in this package. This prevents Prometheus from panicking due to duplicate
	// metric registration.
	testPrometheusMetrics = NewPrometheusMetrics()
}

// TestNewPrometheusMetrics verifies that a new PrometheusMetrics instance is
// created with all its internal metrics properly initialized.
func TestNewPrometheusMetrics(t *testing.T) {
	// Use the global test instance to avoid registration conflicts.
	pm := testPrometheusMetrics

	// Verify that the instance itself is not nil.
	assert.NotNil(t, pm, "PrometheusMetrics instance should not be nil")

	// Verify that all metric vectors are properly initialized.
	assert.NotNil(t, pm.budgetTokensUsed, "budgetTokensUsed should be initialized")
	assert.NotNil(t, pm.budgetCallsUsed, "budgetCallsUsed should be initialized")
	assert.NotNil(t, pm.costPerEvaluation, "costPerEvaluation should be initialized")
	assert.NotNil(t, pm.executionLatency, "executionLatency should be initialized")
	assert.NotNil(t, pm.operationCounter, "operationCounter should be initialized")
	assert.NotNil(t, pm.systemGauges, "systemGauges should be initialized")

	// Verify that PrometheusMetrics correctly implements the MetricsCollector interface.
	var _ ports.MetricsCollector = pm
}

// TestPrometheusMetrics_RecordLatency tests the recording of latency metrics
// with various label combinations.
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
			// This test primarily ensures that recording latency does not panic.
			// Verifying the actual metric values would require the Prometheus
			// testutil package and a more complex setup.
			assert.NotPanics(t, func() {
				pm.RecordLatency(tt.operation, tt.duration, tt.labels)
			}, "RecordLatency should not panic")
		})
	}
}

// TestPrometheusMetrics_RecordCounter tests the recording of various counter
// metrics, including both specific and generic counters.
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
			name:       "record budget tokens used",
			metric:     "budget_tokens_used",
			value:      100.0,
			labels:     map[string]string{"unit": "test-unit"},
			shouldWork: true,
		},
		{
			name:       "record budget calls used",
			metric:     "budget_calls_used",
			value:      5.0,
			labels:     map[string]string{"unit": "test-unit"},
			shouldWork: true,
		},
		{
			name:       "record budget exceeded",
			metric:     "budget_exceeded_total",
			value:      1.0,
			labels:     map[string]string{"limit_type": "tokens", "unit": "test-unit"},
			shouldWork: true,
		},
		{
			name:       "record unknown metric as generic counter",
			metric:     "unknown_metric",
			value:      42.0,
			labels:     map[string]string{"unit": "test-unit"},
			shouldWork: true,
		},
		{
			name:       "record with missing unit label",
			metric:     "budget_tokens_used",
			value:      50.0,
			labels:     map[string]string{},
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

// TestPrometheusMetrics_RecordGauge tests the recording of various gauge
// metrics, including both specific and generic gauges.
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
			name:       "record cost per evaluation",
			metric:     "cost_per_evaluation",
			value:      0.05,
			labels:     map[string]string{"unit": "test-unit"},
			shouldWork: true,
		},
		{
			name:       "record budget remaining tokens",
			metric:     "budget_remaining_tokens",
			value:      850.0,
			labels:     map[string]string{"unit": "test-unit"},
			shouldWork: true,
		},
		{
			name:       "record budget remaining calls",
			metric:     "budget_remaining_calls",
			value:      15.0,
			labels:     map[string]string{"unit": "test-unit"},
			shouldWork: true,
		},
		{
			name:       "record unknown gauge metric",
			metric:     "unknown_gauge",
			value:      123.45,
			labels:     map[string]string{"unit": "test-unit"},
			shouldWork: true,
		},
		{
			name:       "record with empty unit label",
			metric:     "cost_per_evaluation",
			value:      0.03,
			labels:     map[string]string{"unit": ""},
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

// TestPrometheusMetrics_RecordHistogram tests the recording of generic
// histogram metrics.
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
			name:       "record histogram with unit",
			metric:     "test_histogram",
			value:      0.123,
			labels:     map[string]string{"unit": "test-unit"},
			shouldWork: true,
		},
		{
			name:       "record histogram without unit",
			metric:     "another_histogram",
			value:      0.456,
			labels:     map[string]string{"other": "value"},
			shouldWork: true,
		},
		{
			name:       "record histogram with empty unit",
			metric:     "empty_unit_histogram",
			value:      0.789,
			labels:     map[string]string{"unit": ""},
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

// TestPrometheusMetrics_LabelHandling verifies that the metrics collector
// gracefully handles nil, empty, and incomplete label maps.
func TestPrometheusMetrics_LabelHandling(t *testing.T) {
	pm := testPrometheusMetrics

	tests := []struct {
		name   string
		labels map[string]string
	}{
		{"nil labels map", nil},
		{"empty labels map", map[string]string{}},
		{"labels map with unit", map[string]string{"unit": "test-unit"}},
		{"labels map with empty unit", map[string]string{"unit": ""}},
		{"labels map without unit", map[string]string{"other": "value"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.NotPanics(t, func() {
				pm.RecordLatency("test_op", 100*time.Millisecond, tt.labels)
			}, "RecordLatency should handle labels gracefully")

			assert.NotPanics(t, func() {
				pm.RecordCounter("test_counter", 1.0, tt.labels)
			}, "RecordCounter should handle labels gracefully")

			assert.NotPanics(t, func() {
				pm.RecordGauge("test_gauge", 42.0, tt.labels)
			}, "RecordGauge should handle labels gracefully")

			assert.NotPanics(t, func() {
				pm.RecordHistogram("test_hist", 0.5, tt.labels)
			}, "RecordHistogram should handle labels gracefully")
		})
	}
}

// TestPrometheusMetrics_InterfaceCompliance ensures that PrometheusMetrics
// correctly implements the ports.MetricsCollector interface.
func TestPrometheusMetrics_InterfaceCompliance(t *testing.T) {
	var metrics ports.MetricsCollector = testPrometheusMetrics
	require.NotNil(t, metrics, "PrometheusMetrics should implement MetricsCollector")

	// Test that all interface methods can be called without panicking.
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

// TestPrometheusMetrics_BudgetSpecificMetrics tests the recording of metrics
// that are specific to the budget management system.
func TestPrometheusMetrics_BudgetSpecificMetrics(t *testing.T) {
	pm := testPrometheusMetrics

	budgetLabels := map[string]string{"unit": "budget-manager"}

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

// TestPrometheusMetrics_EdgeCases tests various edge cases to ensure the
// metrics collector is robust.
func TestPrometheusMetrics_EdgeCases(t *testing.T) {
	pm := testPrometheusMetrics

	t.Run("zero duration latency", func(t *testing.T) {
		assert.NotPanics(t, func() {
			pm.RecordLatency("zero_duration", 0, map[string]string{"unit": "test"})
		}, "Should handle zero duration gracefully")
	})

	t.Run("negative counter value", func(t *testing.T) {
		// Prometheus counters cannot be negative, so this should panic.
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
		// The system should handle missing labels gracefully by using defaults.
		incompleteLabels := map[string]string{"graph_id": "test-graph"}
		assert.NotPanics(t, func() {
			pm.RecordCounter("budget_tokens_used", 100.0, incompleteLabels)
		}, "Should handle incomplete budget labels gracefully")
	})
}

// BenchmarkPrometheusMetrics_RecordLatency benchmarks the performance of
// recording latency metrics.
func BenchmarkPrometheusMetrics_RecordLatency(b *testing.B) {
	pm := testPrometheusMetrics
	labels := map[string]string{"unit": "benchmark-test"}
	duration := 100 * time.Millisecond

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pm.RecordLatency("benchmark_operation", duration, labels)
	}
}

// BenchmarkPrometheusMetrics_RecordCounter benchmarks the performance of
// recording counter metrics.
func BenchmarkPrometheusMetrics_RecordCounter(b *testing.B) {
	pm := testPrometheusMetrics
	labels := map[string]string{"unit": "benchmark-test"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pm.RecordCounter("benchmark_counter", float64(i), labels)
	}
}

// BenchmarkPrometheusMetrics_RecordGauge benchmarks the performance of
// recording gauge metrics.
func BenchmarkPrometheusMetrics_RecordGauge(b *testing.B) {
	pm := testPrometheusMetrics
	labels := map[string]string{"unit": "benchmark-test"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pm.RecordGauge("benchmark_gauge", float64(i)*0.001, labels)
	}
}
