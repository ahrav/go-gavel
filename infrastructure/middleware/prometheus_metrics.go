// Package middleware provides cross-cutting concerns for the evaluation engine.
package middleware

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/ahrav/go-gavel/internal/ports"
)

// PrometheusMetrics implements the MetricsCollector interface using Prometheus.
// It provides real-time monitoring of budget consumption, execution
// performance, and cost analysis for the evaluation engine.
type PrometheusMetrics struct {
	budgetTokensUsed  *prometheus.CounterVec
	budgetCallsUsed   *prometheus.CounterVec
	costPerEvaluation *prometheus.GaugeVec
	executionLatency  *prometheus.HistogramVec
	operationCounter  *prometheus.CounterVec
	systemGauges      *prometheus.GaugeVec
}

// NewPrometheusMetrics creates a new PrometheusMetrics instance and registers
// all required metrics in the global Prometheus registry.
func NewPrometheusMetrics() *PrometheusMetrics {
	return &PrometheusMetrics{
		// Budget-specific metrics.
		budgetTokensUsed: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "budget_tokens_used",
				Help: "Total number of tokens consumed across all LLM interactions.",
			},
			[]string{"graph_id", "evaluation_type", "budget_limit", "unit"},
		),
		budgetCallsUsed: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "budget_calls_used",
				Help: "Total number of API calls made across all LLM interactions.",
			},
			[]string{"graph_id", "evaluation_type", "budget_limit", "unit"},
		),
		costPerEvaluation: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "cost_per_evaluation",
				Help: "Cost efficiency calculated per evaluation run.",
			},
			[]string{"graph_id", "evaluation_type", "budget_limit", "unit"},
		),

		// General execution metrics for comprehensive observability.
		executionLatency: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "budget_manager_execution_duration_seconds",
				Help:    "Execution time of budget manager operations.",
				Buckets: prometheus.DefBuckets,
			},
			[]string{"operation", "unit"},
		),
		operationCounter: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "budget_manager_operations_total",
				Help: "Total number of operations performed by the budget manager.",
			},
			[]string{"operation", "status", "unit"},
		),
		systemGauges: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "budget_manager_system_state",
				Help: "Current system state values for the budget manager.",
			},
			[]string{"metric", "unit"},
		),
	}
}

// RecordLatency implements the MetricsCollector interface by recording
// execution latency in a Prometheus histogram.
func (pm *PrometheusMetrics) RecordLatency(
	operation string,
	duration time.Duration,
	labels map[string]string,
) {
	unit, ok := labels["unit"]
	if !ok {
		unit = "unknown"
	}
	pm.executionLatency.WithLabelValues(operation, unit).Observe(duration.Seconds())
}

// RecordCounter implements the MetricsCollector interface by incrementing
// Prometheus counters.
func (pm *PrometheusMetrics) RecordCounter(
	metric string, value float64, labels map[string]string,
) {
	unit, ok := labels["unit"]
	if !ok {
		unit = "unknown"
	}

	switch metric {
	case "budget_tokens_used":
		pm.budgetTokensUsed.WithLabelValues(
			labels["graph_id"],
			labels["evaluation_type"],
			labels["budget_limit"],
			unit,
		).Add(value)
	case "budget_calls_used":
		pm.budgetCallsUsed.WithLabelValues(
			labels["graph_id"],
			labels["evaluation_type"],
			labels["budget_limit"],
			unit,
		).Add(value)
	case "budget_exceeded_total":
		status := "exceeded_" + labels["limit_type"]
		pm.operationCounter.WithLabelValues("budget_check", status, unit).Add(value)
	default:
		pm.operationCounter.WithLabelValues(metric, "success", unit).Add(value)
	}
}

// RecordGauge implements the MetricsCollector interface by setting
// Prometheus gauge values.
func (pm *PrometheusMetrics) RecordGauge(
	metric string, value float64, labels map[string]string,
) {
	unit, ok := labels["unit"]
	if !ok {
		unit = "unknown"
	}

	switch metric {
	case "cost_per_evaluation":
		pm.costPerEvaluation.WithLabelValues(
			labels["graph_id"],
			labels["evaluation_type"],
			labels["budget_limit"],
			unit,
		).Set(value)
	case "budget_tokens_used", "budget_calls_used", "budget_remaining_tokens", "budget_remaining_calls":
		pm.systemGauges.WithLabelValues(metric, unit).Set(value)
	default:
		pm.systemGauges.WithLabelValues(metric, unit).Set(value)
	}
}

// RecordHistogram implements the MetricsCollector interface by recording
// values in a Prometheus histogram.
func (pm *PrometheusMetrics) RecordHistogram(
	metric string, value float64, labels map[string]string,
) {
	unit, ok := labels["unit"]
	if !ok {
		unit = "unknown"
	}
	// This currently routes all histograms to the general execution latency
	// metric. In a full implementation, this would use metric-specific
	// histograms.
	pm.executionLatency.WithLabelValues(metric, unit).Observe(value)
}

// Compile-time verification that PrometheusMetrics implements MetricsCollector.
var _ ports.MetricsCollector = (*PrometheusMetrics)(nil)
