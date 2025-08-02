// Package middleware provides cross-cutting concerns for the evaluation engine.
package middleware

import (
	"context"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/ahrav/go-gavel/internal/domain"
	"github.com/ahrav/go-gavel/internal/ports"
)

var _ BudgetObserver = (*OTelBudgetObserver)(nil)

// OTelBudgetObserver implements observability for budget operations using
// OpenTelemetry tracing. It creates spans to track budget usage, sets
// detailed attributes, and records events for threshold warnings or errors.
type OTelBudgetObserver struct {
	metrics  ports.MetricsCollector
	unitName string
	span     trace.Span
}

// NewOTelBudgetObserver creates a new OpenTelemetry budget observer.
func NewOTelBudgetObserver(metrics ports.MetricsCollector, unitName string) *OTelBudgetObserver {
	return &OTelBudgetObserver{
		metrics:  metrics,
		unitName: unitName,
	}
}

// PreCheck implements the BudgetObserver interface. It starts an OpenTelemetry
// span and records the initial budget state and threshold warnings.
func (o *OTelBudgetObserver) PreCheck(ctx context.Context, usage domain.Usage, budget Budget) {
	tracer := otel.Tracer("budget-manager")
	_, span := tracer.Start(ctx, "BudgetManager.Execute")
	o.span = span

	o.addSpanAttributes(usage, budget)
	o.checkBudgetThresholds(usage, budget)
}

// PostCheck implements the BudgetObserver interface. It finalizes the span,
// records metrics, and handles any error conditions that occurred.
func (o *OTelBudgetObserver) PostCheck(
	ctx context.Context,
	usage domain.Usage,
	budget Budget,
	elapsed time.Duration,
	err error,
) {
	defer o.span.End()

	o.addSpanAttributes(usage, budget)

	if o.metrics != nil {
		labels := o.createMetricLabels(budget)
		o.metrics.RecordLatency("budget_manager_execution", elapsed, labels)
	}

	if err != nil {
		if budgetErr, ok := err.(*domain.BudgetExceededError); ok {
			o.span.AddEvent("budget.exceeded", trace.WithAttributes(
				attribute.String("limit_type", budgetErr.LimitType),
				attribute.Int("limit_value", budgetErr.Limit),
				attribute.Int("used_value", budgetErr.Used),
			))
			o.span.SetStatus(codes.Error, "Budget limit exceeded")

			if o.metrics != nil {
				labels := o.createMetricLabels(budget)
				labels["limit_type"] = budgetErr.LimitType
				o.metrics.RecordCounter("budget_exceeded_total", 1, labels)
			}
		} else {
			o.span.SetStatus(codes.Error, err.Error())
		}
		return
	}

	o.span.AddEvent("budget.usage_tracked", trace.WithAttributes(
		attribute.Int64("tokens_consumed", usage.Tokens),
		attribute.Int64("calls_made", usage.Calls),
	))

	o.updateMetrics(usage, budget)
	o.span.SetStatus(codes.Ok, "Budget management completed successfully")
}

// addSpanAttributes sets OpenTelemetry span attributes for budget tracking.
// It includes current usage, remaining budget, and configuration info.
func (o *OTelBudgetObserver) addSpanAttributes(usage domain.Usage, budget Budget) {
	o.span.SetAttributes(
		attribute.String("budget.unit", o.unitName),
		attribute.Int64("budget.tokens_used", usage.Tokens),
		attribute.Int64("budget.calls_made", usage.Calls),
	)

	if budget.MaxTokens > 0 {
		o.span.SetAttributes(
			attribute.Int64("budget.max_tokens", budget.MaxTokens),
			attribute.Int64("budget.remaining_tokens", budget.MaxTokens-usage.Tokens),
		)
	}

	if budget.MaxCalls > 0 {
		o.span.SetAttributes(
			attribute.Int64("budget.max_calls", budget.MaxCalls),
			attribute.Int64("budget.remaining_calls", budget.MaxCalls-usage.Calls),
		)
	}
}

// checkBudgetThresholds examines usage against configurable thresholds and
// generates span events for warning conditions to allow proactive monitoring.
func (o *OTelBudgetObserver) checkBudgetThresholds(usage domain.Usage, budget Budget) {
	// These thresholds may be configurable in future versions.
	const warningThreshold = 0.8
	const criticalThreshold = 0.9

	if budget.MaxTokens > 0 {
		usagePercentage := float64(usage.Tokens) / float64(budget.MaxTokens)
		if usagePercentage >= criticalThreshold {
			o.span.AddEvent("budget.threshold.critical", trace.WithAttributes(
				attribute.String("resource_type", "tokens"),
				attribute.Float64("usage_percentage", usagePercentage*100),
			))
		} else if usagePercentage >= warningThreshold {
			o.span.AddEvent("budget.threshold.warning", trace.WithAttributes(
				attribute.String("resource_type", "tokens"),
				attribute.Float64("usage_percentage", usagePercentage*100),
			))
		}
	}

	if budget.MaxCalls > 0 {
		usagePercentage := float64(usage.Calls) / float64(budget.MaxCalls)
		if usagePercentage >= criticalThreshold {
			o.span.AddEvent("budget.threshold.critical", trace.WithAttributes(
				attribute.String("resource_type", "calls"),
				attribute.Float64("usage_percentage", usagePercentage*100),
			))
		} else if usagePercentage >= warningThreshold {
			o.span.AddEvent("budget.threshold.warning", trace.WithAttributes(
				attribute.String("resource_type", "calls"),
				attribute.Float64("usage_percentage", usagePercentage*100),
			))
		}
	}
}

// updateMetrics sends current budget usage to the metrics collector.
func (o *OTelBudgetObserver) updateMetrics(usage domain.Usage, budget Budget) {
	if o.metrics == nil {
		return
	}

	labels := o.createMetricLabels(budget)
	o.metrics.RecordGauge("budget_tokens_used", float64(usage.Tokens), labels)
	o.metrics.RecordGauge("budget_calls_used", float64(usage.Calls), labels)

	// A simplified cost model for demonstration purposes.
	// TODO: Come back to this eventually.
	if usage.Calls > 0 {
		costPerEval := float64(usage.Tokens) * 0.001 // $0.001 per token
		o.metrics.RecordGauge("cost_per_evaluation", costPerEval, labels)
	}

	if budget.MaxTokens > 0 {
		remaining := budget.MaxTokens - usage.Tokens
		o.metrics.RecordGauge("budget_remaining_tokens", float64(remaining), labels)
	}

	if budget.MaxCalls > 0 {
		remaining := budget.MaxCalls - usage.Calls
		o.metrics.RecordGauge("budget_remaining_calls", float64(remaining), labels)
	}
}

// createMetricLabels creates the standard set of metric labels required
// for observability.
func (o *OTelBudgetObserver) createMetricLabels(budget Budget) map[string]string {
	return map[string]string{
		"budget_limit": o.getBudgetLimitLabel(budget),
		"unit":         o.unitName,
	}
}

// getBudgetLimitLabel creates a descriptive label for the current budget limits.
func (o *OTelBudgetObserver) getBudgetLimitLabel(budget Budget) string {
	if budget.MaxTokens > 0 && budget.MaxCalls > 0 {
		return "tokens_and_calls"
	} else if budget.MaxTokens > 0 {
		return "tokens_only"
	} else if budget.MaxCalls > 0 {
		return "calls_only"
	}
	return "unlimited"
}
