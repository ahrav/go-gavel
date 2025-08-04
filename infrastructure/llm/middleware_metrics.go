package llm

import (
	"context"
	"strings"
	"time"

	"github.com/ahrav/go-gavel/internal/ports"
)

// metricsLLM implements request metrics collection.
// This provides observability into request patterns, latency,
// token usage, and error rates for operational monitoring.
type metricsLLM struct {
	next      CoreLLM
	collector ports.MetricsCollector
}

// MetricsMiddleware creates middleware that collects request metrics.
// This enables monitoring of LLM usage, performance, and costs across providers.
func MetricsMiddleware(collector ports.MetricsCollector) Middleware {
	return func(next CoreLLM) CoreLLM {
		return &metricsLLM{
			next:      next,
			collector: collector,
		}
	}
}

// DoRequest executes the request while collecting detailed metrics.
// This tracks request latency, status codes, token usage, and provider information
// for comprehensive operational observability.
func (m *metricsLLM) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
	start := time.Now()
	response, tokensIn, tokensOut, err := m.next.DoRequest(ctx, prompt, opts)

	labels := map[string]string{
		"provider": m.extractProvider(),
		"model":    m.next.GetModel(),
		"status":   "success",
	}

	if err != nil {
		if err == ErrCircuitOpen {
			labels["status"] = "circuit_open"
		} else if ctx.Err() == context.DeadlineExceeded {
			labels["status"] = "timeout"
		} else {
			labels["status"] = "error"
		}
	}

	if m.collector != nil {
		m.collector.RecordHistogram("llm_latency_seconds", time.Since(start).Seconds(), labels)
		m.collector.RecordCounter("llm_requests_total", 1, labels)

		if err == nil {
			labels["token_type"] = "input"
			m.collector.RecordCounter("llm_tokens_total", float64(tokensIn), labels)

			labels["token_type"] = "output"
			m.collector.RecordCounter("llm_tokens_total", float64(tokensOut), labels)
		}
	}

	return response, tokensIn, tokensOut, err
}

func (m *metricsLLM) extractProvider() string {
	model := m.next.GetModel()
	if strings.Contains(model, "gpt") {
		return "openai"
	} else if strings.Contains(model, "claude") {
		return "anthropic"
	} else if strings.Contains(model, "gemini") {
		return "google"
	}
	return "unknown"
}

// GetModel returns the model name from the wrapped implementation.
func (m *metricsLLM) GetModel() string { return m.next.GetModel() }

// SetModel updates the model name in the wrapped implementation.
func (m *metricsLLM) SetModel(model string) { m.next.SetModel(model) }
