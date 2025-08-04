package llm

import (
	"context"
)

// tracedLLM implements distributed tracing for request observability.
// This provides detailed request traces for debugging and performance
// analysis across distributed systems.
type tracedLLM struct {
	next        CoreLLM
	serviceName string
}

// TracingMiddleware creates middleware that adds distributed tracing to requests.
// This enables tracking of LLM requests across distributed systems
// and helps with debugging and performance analysis.
func TracingMiddleware(serviceName string) Middleware {
	return func(next CoreLLM) CoreLLM {
		return &tracedLLM{
			next:        next,
			serviceName: serviceName,
		}
	}
}

// DoRequest executes the request within a distributed trace span.
// This creates detailed traces with request attributes and timing
// information for comprehensive observability.
func (t *tracedLLM) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
	// In a real implementation, this would integrate with OpenTelemetry or similar
	// For this example, we'll just pass through
	// span, ctx := tracer.Start(ctx, "llm.request",
	//     trace.WithAttributes(
	//         attribute.String("service.name", t.serviceName),
	//         attribute.String("llm.model", t.next.GetModel()),
	//         attribute.Int("llm.prompt.length", len(prompt)),
	//     ),
	// )
	// defer span.End()

	response, tokensIn, tokensOut, err := t.next.DoRequest(ctx, prompt, opts)

	// if err != nil {
	//     span.RecordError(err)
	//     span.SetStatus(codes.Error, err.Error())
	// } else {
	//     span.SetAttributes(
	//         attribute.Int("llm.tokens.input", tokensIn),
	//         attribute.Int("llm.tokens.output", tokensOut),
	//     )
	// }

	return response, tokensIn, tokensOut, err
}

// GetModel returns the model name from the wrapped implementation.
func (t *tracedLLM) GetModel() string { return t.next.GetModel() }

// SetModel updates the model name in the wrapped implementation.
func (t *tracedLLM) SetModel(m string) { t.next.SetModel(m) }
