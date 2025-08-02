package ports

import (
	"context"
	"time"
)

// LLMClient defines the interface for interacting with Large Language
// Model providers.
// Implementations should handle provider-specific details like authentication,
// request formatting, and response parsing.
type LLMClient interface {
	// Complete sends a completion request to the LLM provider.
	// It returns the generated text and any error encountered.
	// The implementation should handle rate limiting, retries, and timeouts.
	//
	// Parameters:
	//   - ctx: Context for cancellation and deadline propagation
	//   - prompt: The input prompt for the LLM
	//   - options: Provider-specific options (temperature, max tokens, etc.)
	//
	// The options map allows flexibility for different providers without
	// changing the interface. Common options include:
	//   - "temperature": float64 (0.0-1.0)
	//   - "max_tokens": int
	//   - "model": string (specific model version)
	Complete(ctx context.Context, prompt string, options map[string]any) (string, error)

	// EstimateTokens calculates the approximate token count for a given text.
	// This is useful for cost estimation and staying within model limits.
	// The estimation method may vary by provider.
	EstimateTokens(text string) (int, error)

	// GetModel returns the model identifier being used by this client.
	// This is useful for logging and debugging purposes.
	GetModel() string
}

// CacheStore defines the interface for caching evaluation results.
// Implementations could use Redis, Memcached, or in-memory storage.
// Caching is optional but can significantly reduce costs for repeated
// evaluations.
type CacheStore interface {
	// Get retrieves a cached value by key.
	// Returns the value and true if found, or nil and false if not found.
	// The implementation should handle serialization/deserialization.
	Get(ctx context.Context, key string) (any, bool, error)

	// Set stores a value in the cache with an expiration time.
	// The implementation should handle serialization of the value.
	// A zero duration means the item doesn't expire.
	Set(ctx context.Context, key string, value any, expiration time.Duration) error

	// Delete removes a value from the cache.
	// Returns nil if the key doesn't exist.
	Delete(ctx context.Context, key string) error

	// Clear removes all values from the cache.
	// This is useful for cache invalidation scenarios.
	Clear(ctx context.Context) error
}

// MetricsCollector defines the interface for collecting operational metrics.
// Implementations should integrate with observability platforms like
// Prometheus,
// OpenTelemetry, or custom monitoring solutions.
type MetricsCollector interface {
	// RecordLatency records the execution time of an operation.
	// The labels map provides additional context for the metric.
	RecordLatency(operation string, duration time.Duration, labels map[string]string)

	// RecordCounter increments a counter metric.
	// This is useful for tracking events like cache hits/misses, errors, etc.
	RecordCounter(metric string, value float64, labels map[string]string)

	// RecordGauge sets the current value of a gauge metric.
	// This is useful for tracking values like queue depth, active
	// connections, etc.
	RecordGauge(metric string, value float64, labels map[string]string)

	// RecordHistogram records a value in a histogram.
	// This is useful for tracking distributions like response sizes,
	// scores, etc.
	RecordHistogram(metric string, value float64, labels map[string]string)
}

// ConfigLoader defines the interface for loading configuration.
// Implementations could read from files, environment variables,
// remote configuration services, or a combination of sources.
type ConfigLoader interface {
	// Load reads configuration from the underlying source.
	// It should populate the provided configuration struct.
	// The config parameter should be a pointer to a struct.
	//
	// Example:
	//
	//	var config AppConfig
	//	err := loader.Load(ctx, &config)
	Load(ctx context.Context, config any) error

	// Watch monitors configuration changes and calls the callback when
	// changes occur.
	// This enables hot-reloading of configuration without service restart.
	// The callback receives the updated configuration.
	// Returns a function to stop watching when called.
	//
	// Example:
	//
	//	stop := loader.Watch(ctx, &config, func(updated any) {
	//	    // Handle configuration update
	//	})
	//	defer stop()
	Watch(ctx context.Context, config any, callback func(any)) (stop func(), err error)
}
