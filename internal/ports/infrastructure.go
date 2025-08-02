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

	// CompleteWithUsage sends a completion request to the LLM provider and
	// returns detailed token usage information for budget tracking.
	// It returns the generated text along with input and output token counts.
	// The implementation should handle rate limiting, retries, and timeouts.
	//
	// Parameters:
	//   - ctx: Context for cancellation and deadline propagation
	//   - prompt: The input prompt for the LLM
	//   - options: Provider-specific options (temperature, max tokens, etc.)
	//
	// Returns:
	//   - output: The generated text response
	//   - tokensIn: Number of tokens in the input prompt
	//   - tokensOut: Number of tokens in the generated output
	//   - error: Any error encountered during the request
	//
	// This method is essential for budget management and cost tracking.
	CompleteWithUsage(ctx context.Context, prompt string, options map[string]any) (output string, tokensIn, tokensOut int, err error)

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

// UnitRegistry defines the interface for creating and managing evaluation units.
// It acts as a factory for different unit types based on configuration.
type UnitRegistry interface {
	// CreateUnit creates a new unit instance based on the provided type and configuration.
	// The configuration is unit-type specific and will be validated by the implementation.
	//
	// Parameters:
	//   - unitType: The type of unit to create (e.g., "llm_judge", "code_analyzer")
	//   - id: Unique identifier for the unit instance
	//   - config: Unit-specific configuration as a map
	//
	// Returns:
	//   - Unit: The created unit instance
	//   - error: Any error encountered during creation
	CreateUnit(unitType string, id string, config map[string]any) (Unit, error)

	// RegisterUnitFactory registers a factory function for a specific unit type.
	// This allows for extensibility by adding new unit types at runtime.
	//
	// Parameters:
	//   - unitType: The type identifier for the unit
	//   - factory: Function that creates instances of this unit type
	RegisterUnitFactory(unitType string, factory UnitFactory) error

	// GetSupportedTypes returns a list of all registered unit types.
	// This is useful for validation and documentation purposes.
	GetSupportedTypes() []string
}

// UnitFactory is a function type for creating unit instances.
// Each unit type should have its own factory function that handles
// type-specific configuration and initialization.
type UnitFactory func(id string, config map[string]any) (Unit, error)
