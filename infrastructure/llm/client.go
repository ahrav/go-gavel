// Package llm provides a unified interface for interacting with various LLM providers
// with built-in support for rate limiting, circuit breaking, metrics, and tracing.
//
// The package abstracts multiple LLM providers (OpenAI, Anthropic, Google) behind
// a common interface while adding production-ready cross-cutting concerns through
// a middleware pattern. This allows applications to switch providers or add
// operational features without changing client code.
//
// Architecture:
//   - Core client implementation with middleware chain composition
//   - Provider implementations abstracted through CoreLLM interface
//   - Pluggable middleware for rate limiting, circuit breaking, metrics, tracing
//   - Advanced registry system for multi-provider scenarios
//   - Multiple token estimation strategies
//   - Factory functions for simple provider creation
//
// Basic usage:
//
//	client, err := llm.NewClient("openai", llm.ClientConfig{
//	    APIKey: os.Getenv("OPENAI_API_KEY"),
//	    Model:  "gpt-4",
//	})
//	response, err := client.Complete(ctx, "Hello world!", nil)
//
// Advanced usage with middleware:
//
//	client, err := llm.NewClient("anthropic", llm.ClientConfig{
//	    APIKey: os.Getenv("ANTHROPIC_API_KEY"),
//	    Model:  "claude-3-sonnet",
//	    Middleware: []llm.Middleware{
//	        llm.RateLimitMiddleware(20, 40),
//	        llm.CircuitBreakerMiddleware(5, 30*time.Second),
//	        llm.MetricsMiddleware(metricsCollector),
//	    },
//	})
package llm

import (
	"context"
	"fmt"
	"time"

	"github.com/ahrav/go-gavel/internal/ports"
)

// CoreLLM defines the minimal interface that LLM providers must implement.
// This interface abstracts the core functionality needed to make requests
// to different LLM services, allowing the middleware system to wrap
// any conforming implementation.
type CoreLLM interface {
	// DoRequest sends a prompt to the LLM provider and returns the response.
	// The opts parameter allows provider-specific configuration such as
	// temperature, max tokens, or other model parameters.
	// Returns the response text, input token count, output token count, and any error.
	DoRequest(
		ctx context.Context,
		prompt string,
		opts map[string]any,
	) (
		response string,
		tokensIn, tokensOut int,
		err error,
	)

	// GetModel returns the currently configured model name.
	GetModel() string

	// SetModel updates the model to use for subsequent requests.
	// This allows dynamic model switching without recreating the client.
	SetModel(model string)
}

// TokenEstimator provides pluggable token estimation strategies.
// Different providers may have different tokenization approaches,
// so this interface allows customization of token counting logic
// for cost estimation and rate limiting purposes.
type TokenEstimator interface {
	// EstimateTokens returns an approximate token count for the given text.
	// This is used for cost estimation and rate limiting when exact
	// token counts are not available before making requests.
	EstimateTokens(text string) int
}

// ClientConfig holds all configuration options for creating an LLM client.
// This struct centralizes all settings for providers, middleware,
// and operational concerns like rate limiting and circuit breaking.
type ClientConfig struct {
	// APIKey authenticates requests to the LLM provider.
	// For Google provider, this field contains the path to credentials file.
	APIKey string

	// Model specifies which LLM model to use for requests.
	// Each provider supports different model names.
	Model string

	// BaseURL overrides the default API endpoint for the provider.
	// Leave empty to use the provider's default endpoint.
	BaseURL string

	// Timeout sets the maximum duration for individual requests.
	// Zero value means no timeout.
	Timeout time.Duration

	// TokenEstimator provides custom token counting logic.
	// If nil, a simple character-based estimator is used.
	TokenEstimator TokenEstimator

	// Middleware allows custom middleware insertion.
	// These are applied in the order specified.
	Middleware []Middleware
}

// Middleware wraps a CoreLLM implementation to add cross-cutting functionality.
// This pattern allows composition of features like rate limiting, circuit breaking,
// metrics collection, and custom behavior without modifying core provider logic.
type Middleware func(CoreLLM) CoreLLM

// Client implements the ports.LLMClient interface with all cross-cutting concerns.
// It wraps a provider-specific CoreLLM implementation with middleware
// to provide production-ready features like resilience and observability.
type Client struct {
	core      CoreLLM
	estimator TokenEstimator
}

// NewClient creates a new LLM client with the specified provider and configuration.
// This function assembles the middleware chain and validates configuration
// before returning a ready-to-use client instance.
func NewClient(providerType string, config ClientConfig) (ports.LLMClient, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("API key is required")
	}

	if config.Model == "" {
		return nil, fmt.Errorf("model is required")
	}

	factory, ok := providerFactories[providerType]
	if !ok {
		return nil, fmt.Errorf("unknown provider: %s", providerType)
	}

	core, err := factory(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create provider: %w", err)
	}

	// Apply middleware in reverse order so the first middleware is the outermost.
	for i := len(config.Middleware) - 1; i >= 0; i-- {
		core = config.Middleware[i](core)
	}

	estimator := config.TokenEstimator
	if estimator == nil {
		estimator = &SimpleTokenEstimator{}
	}

	return &Client{
		core:      core,
		estimator: estimator,
	}, nil
}

// Complete sends a prompt to the LLM and returns the response text.
// This is a convenience method that discards token usage information
// for applications that don't need detailed usage tracking.
func (c *Client) Complete(ctx context.Context, prompt string, options map[string]any) (string, error) {
	response, _, _, err := c.CompleteWithUsage(ctx, prompt, options)
	return response, err
}

// CompleteWithUsage sends a prompt to the LLM and returns detailed usage information.
// This method provides access to token counts for cost calculation and usage tracking.
// The options parameter allows provider-specific configuration like temperature or max tokens.
func (c *Client) CompleteWithUsage(
	ctx context.Context,
	prompt string,
	options map[string]any,
) (string, int, int, error) {
	return c.core.DoRequest(ctx, prompt, options)
}

// EstimateTokens returns an approximate token count for the given text.
// This uses the configured TokenEstimator to provide cost estimates
// before making actual requests to the LLM provider.
func (c *Client) EstimateTokens(text string) (int, error) {
	return c.estimator.EstimateTokens(text), nil
}

// GetModel returns the currently configured model name from the underlying provider.
func (c *Client) GetModel() string { return c.core.GetModel() }

// SimpleTokenEstimator provides basic character-based token estimation.
// This implementation uses a simple heuristic of approximately 4 characters
// per token, which works reasonably well for most English text.
type SimpleTokenEstimator struct{}

// EstimateTokens returns an approximate token count using character-based heuristics.
// This implementation assumes roughly 4 characters per token,
// which provides reasonable estimates for cost calculation and rate limiting.
func (e *SimpleTokenEstimator) EstimateTokens(text string) int {
	return (len(text) + 3) / 4
}

// ProviderFactory creates a CoreLLM implementation from configuration.
// This function signature allows the provider registry to create
// provider instances without knowing their specific implementation details.
type ProviderFactory func(ClientConfig) (CoreLLM, error)

// Provider factory registry for extensibility.
// This allows registration of custom providers at runtime
// while maintaining type safety and initialization validation.
var providerFactories = map[string]ProviderFactory{}

// RegisterProviderFactory allows registration of custom LLM provider factories.
// This enables extension of the client with additional providers
// without modifying the core library code.
func RegisterProviderFactory(providerType string, factory ProviderFactory) {
	providerFactories[providerType] = factory
}
