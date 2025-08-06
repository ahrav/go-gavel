// Package llm/registry provides advanced multi-provider management for LLM clients.
// This file implements the Registry system that enables centralized configuration,
// automatic provider initialization, and dynamic client management across multiple
// LLM providers simultaneously.
//
// The Registry system supports:
//   - Environment-based provider initialization
//   - Default configuration inheritance across providers
//   - Dynamic client registration and retrieval
//   - Provider-specific configuration overrides
//   - Centralized metrics and observability
//   - Model-based client routing (provider/model format)
//
// Usage Examples:
//
// Basic Registry Setup:
//
//	config := llm.RegistryConfig{
//	    DefaultProvider: "openai",
//	    Providers: llm.DefaultProviders,
//	}
//	registry := llm.NewRegistry(config)
//	client, err := registry.GetDefaultClient()
//
// Client Retrieval:
//
//	client, err := registry.GetClient("openai/gpt-4")
//	client, err := registry.GetDefaultClient() // Uses default provider
//	client, err := registry.GetClient("openai") // Uses default model for provider
//
// Custom Registration:
//
//	err := registry.RegisterClient("custom", llm.ClientConfig{
//	    APIKey: "custom-key",
//	    Model:  "gpt-3.5-turbo",
//	})
package llm

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/ahrav/go-gavel/internal/ports"
)

// Registry provides advanced multi-provider management for LLM clients.
// It enables centralized configuration, automatic initialization, and dynamic
// management of multiple LLM providers with shared default settings.
type Registry struct {
	// providers maps provider names to their configuration
	providers map[string]ProviderConfig
	// clients maps "provider/model" keys to their respective LLMClient implementations.
	// Each client handles its own rate limiting and circuit breaking.
	clients map[string]ports.LLMClient
	// defaultProvider specifies the fallback provider when the model field is omitted
	// in the unit configuration.
	defaultProvider string
	// defaultMiddleware specifies middleware applied to all providers unless overridden
	defaultMiddleware []Middleware
	// defaultTimeout sets the default request timeout for all providers
	defaultTimeout time.Duration
	// mu provides thread-safe access to the registry.
	mu sync.RWMutex
}

// ProviderConfig holds provider-specific configuration.
// This struct allows fine-grained control over individual provider settings,
// overriding registry defaults for specific providers.
type ProviderConfig struct {
	// Type specifies the provider implementation type (openai, anthropic, google)
	Type string
	// EnvVar specifies the environment variable name for the API key
	EnvVar string
	// DefaultModel specifies the default model to use if not specified
	DefaultModel string
	// SupportedModels lists all models supported by this provider
	// If empty, no validation is performed (allows any model)
	SupportedModels []string
	// BaseURL overrides the default API endpoint for the provider
	BaseURL string
	// Middleware specifies provider-specific middleware
	Middleware []Middleware
}

// RegistryConfig holds configuration for the provider registry.
// This struct defines default settings that are applied to all providers
// unless overridden by provider-specific configuration.
type RegistryConfig struct {
	// Providers defines the available providers and their configurations
	Providers map[string]ProviderConfig
	// DefaultProvider specifies which provider to use when no provider is specified.
	DefaultProvider string
	// DefaultTimeout sets the default request timeout for all providers.
	DefaultTimeout time.Duration
	// DefaultMiddleware specifies default middleware applied to all providers.
	DefaultMiddleware []Middleware
}

// DefaultProviders provides standard provider configurations for common LLM services.
// Applications can use this as a starting point and override specific settings.
var DefaultProviders = map[string]ProviderConfig{
	"openai": {
		Type:         "openai",
		EnvVar:       "OPENAI_API_KEY",
		DefaultModel: "gpt-4.1",
		SupportedModels: []string{
			// GPT-4.1 series (latest flagship)
			"gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
			// GPT-4o series (omni models)
			"gpt-4o", "gpt-4o-mini", "gpt-4o-audio",
			// GPT-4 series (classic)
			"gpt-4", "gpt-4-turbo",
			// GPT-3.5 series (legacy)
			"gpt-3.5-turbo", "gpt-3.5-turbo-instruct",
			// Reasoning models
			"o4-mini", "o3", "o3-mini", "o1", "o1-mini",
			// Experimental models
			"gpt-4.5",
		},
	},
	"anthropic": {
		Type:         "anthropic",
		EnvVar:       "ANTHROPIC_API_KEY",
		DefaultModel: "claude-4-sonnet",
		SupportedModels: []string{
			// Claude 4 series (latest flagship)
			"claude-4-opus", "claude-4-sonnet", "claude-4.1-opus",
			// Claude 3.7 series
			"claude-3.7-sonnet",
			// Claude 3.5 series
			"claude-3.5-sonnet", "claude-3.5-haiku",
			// Claude 3 series (legacy)
			"claude-3-haiku", "claude-3-sonnet", "claude-3-opus",
		},
	},
	"google": {
		Type:         "google",
		EnvVar:       "GOOGLE_API_KEY",
		DefaultModel: "gemini-2.5-flash",
		SupportedModels: []string{
			// Gemini 2.5 series (latest flagship)
			"gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
			// Gemini 2.0 series
			"gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.0-pro-experimental",
			// Gemini 1.5 series (legacy but supported)
			"gemini-1.5-pro", "gemini-1.5-flash",
		},
	},
}

// NewRegistry creates a new provider registry with advanced configuration options.
// The registry manages multiple LLM providers with shared default settings
// and enables dynamic client management and routing.
func NewRegistry(config RegistryConfig) (*Registry, error) {
	if config.DefaultProvider == "" {
		return nil, fmt.Errorf("default provider cannot be empty")
	}

	// Validate that default provider exists in configuration
	if _, exists := config.Providers[config.DefaultProvider]; !exists {
		return nil, fmt.Errorf("default provider %q not found in providers configuration", config.DefaultProvider)
	}

	return &Registry{
		providers:         config.Providers,
		clients:           make(map[string]ports.LLMClient),
		defaultProvider:   config.DefaultProvider,
		defaultMiddleware: config.DefaultMiddleware,
		defaultTimeout:    config.DefaultTimeout,
	}, nil
}

// GetDefaultClient returns a client for the default provider.
// This method provides explicit access to the default provider client,
// making intent clear and avoiding the ambiguity of empty string parameters.
func (r *Registry) GetDefaultClient() (ports.LLMClient, error) {
	providerConfig, exists := r.providers[r.defaultProvider]
	if !exists {
		return nil, fmt.Errorf("default provider %q not found in configuration", r.defaultProvider)
	}

	return r.GetClient(r.defaultProvider + "/" + providerConfig.DefaultModel)
}

// GetClient retrieves a client by provider name or model string.
// Supports multiple formats:
//   - "provider": Returns client for specified provider with default model
//   - "provider/model": Returns client for specified provider and model
//
// Empty strings are not allowed - use GetDefaultClient() for default provider.
// The method creates clients lazily on first request and caches them for reuse.
// Each unique provider/model combination gets its own client instance.
func (r *Registry) GetClient(spec string) (ports.LLMClient, error) {
	if spec == "" {
		return nil, fmt.Errorf("provider specification cannot be empty; use GetDefaultClient() for default provider")
	}

	provider, model := r.parseSpec(spec)

	key := r.buildCacheKey(provider, model)

	r.mu.RLock()
	if client, exists := r.clients[key]; exists {
		r.mu.RUnlock()
		return client, nil
	}
	r.mu.RUnlock()

	r.mu.Lock()
	defer r.mu.Unlock()

	if client, exists := r.clients[key]; exists {
		return client, nil
	}

	client, err := r.createClient(provider, model)
	if err != nil {
		return nil, err
	}

	r.clients[key] = client
	return client, nil
}

// RegisterClient registers a new client with the registry using custom configuration.
// This method allows dynamic registration of providers with provider-specific
// options while inheriting registry defaults.
func (r *Registry) RegisterClient(name string, config ClientConfig) error {
	if name == "" {
		return fmt.Errorf("client name cannot be empty")
	}

	// Parse provider type from name if it contains "/"
	provider, model := r.parseSpec(name)
	if provider == "" {
		provider = name
		model = config.Model
	}

	// Look up provider configuration
	providerConfig, exists := r.providers[provider]
	if !exists {
		return fmt.Errorf("unknown provider %q", provider)
	}

	// Create client with merged configuration
	client, err := r.createClientWithConfig(providerConfig.Type, config)
	if err != nil {
		return fmt.Errorf("failed to create client %q: %w", name, err)
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	key := r.buildCacheKey(provider, model)
	r.clients[key] = client
	return nil
}

// parseSpec extracts provider name and model from a specification string.
// Supports formats:
//   - "provider" -> (provider, defaultModel)
//   - "provider/model" -> (provider, model)
//
// Empty strings are not supported - caller should validate input.
func (r *Registry) parseSpec(spec string) (provider, model string) {
	parts := strings.SplitN(spec, "/", 2)
	provider = parts[0]

	if len(parts) > 1 {
		model = parts[1]
	} else if providerConfig, ok := r.providers[provider]; ok {
		model = providerConfig.DefaultModel
	}

	return
}

// buildCacheKey creates a consistent cache key from provider and model.
// This ensures proper caching and retrieval of clients.
func (r *Registry) buildCacheKey(provider, model string) string {
	if model == "" {
		return provider
	}
	return provider + "/" + model
}

// createClient creates a new client instance for the given provider and model.
// It handles environment variable loading, configuration merging, model validation, and client initialization.
func (r *Registry) createClient(provider, model string) (ports.LLMClient, error) {
	providerConfig, exists := r.providers[provider]
	if !exists {
		return nil, fmt.Errorf("unknown provider %q", provider)
	}

	if len(providerConfig.SupportedModels) > 0 {
		if !r.isModelSupported(model, providerConfig.SupportedModels) {
			return nil, fmt.Errorf("model %q is not supported by provider %q. Supported models: %v",
				model, provider, providerConfig.SupportedModels)
		}
	}

	apiKey := os.Getenv(providerConfig.EnvVar)
	if apiKey == "" {
		return nil, fmt.Errorf("%s environment variable not set for provider %q", providerConfig.EnvVar, provider)
	}

	config := ClientConfig{
		APIKey:  apiKey,
		Model:   model,
		BaseURL: providerConfig.BaseURL,
		Timeout: r.defaultTimeout,
	}

	config.Middleware = append([]Middleware{}, r.defaultMiddleware...)
	config.Middleware = append(config.Middleware, providerConfig.Middleware...)

	return NewClient(providerConfig.Type, config)
}

// createClientWithConfig creates a client with explicit configuration.
// Used by RegisterClient for custom client registration.
func (r *Registry) createClientWithConfig(providerType string, config ClientConfig) (ports.LLMClient, error) {
	if config.Timeout == 0 {
		config.Timeout = r.defaultTimeout
	}

	middleware := append([]Middleware{}, r.defaultMiddleware...)
	config.Middleware = append(middleware, config.Middleware...)

	return NewClient(providerType, config)
}

// InitializeProviders automatically initializes providers based on environment variables.
// This method discovers available providers by checking environment variables
// and creates clients with default configuration for each available provider.
func (r *Registry) InitializeProviders() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	foundDefault := false

	for providerName, providerConfig := range r.providers {
		apiKey := os.Getenv(providerConfig.EnvVar)
		if apiKey == "" {
			if r.defaultProvider == providerName {
				return fmt.Errorf("%s environment variable not set for default provider %q",
					providerConfig.EnvVar, providerName)
			}
			continue
		}

		if providerName == r.defaultProvider {
			foundDefault = true
		}

		config := ClientConfig{
			APIKey:     apiKey,
			Model:      providerConfig.DefaultModel,
			BaseURL:    providerConfig.BaseURL,
			Timeout:    r.defaultTimeout,
			Middleware: append(append([]Middleware{}, r.defaultMiddleware...), providerConfig.Middleware...),
		}

		client, err := NewClient(providerConfig.Type, config)
		if err != nil {
			return fmt.Errorf("failed to create %s client: %w", providerName, err)
		}

		key := r.buildCacheKey(providerName, providerConfig.DefaultModel)
		r.clients[key] = client
	}

	if !foundDefault {
		return fmt.Errorf("default provider %q not available after initialization", r.defaultProvider)
	}

	return nil
}

// GetRegisteredProviders returns a list of all currently registered provider names.
// This is useful for validation and debugging.
func (r *Registry) GetRegisteredProviders() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	providerSet := make(map[string]bool)
	for key := range r.clients {
		provider, _ := r.parseSpec(key)
		if provider != "" {
			providerSet[provider] = true
		}
	}

	providers := make([]string, 0, len(providerSet))
	for provider := range providerSet {
		providers = append(providers, provider)
	}
	return providers
}

// UpdateDefaultMiddleware updates the default middleware for new clients.
// These middleware will be applied to all subsequently created clients
// but will not affect existing clients.
func (r *Registry) UpdateDefaultMiddleware(middleware ...Middleware) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.defaultMiddleware = append(r.defaultMiddleware, middleware...)
}

// SetDefaultTimeout sets the default timeout for new clients.
// This timeout will be applied to all subsequently created clients
// but will not affect existing clients.
func (r *Registry) SetDefaultTimeout(timeout time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.defaultTimeout = timeout
}

// isModelSupported checks if a model is in the supported models list.
func (r *Registry) isModelSupported(model string, supportedModels []string) bool {
	for _, supportedModel := range supportedModels {
		if model == supportedModel {
			return true
		}
	}
	return false
}
