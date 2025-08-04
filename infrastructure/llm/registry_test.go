package llm

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewRegistry(t *testing.T) {
	config := RegistryConfig{
		DefaultProvider: "openai",
		Providers: map[string]ProviderConfig{
			"openai": {
				Type:         "openai",
				EnvVar:       "OPENAI_API_KEY",
				DefaultModel: "gpt-4",
			},
		},
		DefaultTimeout: 30 * time.Second,
		DefaultMiddleware: []Middleware{
			TimeoutMiddleware(30 * time.Second),
			RetryMiddleware(3, time.Second, 5*time.Second),
		},
	}

	registry, err := NewRegistry(config)
	require.NoError(t, err, "Failed to create registry")
	require.NotNil(t, registry, "Expected non-nil registry")

	assert.Equal(t, "openai", registry.defaultProvider, "Default provider mismatch")
	assert.Len(t, registry.defaultMiddleware, 2, "Expected 2 default middleware")
}

func TestRegistry_RegisterClient(t *testing.T) {
	// First register a custom provider factory
	RegisterProviderFactory("custom", func(config ClientConfig) (CoreLLM, error) {
		return &customProvider{
			apiKey: config.APIKey,
			model:  config.Model,
		}, nil
	})

	// Set up environment variables
	t.Setenv("OPENAI_API_KEY", "test-key")
	t.Setenv("CUSTOM_API_KEY", "custom-key")

	config := RegistryConfig{
		DefaultProvider: "openai",
		Providers: map[string]ProviderConfig{
			"openai": {
				Type:         "openai",
				EnvVar:       "OPENAI_API_KEY",
				DefaultModel: "gpt-4",
			},
			"custom": {
				Type:         "custom",
				EnvVar:       "CUSTOM_API_KEY",
				DefaultModel: "custom-model",
			},
		},
	}
	registry, err := NewRegistry(config)
	require.NoError(t, err, "Failed to create registry")

	// Register a client dynamically (overriding existing configuration)
	err = registry.RegisterClient("custom/special-model", ClientConfig{
		APIKey: "override-key",
		Model:  "special-model",
	})
	require.NoError(t, err, "Failed to register client")

	// Verify the client was registered
	client, err := registry.GetClient("custom/special-model")
	require.NoError(t, err, "Failed to get registered client")

	assert.Equal(t, "special-model", client.GetModel(), "Model mismatch")
}

func TestRegistry_GetClient(t *testing.T) {
	// Set up environment variable
	t.Setenv("OPENAI_API_KEY", "test-key")

	config := RegistryConfig{
		DefaultProvider: "openai",
		Providers: map[string]ProviderConfig{
			"openai": {
				Type:         "openai",
				EnvVar:       "OPENAI_API_KEY",
				DefaultModel: "gpt-4",
			},
		},
	}
	registry, err := NewRegistry(config)
	require.NoError(t, err, "Failed to create registry")

	// Initialize providers
	err = registry.InitializeProviders()
	require.NoError(t, err, "Failed to initialize providers")

	// Test getting client by empty model (should return default)
	client, err := registry.GetClient("")
	assert.NoError(t, err, "Failed to get default client")
	assert.NotNil(t, client, "Expected non-nil client")

	// Test getting client by model string
	client, err = registry.GetClient("openai/gpt-4")
	assert.NoError(t, err, "Failed to get client by model string")
	assert.NotNil(t, client, "Expected non-nil client")

	// Test getting non-existent provider
	_, err = registry.GetClient("nonexistent/model")
	assert.Error(t, err, "Expected error for non-existent provider")
}

func TestRegistry_InitializeProviders(t *testing.T) {
	// Set up environment variables for testing
	originalOpenAI := os.Getenv("OPENAI_API_KEY")
	defer os.Setenv("OPENAI_API_KEY", originalOpenAI)

	os.Setenv("OPENAI_API_KEY", "test-openai-key")

	config := RegistryConfig{
		DefaultProvider: "openai",
		Providers:       DefaultProviders,
		DefaultMiddleware: []Middleware{
			TimeoutMiddleware(30 * time.Second),
		},
	}
	registry, err := NewRegistry(config)
	require.NoError(t, err, "Failed to create registry")

	err = registry.InitializeProviders()
	require.NoError(t, err, "Failed to initialize providers")

	// Check that OpenAI client was created
	providers := registry.GetRegisteredProviders()
	assert.Contains(t, providers, "openai", "Expected OpenAI provider to be registered")

	// Test getting the client
	client, err := registry.GetClient("")
	assert.NoError(t, err, "Failed to get default client")

	// Test functionality
	ctx := context.Background()
	response, err := client.Complete(ctx, "test prompt", nil)
	assert.NoError(t, err, "Failed to complete request")
	assert.NotEmpty(t, response, "Expected non-empty response")
}

func TestRegistry_CachedClient(t *testing.T) {
	// Set up environment variable
	t.Setenv("OPENAI_API_KEY", "test-key")

	config := RegistryConfig{
		DefaultProvider: "openai",
		Providers: map[string]ProviderConfig{
			"openai": {
				Type:         "openai",
				EnvVar:       "OPENAI_API_KEY",
				DefaultModel: "gpt-4",
			},
		},
	}
	registry, err := NewRegistry(config)
	require.NoError(t, err, "Failed to create registry")

	// Initialize providers
	err = registry.InitializeProviders()
	require.NoError(t, err, "Failed to initialize providers")

	// Get client twice with same model string
	client1, err := registry.GetClient("openai/gpt-4")
	require.NoError(t, err, "Failed to get client")

	client2, err := registry.GetClient("openai/gpt-4")
	require.NoError(t, err, "Failed to get client second time")

	// Should be the same instance (cached)
	assert.Same(t, client1, client2, "Expected same client instance from cache")
}

func TestRegistry_CustomProvider(t *testing.T) {
	// Register a custom provider factory
	RegisterProviderFactory("custom", func(config ClientConfig) (CoreLLM, error) {
		return &customProvider{
			apiKey: config.APIKey,
			model:  config.Model,
		}, nil
	})

	// Set environment variable
	t.Setenv("CUSTOM_API_KEY", "custom-key")

	config := RegistryConfig{
		DefaultProvider: "custom",
		Providers: map[string]ProviderConfig{
			"custom": {
				Type:         "custom",
				EnvVar:       "CUSTOM_API_KEY",
				DefaultModel: "custom-model",
			},
		},
	}

	registry, err := NewRegistry(config)
	require.NoError(t, err, "Failed to create registry")

	// Initialize providers
	err = registry.InitializeProviders()
	require.NoError(t, err, "Failed to initialize providers")

	client, err := registry.GetClient("")
	require.NoError(t, err, "Failed to get custom client")

	assert.Equal(t, "custom-model", client.GetModel(), "Model mismatch")
}

// Mock custom provider for testing
type customProvider struct {
	apiKey string
	model  string
}

func (p *customProvider) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
	return "custom response", 10, 10, nil
}

func (p *customProvider) GetModel() string {
	return p.model
}

func (p *customProvider) SetModel(m string) {
	p.model = m
}

func TestRegistry_EnvironmentVariables(t *testing.T) {
	tests := []struct {
		name        string
		provider    string
		envVar      string
		envValue    string
		expectError bool
	}{
		{
			name:        "openai with api key",
			provider:    "openai",
			envVar:      "OPENAI_API_KEY",
			envValue:    "test-key",
			expectError: false,
		},
		{
			name:        "anthropic with api key",
			provider:    "anthropic",
			envVar:      "ANTHROPIC_API_KEY",
			envValue:    "test-key",
			expectError: false,
		},
		{
			name:        "google with credentials",
			provider:    "google",
			envVar:      "GOOGLE_APPLICATION_CREDENTIALS",
			envValue:    "/path/to/creds",
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set environment variable
			t.Setenv(tt.envVar, tt.envValue)

			config := RegistryConfig{
				DefaultProvider: tt.provider,
				Providers: map[string]ProviderConfig{
					tt.provider: {
						Type:         tt.provider,
						EnvVar:       tt.envVar,
						DefaultModel: "test-model",
					},
				},
			}

			registry, err := NewRegistry(config)
			require.NoError(t, err, "Failed to create registry")

			err = registry.InitializeProviders()
			if tt.expectError {
				assert.Error(t, err, "Expected error but got none")
			} else {
				assert.NoError(t, err, "Unexpected error")
			}
		})
	}
}
