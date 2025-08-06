package llm

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestNewRegistry tests the creation of a new registry.
func TestNewRegistry(t *testing.T) {
	config := RegistryConfig{
		DefaultProvider: "openai",
		Providers: map[string]ProviderConfig{
			"openai": {
				Type:         "openai",
				EnvVar:       "OPENAI_API_KEY",
				DefaultModel: "gpt-4.1",
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

// TestRegistry_RegisterClient tests the dynamic registration of a client.
func TestRegistry_RegisterClient(t *testing.T) {
	RegisterProviderFactory("custom", func(config ClientConfig) (CoreLLM, error) {
		return &customProvider{
			apiKey: config.APIKey,
			model:  config.Model,
		}, nil
	})

	t.Setenv("OPENAI_API_KEY", "test-key")
	t.Setenv("CUSTOM_API_KEY", "custom-key")

	config := RegistryConfig{
		DefaultProvider: "openai",
		Providers: map[string]ProviderConfig{
			"openai": {
				Type:         "openai",
				EnvVar:       "OPENAI_API_KEY",
				DefaultModel: "gpt-4.1",
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

	err = registry.RegisterClient("custom/special-model", ClientConfig{
		APIKey: "override-key",
		Model:  "special-model",
	})
	require.NoError(t, err, "Failed to register client")

	client, err := registry.GetClient("custom/special-model")
	require.NoError(t, err, "Failed to get registered client")

	assert.Equal(t, "special-model", client.GetModel(), "Model mismatch")
}

// TestRegistry_GetClient tests the retrieval of clients from the registry.
func TestRegistry_GetClient(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "test-key")

	config := RegistryConfig{
		DefaultProvider: "openai",
		Providers: map[string]ProviderConfig{
			"openai": {
				Type:         "openai",
				EnvVar:       "OPENAI_API_KEY",
				DefaultModel: "gpt-4.1",
			},
		},
	}
	registry, err := NewRegistry(config)
	require.NoError(t, err, "Failed to create registry")

	err = registry.InitializeProviders()
	require.NoError(t, err, "Failed to initialize providers")

	client, err := registry.GetDefaultClient()
	assert.NoError(t, err, "Failed to get default client")
	assert.NotNil(t, client, "Expected non-nil client")

	client, err = registry.GetClient("openai/gpt-4")
	assert.NoError(t, err, "Failed to get client by model string")
	assert.NotNil(t, client, "Expected non-nil client")

	_, err = registry.GetClient("nonexistent/model")
	assert.Error(t, err, "Expected error for non-existent provider")

	// Test that empty strings are rejected
	_, err = registry.GetClient("")
	assert.Error(t, err, "Expected error for empty string")
	assert.Contains(t, err.Error(), "provider specification cannot be empty", "Error message should mention empty string")
}

// TestRegistry_InitializeProviders tests the initialization of providers from config.
func TestRegistry_InitializeProviders(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	// Skip test if no real API key is available
	t.Skip("skipping integration test - requires valid API key")
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

	providers := registry.GetRegisteredProviders()
	assert.Contains(t, providers, "openai", "Expected OpenAI provider to be registered")

	client, err := registry.GetDefaultClient()
	assert.NoError(t, err, "Failed to get default client")

	ctx := context.Background()
	response, err := client.Complete(ctx, "test prompt", nil)
	assert.NoError(t, err, "Failed to complete request")
	assert.NotEmpty(t, response, "Expected non-empty response")
}

// TestRegistry_CachedClient tests that the registry caches and reuses clients.
func TestRegistry_CachedClient(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "test-key")

	config := RegistryConfig{
		DefaultProvider: "openai",
		Providers: map[string]ProviderConfig{
			"openai": {
				Type:         "openai",
				EnvVar:       "OPENAI_API_KEY",
				DefaultModel: "gpt-4.1",
			},
		},
	}
	registry, err := NewRegistry(config)
	require.NoError(t, err, "Failed to create registry")

	err = registry.InitializeProviders()
	require.NoError(t, err, "Failed to initialize providers")

	client1, err := registry.GetClient("openai/gpt-4")
	require.NoError(t, err, "Failed to get client")

	client2, err := registry.GetClient("openai/gpt-4")
	require.NoError(t, err, "Failed to get client second time")

	assert.Same(t, client1, client2, "Expected same client instance from cache")
}

// TestRegistry_CustomProvider tests the registration and use of a custom provider.
func TestRegistry_CustomProvider(t *testing.T) {
	RegisterProviderFactory("custom", func(config ClientConfig) (CoreLLM, error) {
		return &customProvider{
			apiKey: config.APIKey,
			model:  config.Model,
		}, nil
	})

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

	err = registry.InitializeProviders()
	require.NoError(t, err, "Failed to initialize providers")

	client, err := registry.GetDefaultClient()
	require.NoError(t, err, "Failed to get custom client")

	assert.Equal(t, "custom-model", client.GetModel(), "Model mismatch")
}

// customProvider is a mock provider for testing purposes.
type customProvider struct {
	apiKey string
	model  string
}

// DoRequest is a mock implementation of the DoRequest method.
func (p *customProvider) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
	return "custom response", 10, 10, nil
}

// GetModel is a mock implementation of the GetModel method.
func (p *customProvider) GetModel() string {
	return p.model
}

// SetModel is a mock implementation of the SetModel method.
func (p *customProvider) SetModel(m string) {
	p.model = m
}

// TestRegistry_EnvironmentVariables tests that the registry correctly uses
// environment variables to configure providers.
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
			envVar:      "GOOGLE_API_KEY", // Use API key instead of credentials file
			envValue:    "test-google-key",
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

// TestRegistry_ModelValidation tests that the registry validates models against supported models list.
func TestRegistry_ModelValidation(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "test-key")
	t.Setenv("CUSTOM_API_KEY", "custom-key")

	RegisterProviderFactory("custom", func(config ClientConfig) (CoreLLM, error) {
		return &customProvider{
			apiKey: config.APIKey,
			model:  config.Model,
		}, nil
	})

	config := RegistryConfig{
		DefaultProvider: "openai",
		Providers: map[string]ProviderConfig{
			"openai": {
				Type:            "openai",
				EnvVar:          "OPENAI_API_KEY",
				DefaultModel:    "gpt-4",
				SupportedModels: []string{"gpt-4", "gpt-3.5-turbo"},
			},
			"custom": {
				Type:         "custom",
				EnvVar:       "CUSTOM_API_KEY",
				DefaultModel: "custom-model",
				// No SupportedModels specified - should allow any model
			},
		},
	}

	registry, err := NewRegistry(config)
	require.NoError(t, err, "Failed to create registry")

	// Test valid model
	client, err := registry.GetClient("openai/gpt-4")
	assert.NoError(t, err, "Failed to get client with valid model")
	assert.NotNil(t, client, "Expected non-nil client")

	// Test another valid model
	client, err = registry.GetClient("openai/gpt-3.5-turbo")
	assert.NoError(t, err, "Failed to get client with valid model")
	assert.NotNil(t, client, "Expected non-nil client")

	// Test invalid model for provider with supported models list
	_, err = registry.GetClient("openai/invalid-model")
	assert.Error(t, err, "Expected error for invalid model")
	assert.Contains(t, err.Error(), "not supported by provider", "Error should mention unsupported model")
	assert.Contains(t, err.Error(), "invalid-model", "Error should mention the invalid model")

	// Test provider without supported models list (should allow any model)
	client, err = registry.GetClient("custom/any-model")
	assert.NoError(t, err, "Provider without supported models should allow any model")
	assert.NotNil(t, client, "Expected non-nil client")
}

// TestRegistry_GetDefaultClient tests the new GetDefaultClient method.
func TestRegistry_GetDefaultClient(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "test-key")

	config := RegistryConfig{
		DefaultProvider: "openai",
		Providers: map[string]ProviderConfig{
			"openai": {
				Type:         "openai",
				EnvVar:       "OPENAI_API_KEY",
				DefaultModel: "gpt-4.1",
			},
		},
	}

	registry, err := NewRegistry(config)
	require.NoError(t, err, "Failed to create registry")

	// Test GetDefaultClient
	client, err := registry.GetDefaultClient()
	assert.NoError(t, err, "Failed to get default client")
	assert.NotNil(t, client, "Expected non-nil client")
	assert.Equal(t, "gpt-4.1", client.GetModel(), "Default client should use default model")

	// Test that GetDefaultClient is equivalent to GetClient with explicit default provider/model
	explicitClient, err := registry.GetClient("openai/gpt-4.1")
	assert.NoError(t, err, "Failed to get explicit client")

	// Both should have same model
	assert.Equal(t, client.GetModel(), explicitClient.GetModel(), "Default and explicit clients should have same model")
}
