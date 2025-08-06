package llm

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestNewGoogleProvider tests the behavior of the newGoogleProvider function.
// It ensures that the provider is created correctly with valid configurations
// and that it returns an error for invalid configurations, such as an empty
// API key or a file path used as an API key.
func TestNewGoogleProvider(t *testing.T) {
	tests := []struct {
		name           string
		config         ClientConfig
		expectError    bool
		expectedModel  string
		expectedAPIKey string
	}{
		{
			name: "valid API key configuration",
			config: ClientConfig{
				APIKey: "test-api-key",
				Model:  "gemini-pro",
			},
			expectError:    false,
			expectedModel:  "gemini-pro",
			expectedAPIKey: "test-api-key",
		},
		{
			name: "default model when not specified",
			config: ClientConfig{
				APIKey: "test-api-key",
			},
			expectError:    false,
			expectedModel:  GoogleDefaultModel,
			expectedAPIKey: "test-api-key",
		},
		{
			name: "file path authentication should error",
			config: ClientConfig{
				APIKey: "/path/to/credentials.json",
				Model:  "gemini-pro",
			},
			expectError: true,
		},
		{
			name: "empty API key should error",
			config: ClientConfig{
				APIKey: "",
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, err := newGoogleProvider(tt.config)

			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, provider)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, provider)

			googleProvider, ok := provider.(*googleProvider)
			require.True(t, ok)

			assert.Equal(t, tt.expectedModel, googleProvider.GetModel())
		})
	}
}

// TestGoogleProvider_GetSetModel tests the GetModel and SetModel methods of the
// Google provider, ensuring that the model can be retrieved and updated
// correctly.
func TestGoogleProvider_GetSetModel(t *testing.T) {
	provider, err := newGoogleProvider(ClientConfig{
		APIKey: "test-key",
		Model:  "gemini-pro",
	})
	require.NoError(t, err)

	assert.Equal(t, "gemini-pro", provider.GetModel())

	provider.SetModel(GoogleDefaultModel)
	assert.Equal(t, GoogleDefaultModel, provider.GetModel())
}

// TestBuildGenerateContentRequest tests the construction of a content generation
// request. It verifies that the request is correctly assembled with and without
// a system prompt.
func TestBuildGenerateContentRequest(t *testing.T) {
	provider := &googleProvider{
		BaseProvider: BaseProvider{model: "gemini-pro"},
	}

	t.Run("basic prompt", func(t *testing.T) {
		prompt := "Hello, world!"
		options := RequestOptions{Model: "gemini-pro"}

		content := provider.buildGenerateContentRequest(prompt, options)

		require.Len(t, content, 1)
		assert.NotNil(t, content[0])
	})

	t.Run("with system prompt", func(t *testing.T) {
		prompt := "Hello, world!"
		options := RequestOptions{
			Model:  "gemini-pro",
			System: "You are a helpful assistant.",
		}

		content := provider.buildGenerateContentRequest(prompt, options)

		require.Len(t, content, 1)
		assert.NotNil(t, content[0])
	})

	t.Run("empty system prompt ignored", func(t *testing.T) {
		prompt := "Hello, world!"
		options := RequestOptions{Model: "gemini-pro"}

		content := provider.buildGenerateContentRequest(prompt, options)

		require.Len(t, content, 1)
		assert.NotNil(t, content[0])
	})
}

// TestBuildGenerationConfig tests the construction of the generation
// configuration from request options. It ensures that parameters like
// temperature, max tokens, top-p, and top-k are correctly translated into the
// configuration structure.
func TestBuildGenerationConfig(t *testing.T) {
	provider := &googleProvider{
		BaseProvider: BaseProvider{model: "gemini-pro"},
	}

	t.Run("empty options", func(t *testing.T) {
		options := RequestOptions{Model: "gemini-pro"}
		config := provider.buildGenerationConfig(options)

		assert.NotNil(t, config)
		assert.Nil(t, config.Temperature)
		assert.Equal(t, int32(0), config.MaxOutputTokens)
		assert.Nil(t, config.TopP)
		assert.Nil(t, config.TopK)
	})

	t.Run("valid temperature", func(t *testing.T) {
		temp := 0.7
		options := RequestOptions{
			Model:       "gemini-pro",
			Temperature: &temp,
		}
		config := provider.buildGenerationConfig(options)

		assert.NotNil(t, config.Temperature)
		assert.Equal(t, float32(0.7), *config.Temperature)
	})

	t.Run("valid max_tokens", func(t *testing.T) {
		options := RequestOptions{
			Model:     "gemini-pro",
			MaxTokens: 1000,
		}
		config := provider.buildGenerationConfig(options)

		assert.Equal(t, int32(1000), config.MaxOutputTokens)
	})

	t.Run("valid top_p", func(t *testing.T) {
		topP := 0.9
		options := RequestOptions{
			Model: "gemini-pro",
			TopP:  &topP,
		}
		config := provider.buildGenerationConfig(options)

		assert.NotNil(t, config.TopP)
		assert.Equal(t, float32(0.9), *config.TopP)
	})

	t.Run("valid top_k", func(t *testing.T) {
		options := RequestOptions{
			Model: "gemini-pro",
			Extra: map[string]any{"top_k": 20},
		}
		config := provider.buildGenerationConfig(options)

		assert.NotNil(t, config.TopK)
		assert.Equal(t, float32(20), *config.TopK)
	})

	t.Run("all valid options", func(t *testing.T) {
		temp := 0.8
		topP := 0.95
		options := RequestOptions{
			Model:       "gemini-pro",
			Temperature: &temp,
			MaxTokens:   2000,
			TopP:        &topP,
			Extra:       map[string]any{"top_k": 40},
		}
		config := provider.buildGenerationConfig(options)

		assert.NotNil(t, config.Temperature)
		assert.Equal(t, float32(0.8), *config.Temperature)
		assert.Equal(t, int32(2000), config.MaxOutputTokens)
		assert.NotNil(t, config.TopP)
		assert.Equal(t, float32(0.95), *config.TopP)
		assert.NotNil(t, config.TopK)
		assert.Equal(t, float32(40), *config.TopK)
	})
}

// TestHandleError tests the error handling and classification mechanism.
// It ensures that different types of errors, such as context cancellation or
// unknown errors, are correctly categorized into the appropriate ProviderError
// type.
func TestHandleError(t *testing.T) {
	provider := &googleProvider{
		BaseProvider:    BaseProvider{model: "gemini-pro"},
		errorClassifier: &ErrorClassifier{Provider: "google"},
	}

	tests := []struct {
		name         string
		inputError   error
		expectedType ErrorType
	}{
		{
			name:         "context canceled",
			inputError:   context.Canceled,
			expectedType: ErrorTypeNetwork,
		},
		{
			name:         "context timeout",
			inputError:   context.DeadlineExceeded,
			expectedType: ErrorTypeNetwork,
		},
		{
			name:         "generic error",
			inputError:   fmt.Errorf("unknown error"),
			expectedType: ErrorTypeUnknown,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := provider.handleError(tt.inputError)
			provErr, ok := result.(*ProviderError)
			require.True(t, ok)
			assert.Equal(t, tt.expectedType, provErr.Type)
			assert.Equal(t, "google", provErr.Provider)
		})
	}
}

// TestGoogleProvider runs the full provider test suite for the Google provider.
// This suite covers standard behaviors like basic requests, handling of options,
// error management, and context cancellation. It requires the GOOGLE_API_KEY
// environment variable to be set.
func TestGoogleProvider(t *testing.T) {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		t.Skip("GOOGLE_API_KEY not set")
	}

	config := ClientConfig{
		APIKey: apiKey,
		Model:  GoogleDefaultModel,
	}

	suite := NewProviderTestSuite(t, "google", config)

	t.Run("BasicRequest", func(t *testing.T) { suite.TestBasicRequest() })
	t.Run("RequestWithOptions", func(t *testing.T) { suite.TestRequestWithOptions() })
	t.Run("ErrorHandling", func(t *testing.T) { suite.TestErrorHandling() })
	t.Run("ContextCancellation", func(t *testing.T) { suite.TestContextCancellation() })
	t.Run("ModelGetterSetter", func(t *testing.T) { suite.TestModelGetterSetter() })
}

// BenchmarkTokenCounter benchmarks the performance of the token estimation
// function.
func BenchmarkTokenCounter(b *testing.B) {
	text := "This is a sample text for benchmarking token estimation performance"
	counter := NewTokenCounter()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		counter.EstimateTokens(text)
	}
}

// BenchmarkBuildGenerationConfig benchmarks the performance of building the
// generation configuration.
func BenchmarkBuildGenerationConfig(b *testing.B) {
	provider := &googleProvider{
		BaseProvider: BaseProvider{model: "gemini-pro"},
	}

	temp := 0.7
	topP := 0.9
	options := RequestOptions{
		Model:       "gemini-pro",
		Temperature: &temp,
		MaxTokens:   1000,
		TopP:        &topP,
		Extra:       map[string]any{"top_k": 40},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.buildGenerationConfig(options)
	}
}
