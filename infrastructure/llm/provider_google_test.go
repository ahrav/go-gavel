package llm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestNewGoogleProvider tests the creation of a new Google provider.
// It covers various scenarios, including valid and invalid configurations.
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

// TestGoogleProvider_GetSetModel tests the GetModel and SetModel methods
// of the Google provider.
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

// TestIsFilePath tests the isFilePath helper function.
func TestIsFilePath(t *testing.T) {
	tests := []struct {
		input    string
		expected bool
	}{
		{"api-key-string", false},
		{"/path/to/file", true},
		{"C:\\path\\to\\file", true},
		{"credentials.json", true},
		{"sk-1234567890abcdef", false},
		{"./relative/path", true},
		{"../parent/path", true},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := isFilePath(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestBuildGenerateContentRequest tests the buildGenerateContentRequest method.
func TestBuildGenerateContentRequest(t *testing.T) {
	provider := &googleProvider{
		model: "gemini-pro",
	}

	t.Run("basic prompt", func(t *testing.T) {
		prompt := "Hello, world!"
		opts := map[string]any{}

		content := provider.buildGenerateContentRequest(prompt, opts)

		require.Len(t, content, 1)
		// Note: We can't easily test the internal structure without exposing more details
		// This is a basic structure test
		assert.NotNil(t, content[0])
	})

	t.Run("with system prompt", func(t *testing.T) {
		prompt := "Hello, world!"
		opts := map[string]any{
			"system_prompt": "You are a helpful assistant.",
		}

		content := provider.buildGenerateContentRequest(prompt, opts)

		require.Len(t, content, 1)
		// The system prompt should be prepended to the user prompt in Gemini
		assert.NotNil(t, content[0])
	})

	t.Run("empty system prompt ignored", func(t *testing.T) {
		prompt := "Hello, world!"
		opts := map[string]any{
			"system_prompt": "",
		}

		content := provider.buildGenerateContentRequest(prompt, opts)

		require.Len(t, content, 1)
		assert.NotNil(t, content[0])
	})
}

func TestBuildGenerationConfig(t *testing.T) {
	provider := &googleProvider{
		model: "gemini-pro",
	}

	t.Run("empty options", func(t *testing.T) {
		opts := map[string]any{}
		config := provider.buildGenerationConfig(opts)

		assert.NotNil(t, config)
		assert.Nil(t, config.Temperature)
		assert.Equal(t, int32(0), config.MaxOutputTokens)
		assert.Nil(t, config.TopP)
		assert.Nil(t, config.TopK)
	})

	t.Run("valid temperature", func(t *testing.T) {
		opts := map[string]any{
			"temperature": 0.7,
		}
		config := provider.buildGenerationConfig(opts)

		assert.NotNil(t, config.Temperature)
		assert.Equal(t, float32(0.7), *config.Temperature)
	})

	t.Run("invalid temperature ignored", func(t *testing.T) {
		opts := map[string]any{
			"temperature": 3.0, // Too high
		}
		config := provider.buildGenerationConfig(opts)

		assert.Nil(t, config.Temperature)
	})

	t.Run("valid max_tokens", func(t *testing.T) {
		opts := map[string]any{
			"max_tokens": 1000,
		}
		config := provider.buildGenerationConfig(opts)

		assert.Equal(t, int32(1000), config.MaxOutputTokens)
	})

	t.Run("invalid max_tokens ignored", func(t *testing.T) {
		opts := map[string]any{
			"max_tokens": -100, // Negative
		}
		config := provider.buildGenerationConfig(opts)

		assert.Equal(t, int32(0), config.MaxOutputTokens)
	})

	t.Run("valid top_p", func(t *testing.T) {
		opts := map[string]any{
			"top_p": 0.9,
		}
		config := provider.buildGenerationConfig(opts)

		assert.NotNil(t, config.TopP)
		assert.Equal(t, float32(0.9), *config.TopP)
	})

	t.Run("valid top_k", func(t *testing.T) {
		opts := map[string]any{
			"top_k": 20,
		}
		config := provider.buildGenerationConfig(opts)

		assert.NotNil(t, config.TopK)
		assert.Equal(t, float32(20), *config.TopK)
	})

	t.Run("all valid options", func(t *testing.T) {
		opts := map[string]any{
			"temperature": 0.8,
			"max_tokens":  2000,
			"top_p":       0.95,
			"top_k":       40,
		}
		config := provider.buildGenerationConfig(opts)

		assert.NotNil(t, config.Temperature)
		assert.Equal(t, float32(0.8), *config.Temperature)
		assert.Equal(t, int32(2000), config.MaxOutputTokens)
		assert.NotNil(t, config.TopP)
		assert.Equal(t, float32(0.95), *config.TopP)
		assert.NotNil(t, config.TopK)
		assert.Equal(t, float32(40), *config.TopK)
	})
}

func TestHandleGeminiError(t *testing.T) {
	provider := &googleProvider{
		model: "gemini-pro",
	}

	tests := []struct {
		name        string
		inputError  error
		expectedMsg string
	}{
		{
			name:        "authentication error",
			inputError:  fmt.Errorf("401 Unauthorized"),
			expectedMsg: "gemini authentication failed",
		},
		{
			name:        "rate limit error",
			inputError:  fmt.Errorf("429 Too Many Requests"),
			expectedMsg: "gemini rate limit exceeded",
		},
		{
			name:        "model not found error",
			inputError:  fmt.Errorf("404 model not found"),
			expectedMsg: "gemini model 'gemini-pro' not found",
		},
		{
			name:        "server error",
			inputError:  fmt.Errorf("500 Internal Server Error"),
			expectedMsg: "gemini server error",
		},
		{
			name:        "content policy violation",
			inputError:  fmt.Errorf("content blocked by safety policy"),
			expectedMsg: "gemini content policy violation",
		},
		{
			name:        "timeout error",
			inputError:  fmt.Errorf("connection timeout"),
			expectedMsg: "gemini network error",
		},
		{
			name:        "generic error",
			inputError:  fmt.Errorf("unknown error"),
			expectedMsg: "gemini API request failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := provider.handleGeminiError(tt.inputError)
			assert.Contains(t, result.Error(), tt.expectedMsg)
		})
	}
}

// TestGoogleProvider_IntegrationWithMockServer provides a placeholder for
// integration tests with a mock server.
func TestGoogleProvider_IntegrationWithMockServer(t *testing.T) {
	t.Run("mock successful response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			response := map[string]interface{}{
				"candidates": []map[string]interface{}{
					{
						"content": map[string]interface{}{
							"parts": []map[string]interface{}{
								{
									"text": "Hello! This is a mock response from Gemini.",
								},
							},
						},
					},
				},
				"usageMetadata": map[string]interface{}{
					"promptTokenCount":     10,
					"candidatesTokenCount": 12,
				},
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(response)
		}))
		defer server.Close()

		assert.NotNil(t, server)
	})
}

// BenchmarkEstimateTokens benchmarks the token estimation performance.
func BenchmarkEstimateTokens(b *testing.B) {
	text := "This is a sample text for benchmarking token estimation performance"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		EstimateTokens(text)
	}
}

// BenchmarkBuildGenerationConfig benchmarks the performance of building the
// generation configuration.
func BenchmarkBuildGenerationConfig(b *testing.B) {
	provider := &googleProvider{
		model: "gemini-pro",
	}

	opts := map[string]any{
		"temperature": 0.7,
		"max_tokens":  1000,
		"top_p":       0.9,
		"top_k":       40,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.buildGenerationConfig(opts)
	}
}
