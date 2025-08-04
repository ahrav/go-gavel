package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockOpenAIResponse represents a mock response from the OpenAI API for testing.
type mockOpenAIResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// TestOpenAIProvider_DoRequest tests the DoRequest method of the OpenAI provider.
// It covers basic successful requests and requests with options.
func TestOpenAIProvider_DoRequest(t *testing.T) {
	tests := []struct {
		name              string
		prompt            string
		opts              map[string]any
		mockResponse      mockOpenAIResponse
		expectedResponse  string
		expectedTokensIn  int
		expectedTokensOut int
		expectError       bool
	}{
		{
			name:   "successful_basic_request",
			prompt: "Hello, world!",
			opts:   nil,
			mockResponse: mockOpenAIResponse{
				ID:      "chatcmpl-test123",
				Object:  "chat.completion",
				Created: 1677652288,
				Model:   "gpt-4",
				Choices: []struct {
					Index   int `json:"index"`
					Message struct {
						Role    string `json:"role"`
						Content string `json:"content"`
					} `json:"message"`
					FinishReason string `json:"finish_reason"`
				}{
					{
						Index: 0,
						Message: struct {
							Role    string `json:"role"`
							Content string `json:"content"`
						}{
							Role:    "assistant",
							Content: "Hello! How can I help you today?",
						},
						FinishReason: "stop",
					},
				},
				Usage: struct {
					PromptTokens     int `json:"prompt_tokens"`
					CompletionTokens int `json:"completion_tokens"`
					TotalTokens      int `json:"total_tokens"`
				}{
					PromptTokens:     10,
					CompletionTokens: 9,
					TotalTokens:      19,
				},
			},
			expectedResponse:  "Hello! How can I help you today?",
			expectedTokensIn:  10,
			expectedTokensOut: 9,
			expectError:       false,
		},
		{
			name:   "request_with_system_prompt",
			prompt: "What's the weather like?",
			opts: map[string]any{
				"system_prompt": "You are a helpful weather assistant.",
				"temperature":   float32(0.7),
				"max_tokens":    100,
			},
			mockResponse: mockOpenAIResponse{
				ID:      "chatcmpl-test456",
				Object:  "chat.completion",
				Created: 1677652288,
				Model:   "gpt-4",
				Choices: []struct {
					Index   int `json:"index"`
					Message struct {
						Role    string `json:"role"`
						Content string `json:"content"`
					} `json:"message"`
					FinishReason string `json:"finish_reason"`
				}{
					{
						Index: 0,
						Message: struct {
							Role    string `json:"role"`
							Content string `json:"content"`
						}{
							Role:    "assistant",
							Content: "I'd be happy to help with weather information! However, I don't have access to real-time weather data.",
						},
						FinishReason: "stop",
					},
				},
				Usage: struct {
					PromptTokens     int `json:"prompt_tokens"`
					CompletionTokens int `json:"completion_tokens"`
					TotalTokens      int `json:"total_tokens"`
				}{
					PromptTokens:     25,
					CompletionTokens: 22,
					TotalTokens:      47,
				},
			},
			expectedResponse:  "I'd be happy to help with weather information! However, I don't have access to real-time weather data.",
			expectedTokensIn:  25,
			expectedTokensOut: 22,
			expectError:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock HTTP server
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// Verify request method and path
				assert.Equal(t, "POST", r.Method)
				assert.Equal(t, "/v1/chat/completions", r.URL.Path)

				// Verify authorization header
				authHeader := r.Header.Get("Authorization")
				assert.Contains(t, authHeader, "Bearer test-api-key")

				// Return mock response
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(tt.mockResponse)
			}))
			defer server.Close()

			// Create provider with mock server URL
			config := ClientConfig{
				APIKey:  "test-api-key",
				Model:   "gpt-4",
				BaseURL: server.URL + "/v1",
			}

			provider, err := newOpenAIProvider(config)
			require.NoError(t, err)

			// Make request
			response, tokensIn, tokensOut, err := provider.DoRequest(context.Background(), tt.prompt, tt.opts)

			if tt.expectError {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)
			assert.Equal(t, tt.expectedResponse, response)
			assert.Equal(t, tt.expectedTokensIn, tokensIn)
			assert.Equal(t, tt.expectedTokensOut, tokensOut)
		})
	}
}

// TestOpenAIProvider_ErrorHandling tests the error handling of the OpenAI provider.
// It covers various error scenarios, such as authentication and rate limiting.
func TestOpenAIProvider_ErrorHandling(t *testing.T) {
	tests := []struct {
		name           string
		statusCode     int
		responseBody   string
		expectedErrMsg string
	}{
		{
			name:       "authentication_error",
			statusCode: 401,
			responseBody: `{
				"error": {
					"message": "Invalid API key provided",
					"type": "invalid_request_error",
					"code": "invalid_api_key"
				}
			}`,
			expectedErrMsg: "authentication failed - check API key",
		},
		{
			name:       "rate_limit_error",
			statusCode: 429,
			responseBody: `{
				"error": {
					"message": "Rate limit exceeded",
					"type": "insufficient_quota",
					"code": "rate_limit_exceeded"
				}
			}`,
			expectedErrMsg: "rate limit exceeded - consider retry with backoff",
		},
		{
			name:       "server_error",
			statusCode: 500,
			responseBody: `{
				"error": {
					"message": "Internal server error",
					"type": "server_error"
				}
			}`,
			expectedErrMsg: "server error - retry may succeed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tt.statusCode)
				fmt.Fprint(w, tt.responseBody)
			}))
			defer server.Close()

			// Create provider with mock server URL
			config := ClientConfig{
				APIKey:  "test-api-key",
				Model:   "gpt-4",
				BaseURL: server.URL + "/v1",
			}

			provider, err := newOpenAIProvider(config)
			require.NoError(t, err)

			// Make request that should fail
			_, _, _, err = provider.DoRequest(context.Background(), "test prompt", nil)

			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.expectedErrMsg)
		})
	}
}

// TestOpenAIProvider_ContextCancellation tests that the OpenAI provider
// correctly handles context cancellation.
func TestOpenAIProvider_ContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("Server handler should not be called due to context cancellation")
	}))
	defer server.Close()

	config := ClientConfig{
		APIKey:  "test-api-key",
		Model:   "gpt-4",
		BaseURL: server.URL + "/v1",
	}

	provider, err := newOpenAIProvider(config)
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, _, _, err = provider.DoRequest(ctx, "test prompt", nil)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "context canceled")
}

// TestOpenAIProvider_Configuration tests the configuration of the OpenAI provider.
// It covers scenarios like missing API keys and default model selection.
func TestOpenAIProvider_Configuration(t *testing.T) {
	t.Run("missing_api_key", func(t *testing.T) {
		config := ClientConfig{
			Model: "gpt-4",
		}

		_, err := newOpenAIProvider(config)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "OpenAI API key cannot be empty")
	})

	t.Run("default_model", func(t *testing.T) {
		config := ClientConfig{
			APIKey: "test-key",
		}

		provider, err := newOpenAIProvider(config)
		require.NoError(t, err)
		assert.Equal(t, "gpt-3.5-turbo", provider.GetModel())
	})

	t.Run("custom_model", func(t *testing.T) {
		config := ClientConfig{
			APIKey: "test-key",
			Model:  "gpt-3.5-turbo",
		}

		provider, err := newOpenAIProvider(config)
		require.NoError(t, err)
		assert.Equal(t, "gpt-3.5-turbo", provider.GetModel())
	})

	t.Run("model_update", func(t *testing.T) {
		config := ClientConfig{
			APIKey: "test-key",
			Model:  "gpt-4",
		}

		provider, err := newOpenAIProvider(config)
		require.NoError(t, err)

		provider.SetModel("gpt-3.5-turbo")
		assert.Equal(t, "gpt-3.5-turbo", provider.GetModel())
	})
}

// TestOpenAIProvider_Integration tests the OpenAI provider against the live API.
// These tests are skipped if the OPENAI_API_KEY environment variable is not set.
func TestOpenAIProvider_Integration(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping integration tests: OPENAI_API_KEY environment variable not set")
	}

	t.Run("real_api_basic_request", func(t *testing.T) {
		config := ClientConfig{
			APIKey: apiKey,
			Model:  "gpt-3.5-turbo", // Use a cheaper model for testing
		}

		provider, err := newOpenAIProvider(config)
		require.NoError(t, err)

		response, tokensIn, tokensOut, err := provider.DoRequest(
			context.Background(),
			"Say 'Hello, World!' and nothing else.",
			map[string]any{
				"max_tokens":  10,
				"temperature": float32(0.1), // Low temperature for consistent responses
			},
		)

		require.NoError(t, err)
		assert.NotEmpty(t, response)
		assert.Greater(t, tokensIn, 0)
		assert.Greater(t, tokensOut, 0)
		t.Logf("Response: %s (tokens in: %d, out: %d)", response, tokensIn, tokensOut)
	})

	t.Run("real_api_with_system_prompt", func(t *testing.T) {
		config := ClientConfig{
			APIKey: apiKey,
			Model:  "gpt-3.5-turbo",
		}

		provider, err := newOpenAIProvider(config)
		require.NoError(t, err)

		response, tokensIn, tokensOut, err := provider.DoRequest(
			context.Background(),
			"What is 2+2?",
			map[string]any{
				"system_prompt": "You are a helpful math assistant. Always provide direct numerical answers.",
				"max_tokens":    20,
				"temperature":   float32(0.1),
			},
		)

		require.NoError(t, err)
		assert.NotEmpty(t, response)
		assert.Greater(t, tokensIn, 0)
		assert.Greater(t, tokensOut, 0)
		assert.Contains(t, response, "4") // Should contain the answer
		t.Logf("Response: %s (tokens in: %d, out: %d)", response, tokensIn, tokensOut)
	})

	t.Run("real_api_invalid_key", func(t *testing.T) {
		config := ClientConfig{
			APIKey: "invalid-key-test",
			Model:  "gpt-3.5-turbo",
		}

		provider, err := newOpenAIProvider(config)
		require.NoError(t, err)

		_, _, _, err = provider.DoRequest(
			context.Background(),
			"Test prompt",
			nil,
		)

		require.Error(t, err)
		assert.Contains(t, err.Error(), "authentication failed")
	})
}

// TestOpenAIProvider_Performance tests the performance of the OpenAI provider
// under concurrent requests.
func TestOpenAIProvider_Performance(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := mockOpenAIResponse{
			ID:      "test-perf",
			Object:  "chat.completion",
			Created: 1677652288,
			Model:   "gpt-4",
			Choices: []struct {
				Index   int `json:"index"`
				Message struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"message"`
				FinishReason string `json:"finish_reason"`
			}{
				{
					Message: struct {
						Role    string `json:"role"`
						Content string `json:"content"`
					}{
						Role:    "assistant",
						Content: "Test response",
					},
				},
			},
			Usage: struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			}{
				PromptTokens:     5,
				CompletionTokens: 2,
				TotalTokens:      7,
			},
		}
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	config := ClientConfig{
		APIKey:  "test-key",
		Model:   "gpt-4",
		BaseURL: server.URL + "/v1",
	}

	provider, err := newOpenAIProvider(config)
	require.NoError(t, err)

	// Test multiple concurrent requests
	t.Run("concurrent_requests", func(t *testing.T) {
		const numRequests = 10
		responses := make(chan struct{}, numRequests)

		for i := 0; i < numRequests; i++ {
			go func(id int) {
				_, _, _, err := provider.DoRequest(
					context.Background(),
					fmt.Sprintf("Request %d", id),
					nil,
				)
				assert.NoError(t, err)
				responses <- struct{}{}
			}(i)
		}

		// Wait for all requests to complete
		for i := 0; i < numRequests; i++ {
			<-responses
		}
	})
}

// TestOpenAIProvider_TypeSafety tests the provider's ability to handle various
// data types for options, ensuring type safety and graceful degradation.
func TestOpenAIProvider_TypeSafety(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := mockOpenAIResponse{
			ID:      "test-type-safety",
			Object:  "chat.completion",
			Created: 1677652288,
			Model:   "gpt-3.5-turbo",
			Choices: []struct {
				Index   int `json:"index"`
				Message struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"message"`
				FinishReason string `json:"finish_reason"`
			}{
				{
					Message: struct {
						Role    string `json:"role"`
						Content string `json:"content"`
					}{
						Role:    "assistant",
						Content: "Test response",
					},
				},
			},
			Usage: struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			}{
				PromptTokens:     5,
				CompletionTokens: 2,
				TotalTokens:      7,
			},
		}
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	config := ClientConfig{
		APIKey:  "test-key",
		Model:   "gpt-3.5-turbo",
		BaseURL: server.URL + "/v1",
	}

	provider, err := newOpenAIProvider(config)
	require.NoError(t, err)

	tests := []struct {
		name string
		opts map[string]any
	}{
		{
			name: "mixed_numeric_types",
			opts: map[string]any{
				"temperature":       float64(0.8), // float64 instead of float32
				"max_tokens":        int64(100),   // int64 instead of int
				"top_p":             float32(0.9), // correct type
				"frequency_penalty": int(1),       // int instead of float32
				"presence_penalty":  float64(0.5), // float64 instead of float32
			},
		},
		{
			name: "invalid_bounds_should_be_clamped",
			opts: map[string]any{
				"temperature":       float32(-1.0), // Should be clamped to 0
				"top_p":             float32(1.5),  // Should be ignored (> 1)
				"frequency_penalty": float32(3.0),  // Should be ignored (> 2)
				"presence_penalty":  float32(-3.0), // Should be ignored (< -2)
			},
		},
		{
			name: "invalid_types_should_be_ignored",
			opts: map[string]any{
				"temperature":       "invalid",        // string - should be ignored
				"max_tokens":        "100",            // string - should be ignored
				"top_p":             []int{1, 2, 3},   // slice - should be ignored
				"frequency_penalty": map[string]int{}, // map - should be ignored
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// This should not panic despite type mismatches
			_, _, _, err := provider.DoRequest(
				context.Background(),
				"Test prompt",
				tt.opts,
			)
			assert.NoError(t, err)
		})
	}
}

// TestOpenAIProvider_ThreadSafety tests the thread safety of the OpenAI provider,
// particularly around concurrent access to the model field.
func TestOpenAIProvider_ThreadSafety(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := mockOpenAIResponse{
			ID:      "test-thread-safety",
			Object:  "chat.completion",
			Created: 1677652288,
			Model:   "gpt-3.5-turbo",
			Choices: []struct {
				Index   int `json:"index"`
				Message struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"message"`
				FinishReason string `json:"finish_reason"`
			}{
				{
					Message: struct {
						Role    string `json:"role"`
						Content string `json:"content"`
					}{
						Role:    "assistant",
						Content: "Thread safe response",
					},
				},
			},
			Usage: struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			}{
				PromptTokens:     5,
				CompletionTokens: 3,
				TotalTokens:      8,
			},
		}
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	config := ClientConfig{
		APIKey:  "test-key",
		Model:   "gpt-3.5-turbo",
		BaseURL: server.URL + "/v1",
	}

	provider, err := newOpenAIProvider(config)
	require.NoError(t, err)

	const numGoroutines = 10
	const numOperations = 100

	// Test concurrent model reads and writes
	done := make(chan bool, numGoroutines*2)

	// Concurrent readers
	for i := 0; i < numGoroutines; i++ {
		go func() {
			for j := 0; j < numOperations; j++ {
				model := provider.GetModel()
				assert.NotEmpty(t, model)
			}
			done <- true
		}()
	}

	// Concurrent writers
	models := []string{"gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"}
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			for j := 0; j < numOperations; j++ {
				provider.SetModel(models[j%len(models)])
			}
			done <- true
		}(i)
	}

	// Wait for all goroutines to complete
	for i := 0; i < numGoroutines*2; i++ {
		<-done
	}

	// Verify final state is consistent
	finalModel := provider.GetModel()
	assert.Contains(t, models, finalModel)
}

// TestOpenAIProvider_TokenHandlingFallback tests the fallback mechanism for
// token estimation when the API response does not provide usage information.
func TestOpenAIProvider_TokenHandlingFallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := map[string]interface{}{
			"id":      "test-fallback",
			"object":  "chat.completion",
			"created": 1677652288,
			"model":   "gpt-3.5-turbo",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "Fallback response",
					},
					"finish_reason": "stop",
				},
			},
		}
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	config := ClientConfig{
		APIKey:  "test-key",
		Model:   "gpt-3.5-turbo",
		BaseURL: server.URL + "/v1",
	}

	provider, err := newOpenAIProvider(config)
	require.NoError(t, err)

	response, tokensIn, tokensOut, err := provider.DoRequest(
		context.Background(),
		"Test prompt for fallback",
		nil,
	)

	require.NoError(t, err)
	assert.Equal(t, "Fallback response", response)

	assert.Greater(t, tokensIn, 0)
	assert.Greater(t, tokensOut, 0)

	assert.InDelta(t, 6, tokensIn, 2)
	assert.InDelta(t, 4, tokensOut, 2)
}

// TestOpenAIProvider_TimeoutConfiguration tests the timeout configuration of
// the OpenAI provider.
func TestOpenAIProvider_TimeoutConfiguration(t *testing.T) {
	t.Run("timeout_configured", func(t *testing.T) {
		config := ClientConfig{
			APIKey:  "test-key",
			Model:   "gpt-3.5-turbo",
			Timeout: 30 * time.Second,
		}

		provider, err := newOpenAIProvider(config)
		require.NoError(t, err)
		assert.NotNil(t, provider)
	})

	t.Run("no_timeout_configured", func(t *testing.T) {
		config := ClientConfig{
			APIKey: "test-key",
			Model:  "gpt-3.5-turbo",
		}

		provider, err := newOpenAIProvider(config)
		require.NoError(t, err)
		assert.NotNil(t, provider)
	})
}

// TestSafeTypeConversion tests the internal helper functions for safe type
// conversion of options.
func TestSafeTypeConversion(t *testing.T) {
	t.Run("safeFloat32", func(t *testing.T) {
		tests := []struct {
			input    any
			expected float32
			ok       bool
		}{
			{float32(1.5), 1.5, true},
			{float64(2.5), 2.5, true},
			{int(3), 3.0, true},
			{int64(4), 4.0, true},
			{"invalid", 0, false},
			{nil, 0, false},
		}

		for _, tt := range tests {
			result, ok := safeFloat32(tt.input)
			assert.Equal(t, tt.ok, ok)
			if tt.ok {
				assert.Equal(t, tt.expected, result)
			}
		}
	})

	t.Run("safeInt", func(t *testing.T) {
		tests := []struct {
			input    any
			expected int
			ok       bool
		}{
			{int(1), 1, true},
			{int64(2), 2, true},
			{float32(3.9), 3, true}, // Should truncate
			{float64(4.1), 4, true}, // Should truncate
			{"invalid", 0, false},
			{nil, 0, false},
		}

		for _, tt := range tests {
			result, ok := safeInt(tt.input)
			assert.Equal(t, tt.ok, ok)
			if tt.ok {
				assert.Equal(t, tt.expected, result)
			}
		}
	})

	t.Run("validateTemperature", func(t *testing.T) {
		tests := []struct {
			input    float32
			expected float32
		}{
			{-1.0, 0.0}, // Below minimum
			{0.0, 0.0},  // At minimum
			{1.0, 1.0},  // Valid middle
			{2.0, 2.0},  // At maximum
			{3.0, 2.0},  // Above maximum
		}

		for _, tt := range tests {
			result := validateTemperature(tt.input)
			assert.Equal(t, tt.expected, result)
		}
	})

	t.Run("validateMaxTokens", func(t *testing.T) {
		tests := []struct {
			input    int
			expected int
		}{
			{-10, 0},   // Negative
			{0, 0},     // Zero
			{100, 100}, // Positive
		}

		for _, tt := range tests {
			result := validateMaxTokens(tt.input)
			assert.Equal(t, tt.expected, result)
		}
	})
}
