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

// TestOpenAIProvider_DoRequest tests the DoRequest method for the OpenAI provider.
// It verifies successful requests with and without optional parameters.
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
				"system":      "You are a helpful weather assistant.",
				"temperature": float32(0.7),
				"max_tokens":  100,
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
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "POST", r.Method)
				assert.Equal(t, "/v1/chat/completions", r.URL.Path)

				authHeader := r.Header.Get("Authorization")
				assert.Contains(t, authHeader, "Bearer test-api-key")

				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(tt.mockResponse)
			}))
			defer server.Close()

			config := ClientConfig{
				APIKey:  "test-api-key",
				Model:   "gpt-4",
				BaseURL: server.URL + "/v1",
			}

			provider, err := newOpenAIProvider(config)
			require.NoError(t, err)

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

// TestOpenAIProvider_ErrorHandling tests the error handling capabilities of the OpenAI provider.
// It ensures that API errors, such as authentication and rate limiting, are handled correctly.
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
			expectedErrMsg: "authentication failed",
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
			expectedErrMsg: "rate limit exceeded",
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
			expectedErrMsg: "server error",
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

			config := ClientConfig{
				APIKey:  "test-api-key",
				Model:   "gpt-4",
				BaseURL: server.URL + "/v1",
			}

			provider, err := newOpenAIProvider(config)
			require.NoError(t, err)

			_, _, _, err = provider.DoRequest(context.Background(), "test prompt", nil)

			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.expectedErrMsg)
		})
	}
}

// TestOpenAIProvider_ContextCancellation verifies that the OpenAI provider
// correctly handles request cancellation through context.
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

// TestOpenAIProvider_Configuration validates the configuration handling
// for the OpenAI provider, including API key validation and model management.
func TestOpenAIProvider_Configuration(t *testing.T) {
	t.Run("missing_api_key", func(t *testing.T) {
		config := ClientConfig{
			Model: "gpt-4",
		}

		_, err := newOpenAIProvider(config)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "API key cannot be empty")
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

// TestOpenAIProvider_Integration performs integration tests against the live OpenAI API.
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
				"temperature": float32(0.1), // Use low temperature for consistent responses.
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
				"system":      "You are a helpful math assistant. Always provide direct numerical answers.",
				"max_tokens":  20,
				"temperature": float32(0.1),
			},
		)

		require.NoError(t, err)
		assert.NotEmpty(t, response)
		assert.Greater(t, tokensIn, 0)
		assert.Greater(t, tokensOut, 0)
		assert.Contains(t, response, "4")
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

		for i := 0; i < numRequests; i++ {
			<-responses
		}
	})
}

// TestOpenAIProvider_TypeSafety ensures that the provider handles various
// data types for options gracefully, maintaining type safety.
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
			// This call should not panic despite type mismatches in options.
			_, _, _, err := provider.DoRequest(
				context.Background(),
				"Test prompt",
				tt.opts,
			)
			assert.NoError(t, err)
		})
	}
}

// TestOpenAIProvider_ThreadSafety verifies the thread-safe operations of the provider,
// focusing on concurrent access to shared fields like the model name.
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

	done := make(chan bool, numGoroutines*2)

	for i := 0; i < numGoroutines; i++ {
		go func() {
			for j := 0; j < numOperations; j++ {
				model := provider.GetModel()
				assert.NotEmpty(t, model)
			}
			done <- true
		}()
	}

	models := []string{"gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"}
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			for j := 0; j < numOperations; j++ {
				provider.SetModel(models[j%len(models)])
			}
			done <- true
		}(i)
	}

	for i := 0; i < numGoroutines*2; i++ {
		<-done
	}

	finalModel := provider.GetModel()
	assert.Contains(t, models, finalModel)
}

// TestOpenAIProvider_TokenHandlingFallback checks the fallback mechanism for token
// estimation when the API response omits usage data.
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

// TestOpenAIProvider_TimeoutConfiguration verifies that the provider's
// timeout settings are correctly applied.
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

// TestOpenAIProvider runs the standard provider test suite for the OpenAI provider.
func TestOpenAIProvider(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	config := ClientConfig{
		APIKey: apiKey,
		Model:  "gpt-3.5-turbo", // Use cheaper model for tests
	}

	suite := NewProviderTestSuite(t, "openai", config)

	t.Run("BasicRequest", func(t *testing.T) { suite.TestBasicRequest() })
	t.Run("RequestWithOptions", func(t *testing.T) { suite.TestRequestWithOptions() })
	t.Run("ErrorHandling", func(t *testing.T) { suite.TestErrorHandling() })
	t.Run("ContextCancellation", func(t *testing.T) { suite.TestContextCancellation() })
	t.Run("ModelGetterSetter", func(t *testing.T) { suite.TestModelGetterSetter() })
}
