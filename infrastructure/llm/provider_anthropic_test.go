package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockUsage provides a mock structure for token usage information in test responses.
type mockUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// mockContent provides a mock structure for content blocks in test responses.
type mockContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// mockResponse provides a mock structure for a successful API response in tests.
type mockResponse struct {
	ID      string        `json:"id"`
	Type    string        `json:"type"`
	Role    string        `json:"role"`
	Content []mockContent `json:"content"`
	Model   string        `json:"model"`
	Usage   mockUsage     `json:"usage"`
}

// mockErrorResponse provides a mock structure for an error response in tests.
type mockErrorResponse struct {
	Type  string    `json:"type"`
	Error mockError `json:"error"`
}

// mockError provides a mock structure for error details in test responses.
type mockError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// TestNewAnthropicProvider tests the creation of a new Anthropic provider.
// It covers various scenarios, including valid and invalid configurations.
func TestNewAnthropicProvider(t *testing.T) {
	tests := []struct {
		name        string
		config      ClientConfig
		expectError bool
		errorMsg    string
	}{
		{
			name: "valid config with all fields",
			config: ClientConfig{
				APIKey:  "test-api-key",
				Model:   AnthropicDefaultModel,
				BaseURL: "https://api.anthropic.com",
			},
			expectError: false,
		},
		{
			name: "valid config with minimal fields",
			config: ClientConfig{
				APIKey: "test-api-key",
			},
			expectError: false,
		},
		{
			name: "empty API key",
			config: ClientConfig{
				APIKey: "",
			},
			expectError: true,
			errorMsg:    "anthropic API key cannot be empty",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, err := newAnthropicProvider(tt.config)

			if tt.expectError {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errorMsg)
				assert.Nil(t, provider)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, provider)

				anthProvider := provider.(*anthropicProvider)
				assert.NotNil(t, anthProvider.client)

				expectedModel := tt.config.Model
				if expectedModel == "" {
					expectedModel = AnthropicDefaultModel
				}
				assert.Equal(t, expectedModel, anthProvider.model)
			}
		})
	}
}

// TestAnthropicProvider_GetSetModel tests the GetModel and SetModel methods
// of the Anthropic provider.
func TestAnthropicProvider_GetSetModel(t *testing.T) {
	provider, err := newAnthropicProvider(ClientConfig{APIKey: "test-key"})
	require.NoError(t, err)

	assert.Equal(t, AnthropicDefaultModel, provider.GetModel())

	provider.SetModel("claude-3-opus-20240229")
	assert.Equal(t, "claude-3-opus-20240229", provider.GetModel())
}

// TestAnthropicProvider_DoRequest_Success tests a successful request to the
// Anthropic provider.
func TestAnthropicProvider_DoRequest_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
		assert.Contains(t, r.Header.Get("User-Agent"), "Anthropic")

		var reqBody map[string]interface{}
		err := json.NewDecoder(r.Body).Decode(&reqBody)
		require.NoError(t, err)

		assert.Equal(t, AnthropicDefaultModel, reqBody["model"])
		assert.Equal(t, float64(DefaultMaxTokens), reqBody["max_tokens"])

		messages := reqBody["messages"].([]interface{})
		assert.Len(t, messages, 1)

		response := mockResponse{
			ID:   "msg_test_id",
			Type: "message",
			Role: "assistant",
			Content: []mockContent{
				{Type: "text", Text: "Hello! This is a test response."},
			},
			Model: AnthropicDefaultModel,
			Usage: mockUsage{
				InputTokens:  10,
				OutputTokens: 15,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider, err := newAnthropicProvider(ClientConfig{
		APIKey:  "test-api-key",
		BaseURL: server.URL,
	})
	require.NoError(t, err)

	ctx := context.Background()
	response, tokensIn, tokensOut, err := provider.DoRequest(ctx, "Hello, world!", map[string]any{})

	require.NoError(t, err)
	assert.Equal(t, "Hello! This is a test response.", response)
	assert.Equal(t, 10, tokensIn)
	assert.Equal(t, 15, tokensOut)
}

// TestAnthropicProvider_DoRequest_WithOptions tests a request to the Anthropic
// provider with custom options.
func TestAnthropicProvider_DoRequest_WithOptions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody map[string]interface{}
		err := json.NewDecoder(r.Body).Decode(&reqBody)
		require.NoError(t, err)

		assert.Equal(t, "claude-3-opus-20240229", reqBody["model"])
		assert.Equal(t, float64(2048), reqBody["max_tokens"])
		assert.Equal(t, 0.7, reqBody["temperature"])

		system := reqBody["system"].([]interface{})
		assert.Len(t, system, 1)
		systemMsg := system[0].(map[string]interface{})
		assert.Equal(t, "You are a helpful assistant.", systemMsg["text"])

		response := mockResponse{
			ID:   "msg_test_id",
			Type: "message",
			Role: "assistant",
			Content: []mockContent{
				{Type: "text", Text: "Custom response with options."},
			},
			Model: "claude-3-opus-20240229",
			Usage: mockUsage{
				InputTokens:  20,
				OutputTokens: 25,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider, err := newAnthropicProvider(ClientConfig{
		APIKey:  "test-api-key",
		BaseURL: server.URL,
	})
	require.NoError(t, err)

	opts := map[string]any{
		"model":       "claude-3-opus-20240229",
		"max_tokens":  2048,
		"temperature": 0.7,
		"system":      "You are a helpful assistant.",
	}

	ctx := context.Background()
	response, tokensIn, tokensOut, err := provider.DoRequest(ctx, "Test prompt", opts)

	require.NoError(t, err)
	assert.Equal(t, "Custom response with options.", response)
	assert.Equal(t, 20, tokensIn)
	assert.Equal(t, 25, tokensOut)
}

// TestAnthropicProvider_DoRequest_MultipleContentBlocks tests a response from
// the Anthropic provider that contains multiple content blocks.
func TestAnthropicProvider_DoRequest_MultipleContentBlocks(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := mockResponse{
			ID:   "msg_test_id",
			Type: "message",
			Role: "assistant",
			Content: []mockContent{
				{Type: "text", Text: "First part of response. "},
				{Type: "text", Text: "Second part of response."},
			},
			Model: AnthropicDefaultModel,
			Usage: mockUsage{
				InputTokens:  10,
				OutputTokens: 20,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider, err := newAnthropicProvider(ClientConfig{
		APIKey:  "test-api-key",
		BaseURL: server.URL,
	})
	require.NoError(t, err)

	ctx := context.Background()
	response, tokensIn, tokensOut, err := provider.DoRequest(ctx, "Test", map[string]any{})

	require.NoError(t, err)
	assert.Equal(t, "First part of response. Second part of response.", response)
	assert.Equal(t, 10, tokensIn)
	assert.Equal(t, 20, tokensOut)
}

// TestAnthropicProvider_DoRequest_AuthError tests the handling of an
// authentication error from the Anthropic provider.
func TestAnthropicProvider_DoRequest_AuthError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		errorResp := mockErrorResponse{
			Type: "error",
			Error: mockError{
				Type:    "authentication_error",
				Message: "invalid api key",
			},
		}
		json.NewEncoder(w).Encode(errorResp)
	}))
	defer server.Close()

	provider, err := newAnthropicProvider(ClientConfig{
		APIKey:  "invalid-key",
		BaseURL: server.URL,
	})
	require.NoError(t, err)

	ctx := context.Background()
	response, tokensIn, tokensOut, err := provider.DoRequest(ctx, "Test", map[string]any{})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "anthropic authentication failed")
	assert.Contains(t, err.Error(), "401")
	assert.Empty(t, response)
	assert.Equal(t, 0, tokensIn)
	assert.Equal(t, 0, tokensOut)
}

// TestAnthropicProvider_DoRequest_RateLimitError tests the handling of a
// rate limit error from the Anthropic provider.
func TestAnthropicProvider_DoRequest_RateLimitError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		errorResp := mockErrorResponse{
			Type: "error",
			Error: mockError{
				Type:    "rate_limit_error",
				Message: "rate limit exceeded",
			},
		}
		json.NewEncoder(w).Encode(errorResp)
	}))
	defer server.Close()

	provider, err := newAnthropicProvider(ClientConfig{
		APIKey:  "test-api-key",
		BaseURL: server.URL,
	})
	require.NoError(t, err)

	ctx := context.Background()
	response, tokensIn, tokensOut, err := provider.DoRequest(ctx, "Test", map[string]any{})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "anthropic rate limit exceeded")
	assert.Contains(t, err.Error(), "429")
	assert.Empty(t, response)
	assert.Equal(t, 0, tokensIn)
	assert.Equal(t, 0, tokensOut)
}

// TestAnthropicProvider_DoRequest_ContextCancellation tests the handling of
// context cancellation during a request to the Anthropic provider.
func TestAnthropicProvider_DoRequest_ContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond)

		response := mockResponse{
			ID:   "msg_test_id",
			Type: "message",
			Role: "assistant",
			Content: []mockContent{
				{Type: "text", Text: "Response"},
			},
			Model: AnthropicDefaultModel,
			Usage: mockUsage{InputTokens: 5, OutputTokens: 5},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider, err := newAnthropicProvider(ClientConfig{
		APIKey:  "test-api-key",
		BaseURL: server.URL,
	})
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	response, tokensIn, tokensOut, err := provider.DoRequest(ctx, "Test", map[string]any{})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "context deadline exceeded")
	assert.Empty(t, response)
	assert.Equal(t, 0, tokensIn)
	assert.Equal(t, 0, tokensOut)
}

// TestAnthropicProvider_DoRequest_TokenFallback tests the token estimation
// fallback mechanism when the API response does not include usage information.
func TestAnthropicProvider_DoRequest_TokenFallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := mockResponse{
			ID:   "msg_test_id",
			Type: "message",
			Role: "assistant",
			Content: []mockContent{
				{Type: "text", Text: "Test response"},
			},
			Model: AnthropicDefaultModel,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider, err := newAnthropicProvider(ClientConfig{
		APIKey:  "test-api-key",
		BaseURL: server.URL,
	})
	require.NoError(t, err)

	ctx := context.Background()
	response, tokensIn, tokensOut, err := provider.DoRequest(ctx, "Hello world", map[string]any{})

	require.NoError(t, err)
	assert.Equal(t, "Test response", response)
	assert.Greater(t, tokensIn, 0)
	assert.Greater(t, tokensOut, 0)
}

// TestAnthropicProvider_DoRequest_InvalidOptions tests that the provider
// handles invalid options gracefully by falling back to default values.
func TestAnthropicProvider_DoRequest_InvalidOptions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody map[string]interface{}
		json.NewDecoder(r.Body).Decode(&reqBody)

		assert.Equal(t, AnthropicDefaultModel, reqBody["model"])
		assert.Equal(t, float64(DefaultMaxTokens), reqBody["max_tokens"])

		_, hasTemp := reqBody["temperature"]
		assert.False(t, hasTemp)

		response := mockResponse{
			ID:      "msg_test_id",
			Type:    "message",
			Role:    "assistant",
			Content: []mockContent{{Type: "text", Text: "Response"}},
			Model:   AnthropicDefaultModel,
			Usage:   mockUsage{InputTokens: 5, OutputTokens: 5},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider, err := newAnthropicProvider(ClientConfig{
		APIKey:  "test-api-key",
		BaseURL: server.URL,
	})
	require.NoError(t, err)

	opts := map[string]any{
		"model":       "",
		"max_tokens":  -1,
		"temperature": 2.0,
		"system":      "",
	}

	ctx := context.Background()
	response, tokensIn, tokensOut, err := provider.DoRequest(ctx, "Test", opts)

	require.NoError(t, err)
	assert.Equal(t, "Response", response)
	assert.Equal(t, 5, tokensIn)
	assert.Equal(t, 5, tokensOut)
}
