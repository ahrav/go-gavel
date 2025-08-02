package testutils

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ahrav/go-gavel/internal/ports"
)

func TestMockLLMClient_Complete(t *testing.T) {
	tests := []struct {
		name           string
		prompt         string
		options        map[string]any
		expectedResult string
		expectError    bool
	}{
		{
			name:           "matches generate pattern",
			prompt:         "Please generate a comprehensive response",
			options:        nil,
			expectedResult: "This is a comprehensive answer that addresses the key aspects of the question with detailed analysis and supporting evidence.",
		},
		{
			name:           "matches answer pattern",
			prompt:         "Please answer this question thoroughly",
			options:        nil,
			expectedResult: "A concise and focused response that directly answers the question with clear reasoning.",
		},
		{
			name:           "matches judge pattern",
			prompt:         "Judge the quality of this response",
			options:        nil,
			expectedResult: `{"score": 0.91, "confidence": 0.95, "reasoning": "Excellent answer with thorough analysis and well-supported conclusions.", "version": 1}`,
		},
		{
			name:           "falls back to default for unmatched pattern",
			prompt:         "Random prompt that doesn't match any pattern",
			options:        nil,
			expectedResult: "This is a standard response for testing purposes with moderate length and complexity.",
		},
		{
			name:        "fails with empty prompt",
			prompt:      "",
			options:     nil,
			expectError: true,
		},
		{
			name:   "adds variation with high temperature",
			prompt: "Generate a detailed response to this complex question",
			options: map[string]any{
				"temperature": 0.8,
			},
			expectedResult: "This is a comprehensive answer that addresses the key aspects of the question with detailed analysis and supporting evidence. Further details enhance the overall quality.",
		},
		{
			name:   "no variation with low temperature",
			prompt: "Generate a response",
			options: map[string]any{
				"temperature": 0.2,
			},
			expectedResult: "This is a comprehensive answer that addresses the key aspects of the question with detailed analysis and supporting evidence.",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewMockLLMClient("test-model")
			ctx := context.Background()

			result, err := client.Complete(ctx, tt.prompt, tt.options)

			if tt.expectError {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.expectedResult, result)
			}
		})
	}
}

func TestMockLLMClient_EstimateTokens(t *testing.T) {
	tests := []struct {
		name           string
		text           string
		expectedTokens int
	}{
		{
			name:           "empty text returns zero",
			text:           "",
			expectedTokens: 0,
		},
		{
			name:           "short text returns minimum one token",
			text:           "Hi",
			expectedTokens: 1,
		},
		{
			name:           "medium text returns reasonable estimate",
			text:           "This is a test sentence.",
			expectedTokens: 6, // 25 characters / 4 = 6.25, rounded down to 6
		},
		{
			name:           "long text returns proportional estimate",
			text:           "This is a much longer sentence that contains multiple words and should result in a higher token count estimate based on the character-to-token ratio.",
			expectedTokens: 37, // 149 characters / 4 = 37.25, rounded down to 37
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewMockLLMClient("test-model")

			tokens, err := client.EstimateTokens(tt.text)

			require.NoError(t, err)
			assert.Equal(t, tt.expectedTokens, tokens)
		})
	}
}

func TestMockLLMClient_GetModel(t *testing.T) {
	client := NewMockLLMClient("custom-model-v2")
	assert.Equal(t, "custom-model-v2", client.GetModel())
}

func TestMockLLMClient_SetModel(t *testing.T) {
	client := NewMockLLMClient("initial-model")
	assert.Equal(t, "initial-model", client.GetModel())

	client.SetModel("updated-model")
	assert.Equal(t, "updated-model", client.GetModel())
}

func TestMockLLMClient_AddResponse(t *testing.T) {
	client := NewMockLLMClient("test-model")

	// Add custom response.
	customResponse := MockResponse{
		Pattern:    "custom",
		Response:   "This is a custom response for testing",
		TokensUsed: 10,
	}
	client.AddResponse(customResponse)

	ctx := context.Background()
	result, err := client.Complete(ctx, "This is a custom request", nil)

	require.NoError(t, err)
	assert.Equal(t, "This is a custom response for testing", result)

	// Verify token usage.
	tokens := client.GetTokenUsage("custom")
	assert.Equal(t, 10, tokens)
}

func TestMockLLMClient_Reset(t *testing.T) {
	client := NewMockLLMClient("test-model")

	// Add custom response.
	customResponse := MockResponse{
		Pattern:    "custom",
		Response:   "Custom response",
		TokensUsed: 5,
	}
	client.AddResponse(customResponse)

	// Verify custom response works.
	ctx := context.Background()
	result, err := client.Complete(ctx, "custom prompt", nil)
	require.NoError(t, err)
	assert.Equal(t, "Custom response", result)

	// Reset and verify default behavior is restored.
	client.Reset()
	result, err = client.Complete(ctx, "custom prompt", nil)
	require.NoError(t, err)
	assert.NotEqual(t, "Custom response", result)
	assert.Contains(t, result, "standard response") // Should match default pattern
}

func TestMockLLMClient_GetTokenUsage(t *testing.T) {
	client := NewMockLLMClient("test-model")

	// Test default token usage.
	generateTokens := client.GetTokenUsage("generate")
	assert.Equal(t, 25, generateTokens)

	scoreTokens := client.GetTokenUsage("score")
	assert.Equal(t, 22, scoreTokens)

	// Test unknown pattern returns default.
	unknownTokens := client.GetTokenUsage("unknown")
	defaultTokens := client.GetTokenUsage("")
	assert.Equal(t, defaultTokens, unknownTokens)
}

func TestMockLLMClient_ContextCancellation(t *testing.T) {
	client := NewMockLLMClient("test-model")

	// Test with cancelled context.
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err := client.Complete(ctx, "test prompt", nil)
	require.Error(t, err)
	assert.Equal(t, context.Canceled, err)
}

func TestMockLLMClient_VariationLogic(t *testing.T) {
	client := NewMockLLMClient("test-model")
	ctx := context.Background()

	tests := []struct {
		name           string
		prompt         string
		expectedSuffix string
	}{
		{
			name:           "very long prompt adds comprehensive suffix",
			prompt:         "This is a very long prompt that contains multiple sentences and a lot of detailed information that should trigger the comprehensive analysis variation logic based on the length of the input text. " + "Additional text to make it even longer and ensure we hit the 200+ character threshold for the most comprehensive variation response.",
			expectedSuffix: "Additionally, this comprehensive analysis covers multiple dimensions of the topic.",
		},
		{
			name:           "medium prompt adds context suffix",
			prompt:         "This is a medium-length prompt that should trigger the moderate variation response logic.",
			expectedSuffix: "Further details enhance the overall quality.",
		},
		{
			name:           "short prompt adds details suffix",
			prompt:         "This is a shorter prompt that should get details.",
			expectedSuffix: "", // 50 chars or less, no variation
		},
		{
			name:           "very short prompt gets no suffix variation",
			prompt:         "Short",
			expectedSuffix: "", // No variation for very short prompts
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			options := map[string]any{"temperature": 0.8} // High temperature to trigger variation

			result, err := client.Complete(ctx, tt.prompt, options)
			require.NoError(t, err)

			if tt.expectedSuffix != "" {
				assert.Contains(t, result, tt.expectedSuffix,
					"Response should contain expected variation suffix")
			} else {
				// For very short prompts, should not contain any of the variation suffixes
				assert.NotContains(t, result, "Additionally, this comprehensive")
				assert.NotContains(t, result, "This response includes additional")
				assert.NotContains(t, result, "Further details enhance")
			}
		})
	}
}

func TestMockLLMClient_InterfaceCompliance(t *testing.T) {
	// Verify that MockLLMClient implements the LLMClient interface.
	var client ports.LLMClient = NewMockLLMClient("test-model")

	// Test all interface methods are available.
	ctx := context.Background()

	// Test Complete method.
	response, err := client.Complete(ctx, "test", nil)
	require.NoError(t, err)
	assert.NotEmpty(t, response)

	// Test EstimateTokens method.
	tokens, err := client.EstimateTokens("test text")
	require.NoError(t, err)
	assert.Greater(t, tokens, 0)

	// Test GetModel method.
	model := client.GetModel()
	assert.Equal(t, "test-model", model)
}
