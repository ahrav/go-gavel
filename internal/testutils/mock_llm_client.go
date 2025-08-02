package testutils

import (
	"context"
	"fmt"
	"strings"

	"github.com/ahrav/go-gavel/internal/ports"
)

// MockLLMClient implements the LLMClient interface with deterministic responses
// for consistent testing and development workflows.
// It provides pre-defined responses based on prompt patterns and includes
// realistic token count metadata for budget tracking integration.
type MockLLMClient struct {
	// model is the mock model identifier.
	model string
	// responses maps prompt patterns to pre-defined responses.
	responses map[string]string
	// tokenCounts maps response types to token usage estimates.
	tokenCounts map[string]int
}

// MockResponse defines a pre-configured response pattern for the mock client.
type MockResponse struct {
	// Pattern is used to match against prompts (substring matching).
	Pattern string
	// Response is the text returned for matching prompts.
	Response string
	// TokensUsed is the estimated token count for this response.
	TokensUsed int
}

// NewMockLLMClient creates a new MockLLMClient with pre-configured responses
// for different unit types and scenarios.
// The mock provides deterministic responses to enable reliable testing
// of the evaluation pipeline.
func NewMockLLMClient(model string) *MockLLMClient {
	client := &MockLLMClient{
		model:       model,
		responses:   make(map[string]string),
		tokenCounts: make(map[string]int),
	}

	// Configure default responses for different unit types.
	client.setupDefaultResponses()

	return client
}

// setupDefaultResponses configures standard responses for common evaluation scenarios.
// These responses are designed to provide realistic variety while maintaining
// deterministic behavior for testing.
func (m *MockLLMClient) setupDefaultResponses() {
	// AnswererUnit responses - different answers for variety.
	m.AddResponse(MockResponse{
		Pattern:    "generate",
		Response:   "This is a comprehensive answer that addresses the key aspects of the question with detailed analysis and supporting evidence.",
		TokensUsed: 25,
	})

	m.AddResponse(MockResponse{
		Pattern:    "answer",
		Response:   "A concise and focused response that directly answers the question with clear reasoning.",
		TokensUsed: 18,
	})

	m.AddResponse(MockResponse{
		Pattern:    "provide",
		Response:   "An alternative perspective that considers multiple viewpoints and provides balanced analysis.",
		TokensUsed: 20,
	})

	// ScoreJudgeUnit responses - scoring and evaluation patterns with JSON format.
	m.AddResponse(MockResponse{
		Pattern:    "score",
		Response:   `{"score": 0.85, "confidence": 0.9, "reasoning": "This answer demonstrates strong understanding and provides clear reasoning.", "version": 1}`,
		TokensUsed: 22,
	})

	m.AddResponse(MockResponse{
		Pattern:    "evaluate",
		Response:   `{"score": 0.72, "confidence": 0.8, "reasoning": "The response shows good comprehension but could benefit from more detail.", "version": 1}`,
		TokensUsed: 20,
	})

	m.AddResponse(MockResponse{
		Pattern:    "judge",
		Response:   `{"score": 0.91, "confidence": 0.95, "reasoning": "Excellent answer with thorough analysis and well-supported conclusions.", "version": 1}`,
		TokensUsed: 24,
	})

	m.AddResponse(MockResponse{
		Pattern:    "rate",
		Response:   `{"score": 0.88, "confidence": 0.92, "reasoning": "This answer provides clear and accurate information with good supporting details.", "version": 1}`,
		TokensUsed: 23,
	})

	// Default response for unmatched patterns.
	m.AddResponse(MockResponse{
		Pattern:    "",
		Response:   "This is a standard response for testing purposes with moderate length and complexity.",
		TokensUsed: 15,
	})
}

// AddResponse adds a new response pattern to the mock client.
// This allows customization of responses for specific testing scenarios.
func (m *MockLLMClient) AddResponse(response MockResponse) {
	m.responses[response.Pattern] = response.Response
	m.tokenCounts[response.Pattern] = response.TokensUsed
}

// Complete implements the LLMClient.Complete method with deterministic responses
// based on prompt pattern matching.
// It selects appropriate responses based on prompt content and returns
// consistent results for identical inputs.
func (m *MockLLMClient) Complete(ctx context.Context, prompt string, options map[string]any) (string, error) {
	if ctx.Err() != nil {
		return "", ctx.Err()
	}

	if prompt == "" {
		return "", fmt.Errorf("prompt cannot be empty")
	}

	// Find the best matching response pattern.
	response := m.findMatchingResponse(prompt)

	// Apply temperature variation if specified in options.
	if temp, ok := options["temperature"].(float64); ok && temp > 0.5 {
		// For higher temperatures, add slight variation (deterministic).
		response = m.addVariation(response, prompt)
	}

	return response, nil
}

// EstimateTokens implements the LLMClient.EstimateTokens method using
// a simple estimation algorithm based on text length.
// This provides realistic token estimates for budget tracking and validation.
func (m *MockLLMClient) EstimateTokens(text string) (int, error) {
	if text == "" {
		return 0, nil
	}

	// Simple token estimation: approximately 4 characters per token.
	// This is a reasonable approximation for English text with GPT-style tokenization.
	tokens := len(text) / 4
	if tokens == 0 {
		tokens = 1 // Minimum one token for non-empty text.
	}

	return tokens, nil
}

// GetModel implements the LLMClient.GetModel method returning the mock model identifier.
// This is used for logging, debugging, and configuration validation.
func (m *MockLLMClient) GetModel() string {
	return m.model
}

// findMatchingResponse selects the most appropriate response based on prompt content.
// It uses substring matching to identify the best response pattern.
func (m *MockLLMClient) findMatchingResponse(prompt string) string {
	promptLower := strings.ToLower(prompt)

	// Define patterns with priority and whether they need word boundaries
	// This prevents "generate" from matching "rate"
	type patternInfo struct {
		pattern           string
		needsWordBoundary bool
	}

	patterns := []patternInfo{
		// Scoring patterns need JSON responses - check these first
		{"rate", true}, // Use word boundary to avoid "generate" matching "rate"
		{"score", false},
		{"judge", false},
		{"evaluate", false},
		// Answer generation patterns
		{"generate", false},
		{"answer", false},
		{"provide", false},
	}

	// Check patterns in order
	for _, p := range patterns {
		if response, ok := m.responses[p.pattern]; ok {
			if p.needsWordBoundary {
				// Check for word boundaries to avoid partial matches
				if strings.Contains(promptLower, " "+p.pattern+" ") ||
					strings.Contains(promptLower, " "+p.pattern+":") ||
					strings.Contains(promptLower, " "+p.pattern+".") ||
					strings.HasPrefix(promptLower, p.pattern+" ") ||
					strings.HasSuffix(promptLower, " "+p.pattern) {
					return response
				}
			} else if strings.Contains(promptLower, p.pattern) {
				return response
			}
		}
	}

	// Check any other custom patterns that might have been added
	knownPatterns := []string{"rate", "score", "judge", "evaluate", "generate", "answer", "provide"}
	for pattern, response := range m.responses {
		if pattern != "" && !contains(knownPatterns, pattern) && strings.Contains(promptLower, pattern) {
			return response
		}
	}

	// Fall back to default response if no pattern matches.
	if defaultResponse, ok := m.responses[""]; ok {
		return defaultResponse
	}

	// Ultimate fallback if no default is configured.
	return "Mock response for testing purposes."
}

// contains checks if a string slice contains a specific string
func contains(slice []string, str string) bool {
	for _, s := range slice {
		if s == str {
			return true
		}
	}
	return false
}

// addVariation adds deterministic variation to responses based on prompt characteristics.
// This simulates the natural variation in LLM responses while maintaining determinism.
func (m *MockLLMClient) addVariation(response string, prompt string) string {
	// Add variation based on prompt length (deterministic).
	promptLength := len(prompt)

	switch {
	case promptLength > 200:
		return response + " Additionally, this comprehensive analysis covers multiple dimensions of the topic."
	case promptLength > 100:
		return response + " This response includes additional context for completeness."
	case promptLength > 50:
		return response + " Further details enhance the overall quality."
	default:
		return response
	}
}

// GetTokenUsage returns the estimated token usage for a given response pattern.
// This supports budget tracking and performance analysis in tests.
func (m *MockLLMClient) GetTokenUsage(pattern string) int {
	if tokens, ok := m.tokenCounts[pattern]; ok {
		return tokens
	}
	return m.tokenCounts[""] // Default token count.
}

// Reset clears all custom responses and restores default configuration.
// This is useful for test cleanup and ensuring consistent starting state.
func (m *MockLLMClient) Reset() {
	m.responses = make(map[string]string)
	m.tokenCounts = make(map[string]int)
	m.setupDefaultResponses()
}

// SetModel updates the mock model identifier.
// This allows testing with different model configurations.
func (m *MockLLMClient) SetModel(model string) {
	m.model = model
}

// Verify interface compliance at compile time.
var _ ports.LLMClient = (*MockLLMClient)(nil)
