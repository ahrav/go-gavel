package llm

import (
	"context"
	"fmt"
	"strings"

	"google.golang.org/genai"
)

// Google provider constants
const (
	// GoogleDefaultModel is the default Google model (Gemini 2.0 Flash)
	GoogleDefaultModel = "gemini-2.0-flash-exp"
)

func init() {
	RegisterProviderFactory("google", newGoogleProvider)
}

// googleProvider implements the CoreLLM interface for Google's Gemini API.
// This provider handles Google-specific authentication and request formatting
// while providing a consistent interface for the middleware system.
type googleProvider struct {
	client *genai.Client
	model  string
}

// newGoogleProvider creates a new Google Gemini provider instance.
// This factory function configures the provider for Google's Gemini API
// supporting both API key and service account authentication.
func newGoogleProvider(config ClientConfig) (CoreLLM, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("google API key or credentials path cannot be empty")
	}

	model := config.Model
	if model == "" {
		model = GoogleDefaultModel
	}

	ctx := context.Background()
	var client *genai.Client
	var err error

	// For simplicity, we'll primarily use API key authentication
	// In production, service account authentication would require additional
	// configuration like project ID and location for Vertex AI
	if isFilePath(config.APIKey) {
		// For file-based authentication, we'd need project and location
		// For now, we'll return an error suggesting API key usage
		return nil, fmt.Errorf("file-based authentication requires additional configuration (project, location). Please use API key authentication")
	} else {
		// API key authentication
		client, err = genai.NewClient(ctx, &genai.ClientConfig{
			APIKey:  config.APIKey,
			Backend: genai.BackendGeminiAPI,
		})
	}

	if err != nil {
		return nil, fmt.Errorf("failed to create Google client: %w", err)
	}

	return &googleProvider{
		client: client,
		model:  model,
	}, nil
}

// DoRequest sends a request to Google's Gemini API and returns the response.
// This method handles Google-specific request formatting, authentication,
// and response parsing while maintaining token usage tracking.
func (p *googleProvider) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
	// Build the request from prompt and options
	req := p.buildGenerateContentRequest(prompt, opts)

	// Build generation config from options
	config := p.buildGenerationConfig(opts)

	// Make API call
	resp, err := p.client.Models.GenerateContent(ctx, p.model, req, config)
	if err != nil {
		return "", 0, 0, p.handleGeminiError(err)
	}

	// Extract response text using the convenience method
	content := resp.Text()
	if content == "" {
		return "", 0, 0, fmt.Errorf("empty response from Gemini")
	}

	// Extract token usage or use estimation
	tokensIn := p.getTokenCount(resp.UsageMetadata, true, prompt)
	tokensOut := p.getTokenCount(resp.UsageMetadata, false, content)

	return content, tokensIn, tokensOut, nil
}

// getTokenCount returns the actual token count from API or falls back to estimation
func (p *googleProvider) getTokenCount(usage *genai.GenerateContentResponseUsageMetadata, isInput bool, text string) int {
	if usage != nil {
		if isInput && usage.PromptTokenCount > 0 {
			return int(usage.PromptTokenCount)
		}
		if !isInput && usage.CandidatesTokenCount > 0 {
			return int(usage.CandidatesTokenCount)
		}
	}
	return EstimateTokens(text)
}

// GetModel returns the currently configured Gemini model name.
func (p *googleProvider) GetModel() string { return p.model }

// SetModel updates the Gemini model for subsequent requests.
func (p *googleProvider) SetModel(m string) { p.model = m }

// buildGenerateContentRequest creates a GenerateContentRequest from prompt and options.
// This handles the mapping from the simple string prompt interface to Gemini's structured format.
func (p *googleProvider) buildGenerateContentRequest(prompt string, opts map[string]any) []*genai.Content {
	// Handle system prompt by prepending to user prompt (Gemini doesn't have separate system role)
	finalPrompt := prompt
	if systemPrompt := ExtractOptionalString(opts, "system_prompt", "", IsNonEmptyString); systemPrompt != "" {
		finalPrompt = systemPrompt + "\n\n" + prompt
	}

	// Create content with final prompt
	content := []*genai.Content{
		genai.NewContentFromText(finalPrompt, genai.RoleUser),
	}

	return content
}

// buildGenerationConfig creates a GenerateContentConfig from options.
// This handles the mapping of common parameters to Gemini's configuration format.
func (p *googleProvider) buildGenerationConfig(opts map[string]any) *genai.GenerateContentConfig {
	config := &genai.GenerateContentConfig{}

	// Temperature (0.0 to 2.0) - Gemini supports higher temperatures than Anthropic
	if temp := ExtractOptionalFloat64(opts, "temperature", -1, IsValidGeminiTemperature); temp != -1 {
		config.Temperature = genai.Ptr(float32(temp))
	}

	// MaxOutputTokens - only set if explicitly provided (Google provider doesn't use DefaultMaxTokens)
	if maxTokens := ExtractOptionalInt(opts, "max_tokens", 0, IsPositiveInt); maxTokens > 0 {
		config.MaxOutputTokens = int32(maxTokens) // #nosec G115 - maxTokens is validated positive and within reasonable bounds
	}

	// TopP (0.0 to 1.0)
	if topP := ExtractOptionalFloat64(opts, "top_p", -1, IsValidTopP); topP != -1 {
		config.TopP = genai.Ptr(float32(topP))
	}

	// TopK (1 to 40) - Gemini-specific parameter
	if topK := ExtractOptionalInt(opts, "top_k", -1, IsValidGeminiTopK); topK != -1 {
		config.TopK = genai.Ptr(float32(topK))
	}

	return config
}

// handleGeminiError converts Gemini-specific errors to appropriate error types.
// This provides better error handling and debugging information for the application.
func (p *googleProvider) handleGeminiError(err error) error {
	// Check for common error patterns in the error message
	errorMsg := err.Error()

	// Authentication errors
	if strings.Contains(errorMsg, "401") || strings.Contains(errorMsg, "Unauthorized") ||
		strings.Contains(errorMsg, "API key") || strings.Contains(errorMsg, "authentication") {
		return fmt.Errorf("gemini authentication failed: check API key or credentials: %w", err)
	}

	// Rate limiting errors
	if strings.Contains(errorMsg, "429") || strings.Contains(errorMsg, "quota") ||
		strings.Contains(errorMsg, "rate limit") {
		return fmt.Errorf("gemini rate limit exceeded: %w", err)
	}

	// Model not found errors
	if strings.Contains(errorMsg, "404") || strings.Contains(errorMsg, "model") &&
		strings.Contains(errorMsg, "not found") {
		return fmt.Errorf("gemini model '%s' not found or not accessible: %w", p.model, err)
	}

	// Server errors
	if strings.Contains(errorMsg, "500") || strings.Contains(errorMsg, "502") ||
		strings.Contains(errorMsg, "503") || strings.Contains(errorMsg, "504") {
		return fmt.Errorf("gemini server error: %w", err)
	}

	// Content policy violations
	if strings.Contains(errorMsg, "content") && (strings.Contains(errorMsg, "policy") ||
		strings.Contains(errorMsg, "safety") || strings.Contains(errorMsg, "blocked")) {
		return fmt.Errorf("gemini content policy violation: request blocked by safety filters: %w", err)
	}

	// Generic network/timeout errors
	if strings.Contains(errorMsg, "timeout") || strings.Contains(errorMsg, "connection") {
		return fmt.Errorf("gemini network error: %w", err)
	}

	// Default case: wrap the original error with context
	return fmt.Errorf("gemini API request failed: %w", err)
}

// isFilePath checks if the provided string looks like a file path.
// This helps determine whether to use API key or service account authentication.
func isFilePath(s string) bool {
	// Check if it contains path separators or file extensions
	return strings.Contains(s, "/") || strings.Contains(s, "\\") || strings.Contains(s, ".json")
}

// IsValidGeminiTopK returns true if the integer is a valid Gemini top_k value (1 to 40).
func IsValidGeminiTopK(val int) bool { return val >= 1 && val <= 40 }

// IsValidGeminiTemperature returns true if the float is a valid Gemini temperature (0.0 to 2.0).
func IsValidGeminiTemperature(val float64) bool { return val >= 0.0 && val <= 2.0 }
