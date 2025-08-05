// Package llm provides a unified interface for interacting with various Large
// Language Models (LLMs).
// It abstracts provider-specific details, offering a consistent API for
// features like text generation, request management, and error handling.
// This package also includes middleware support for implementing cross-cutting
// concerns such as caching, rate limiting, and metrics.
package llm

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"google.golang.org/api/googleapi"
	"google.golang.org/genai"
)

// Google provider constants define model names and other provider-specific
// values.
const (
	// GoogleDefaultModel is the default model for the Google provider.
	// It is currently set to Gemini 2.0 Flash.
	GoogleDefaultModel = "gemini-2.0-flash-exp"
)

func init() {
	RegisterProviderFactory("google", newGoogleProvider)
}

// googleProvider implements the CoreLLM interface for Google's Gemini API.
// It handles Google-specific authentication, request formatting, and error
// handling, while conforming to the common interface for middleware
// compatibility.
type googleProvider struct {
	BaseProvider
	client          *genai.Client
	tokenCounter    *TokenCounter
	errorClassifier *ErrorClassifier
}

// newGoogleProvider creates a new Google Gemini provider instance.
// This factory function configures the provider with the necessary client and
// authenticates using the provided configuration.
// It returns an error if the required configuration is missing or invalid.
func newGoogleProvider(config ClientConfig) (CoreLLM, error) {
	if config.APIKey == "" {
		return nil, ErrEmptyAPIKey
	}

	model := config.Model
	if model == "" {
		model = GoogleDefaultModel
	}

	ctx := context.Background()
	var client *genai.Client
	var err error

	// Configure authentication using the provided API key or service account.
	authConfig, err := buildAuthConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to configure authentication: %w", err)
	}

	client, err = genai.NewClient(ctx, authConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create Google client: %w", err)
	}

	return &googleProvider{
		BaseProvider:    BaseProvider{model: model},
		client:          client,
		tokenCounter:    NewTokenCounter(),
		errorClassifier: &ErrorClassifier{Provider: "google"},
	}, nil
}

// DoRequest sends a request to the Google Gemini API and returns the response.
// It formats the request, handles authentication, and parses the response,
// while also tracking token usage.
// This method returns the generated content, token counts, and any errors
// that occurred.
func (p *googleProvider) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
	options := ParseRequestOptions(opts, p.model)

	req := p.buildGenerateContentRequest(prompt, options)
	config := p.buildGenerationConfig(options)

	resp, err := p.client.Models.GenerateContent(ctx, options.Model, req, config)
	if err != nil {
		return "", 0, 0, p.handleError(err)
	}

	content := resp.Text()
	if content == "" {
		return "", 0, 0, ErrEmptyResponse
	}

	tokensIn := p.getTokenCount(resp.UsageMetadata, true, prompt)
	tokensOut := p.getTokenCount(resp.UsageMetadata, false, content)

	return content, tokensIn, tokensOut, nil
}

// getTokenCount retrieves the token count from the API response metadata.
// If the token count is not available in the metadata, it falls back to
// estimating the tokens based on the text content.
func (p *googleProvider) getTokenCount(usage *genai.GenerateContentResponseUsageMetadata, isInput bool, text string) int {
	if usage != nil {
		if isInput && usage.PromptTokenCount > 0 {
			return int(usage.PromptTokenCount)
		}
		if !isInput && usage.CandidatesTokenCount > 0 {
			return int(usage.CandidatesTokenCount)
		}
	}
	// Fallback to estimation if usage metadata is not available.
	return p.tokenCounter.EstimateTokens(text)
}

// buildGenerateContentRequest creates the content for a Google Gemini API
// request.
// It prepends the system prompt to the user prompt, as Google's API does not
// have a separate system role.
func (p *googleProvider) buildGenerateContentRequest(prompt string, options RequestOptions) []*genai.Content {
	finalPrompt := prompt
	if options.System != "" {
		// Prepend the system prompt to the user prompt in a structured format.
		finalPrompt = fmt.Sprintf("System: %s\n\nUser: %s", options.System, prompt)
	}

	return []*genai.Content{
		genai.NewContentFromText(finalPrompt, genai.RoleUser),
	}
}

// buildGenerationConfig creates the generation configuration for a Google
// Gemini API request.
// It validates and sets parameters such as temperature, max tokens, and top P.
func (p *googleProvider) buildGenerationConfig(options RequestOptions) *genai.GenerateContentConfig {
	config := &genai.GenerateContentConfig{}

	if options.Temperature != nil {
		// Clamp temperature to the supported range of 0.0 to 2.0 for Gemini.
		temp := clamp(*options.Temperature, 0.0, 2.0)
		config.Temperature = genai.Ptr(float32(temp))
	}

	if options.MaxTokens > 0 {
		// Safely convert max tokens to int32, respecting the maximum value.
		if options.MaxTokens > math.MaxInt32 {
			config.MaxOutputTokens = math.MaxInt32
		} else {
			config.MaxOutputTokens = int32(options.MaxTokens)
		}
	}

	if options.TopP != nil {
		topP := clamp(*options.TopP, 0.0, 1.0)
		config.TopP = genai.Ptr(float32(topP))
	}

	if topK, ok := options.Extra["top_k"].(int); ok {
		// Clamp top K to the Gemini-specific supported range of 1 to 40.
		topK = clampInt(topK, 1, 40)
		config.TopK = genai.Ptr(float32(topK))
	}

	return config
}

// handleError provides structured error handling for Google API responses.
// It classifies errors based on their type, such as context errors or API
// errors, and returns a standardized ProviderError.
func (p *googleProvider) handleError(err error) error {
	if isContextError(err) {
		return p.errorClassifier.ClassifyContextError(err)
	}

	if apiErr, ok := err.(*googleapi.Error); ok {
		message := apiErr.Message
		if message == "" && len(apiErr.Errors) > 0 {
			message = apiErr.Errors[0].Message
		}

		// Provide special handling for content policy violations to return a
		// clear error.
		if containsContentPolicyError(apiErr) {
			return NewProviderError("google", ErrorTypeContentPolicy, apiErr.Code,
				"request blocked by safety filters", err)
		}

		return p.errorClassifier.ClassifyHTTPError(apiErr.Code, message, err)
	}

	return NewProviderError("google", ErrorTypeUnknown, 0, "request failed", err)
}

// buildAuthConfig creates the appropriate authentication configuration based on
// the client settings.
// It supports both API key and service account authentication.
func buildAuthConfig(config ClientConfig) (*genai.ClientConfig, error) {
	if looksLikeFilePath(config.APIKey) {
		// Ensure the credentials file exists before proceeding.
		if !fileExists(config.APIKey) {
			return nil, fmt.Errorf("credentials file not found: %s", config.APIKey)
		}

		// For production environments, service account authentication should be
		// fully implemented.
		// This implementation provides a clear error message for guidance.
		return nil, fmt.Errorf("service account authentication requires additional configuration. " +
			"Please use API key authentication or set GOOGLE_APPLICATION_CREDENTIALS environment variable")
	}

	return &genai.ClientConfig{
		APIKey:  config.APIKey,
		Backend: genai.BackendGeminiAPI,
	}, nil
}

// looksLikeFilePath checks if a string appears to be a file path.
// It performs checks for absolute paths, relative paths, and common credential
// file extensions.
func looksLikeFilePath(s string) bool {
	if filepath.IsAbs(s) {
		return true
	}

	if strings.Contains(s, "/") || strings.Contains(s, "\\") {
		return true
	}

	lower := strings.ToLower(s)
	if strings.HasSuffix(lower, ".json") ||
		strings.HasSuffix(lower, ".p12") ||
		strings.HasSuffix(lower, ".pem") ||
		strings.Contains(lower, "credentials") {
		return true
	}

	return false
}

// fileExists checks if a file exists at the given path.
func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// isContextError checks if an error is a context-related error, such as a
// deadline exceeded or cancellation.
func isContextError(err error) bool {
	return errors.Is(err, context.DeadlineExceeded) ||
		errors.Is(err, context.Canceled)
}

// containsContentPolicyError checks if a Google API error is related to
// content policy violations.
func containsContentPolicyError(apiErr *googleapi.Error) bool {
	if apiErr.Message != "" {
		lower := strings.ToLower(apiErr.Message)
		if strings.Contains(lower, "safety") ||
			strings.Contains(lower, "policy") ||
			strings.Contains(lower, "blocked") {
			return true
		}
	}

	for _, e := range apiErr.Errors {
		if e.Reason == "SAFETY" || e.Reason == "BLOCKED" {
			return true
		}
	}

	return false
}

// clamp restricts a float64 value to a specified range.
func clamp(val, min, max float64) float64 {
	if val < min {
		return min
	}
	if val > max {
		return max
	}
	return val
}

// clampInt restricts an integer value to a specified range.
func clampInt(val, min, max int) int {
	if val < min {
		return min
	}
	if val > max {
		return max
	}
	return val
}
