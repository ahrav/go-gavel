package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"sync/atomic"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

const (
	// OpenAI API parameter validation constants
	minTemperature = 0.0
	maxTemperature = 2.0
	minTopP        = 0.0
	maxTopP        = 1.0
	minPenalty     = -2.0
	maxPenalty     = 2.0
	minMaxTokens   = 1

	// Configuration validation constants
	minTimeout = 1 * time.Second  // Minimum reasonable timeout
	maxTimeout = 10 * time.Minute // Maximum reasonable timeout

	// Float32 precision and range constants
	maxSafeFloat32Integer = 16777216 // 2^24, beyond this integers lose precision in float32
	minSafeFloat32Integer = -16777216
	maxFloat32            = 3.4028235e+38
	minFloat32            = -3.4028235e+38
	minFloat32Subnormal   = 1.175494e-38
)

func init() {
	RegisterProviderFactory("openai", newOpenAIProvider)
}

// openAIProvider implements the CoreLLM interface for OpenAI's API.
// This provider handles OpenAI-specific request formatting and response parsing
// while conforming to the common interface for middleware compatibility.
type openAIProvider struct {
	client *openai.Client
	model  atomic.Value // Thread-safe storage for model string
}

// newOpenAIProvider creates a new OpenAI provider instance.
// This factory function initializes the provider with configuration
// and validates required settings like API key presence.
func newOpenAIProvider(config ClientConfig) (CoreLLM, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("OpenAI API key cannot be empty")
	}

	model := config.Model
	if model == "" {
		model = "gpt-3.5-turbo" // Default to cost-effective model
	}

	clientConfig := openai.DefaultConfig(config.APIKey)

	if config.BaseURL != "" {
		if err := validateBaseURL(config.BaseURL); err != nil {
			return nil, fmt.Errorf("invalid BaseURL: %w", err)
		}
		clientConfig.BaseURL = config.BaseURL
	}

	if config.Timeout > 0 {
		validatedTimeout := validateTimeout(config.Timeout)
		clientConfig.HTTPClient = &http.Client{
			Timeout: validatedTimeout,
		}
	}

	client := openai.NewClientWithConfig(clientConfig)

	provider := &openAIProvider{
		client: client,
	}
	provider.model.Store(model)

	return provider, nil
}

// DoRequest sends a request to OpenAI's API and returns the response.
// This method handles OpenAI-specific request formatting, authentication,
// and response parsing while providing token usage information.
func (p *openAIProvider) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
	req := p.buildChatCompletionRequest(prompt, opts)
	resp, err := p.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", 0, 0, p.handleOpenAIError(err)
	}

	if len(resp.Choices) == 0 {
		return "", 0, 0, fmt.Errorf("no response choices returned from OpenAI")
	}

	content := resp.Choices[0].Message.Content

	var tokensIn, tokensOut int
	if resp.Usage.PromptTokens > 0 && resp.Usage.CompletionTokens > 0 {
		tokensIn = resp.Usage.PromptTokens
		tokensOut = resp.Usage.CompletionTokens
	} else {
		// Fallback to estimation if OpenAI doesn't provide usage data
		tokensIn = p.estimateTokens(prompt)
		tokensOut = p.estimateTokens(content)
	}

	return content, tokensIn, tokensOut, nil
}

// GetModel returns the currently configured OpenAI model name.
// This method is thread-safe using atomic operations.
func (p *openAIProvider) GetModel() string { return p.model.Load().(string) }

// SetModel updates the OpenAI model for subsequent requests.
// This method is thread-safe using atomic operations.
func (p *openAIProvider) SetModel(m string) { p.model.Store(m) }

// buildChatCompletionRequest creates a ChatCompletionRequest from prompt and options.
// This method orchestrates message building and parameter application.
func (p *openAIProvider) buildChatCompletionRequest(prompt string, opts map[string]any) openai.ChatCompletionRequest {
	req := openai.ChatCompletionRequest{
		Model:    p.model.Load().(string),
		Messages: p.buildMessages(prompt, opts),
	}

	p.applyRequestParameters(&req, opts)
	return req
}

// buildMessages creates the messages array from prompt and options.
// Handles both structured messages and simple prompt with optional system prompt.
func (p *openAIProvider) buildMessages(prompt string, opts map[string]any) []openai.ChatCompletionMessage {
	// Use structured messages if provided
	if messages, ok := opts["messages"].([]openai.ChatCompletionMessage); ok {
		return messages
	}

	// Build messages from prompt and system prompt
	messages := make([]openai.ChatCompletionMessage, 0, 2)

	// Add system prompt first if provided
	if systemPrompt, ok := opts["system_prompt"].(string); ok && systemPrompt != "" {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: systemPrompt,
		})
	}

	// Add user prompt
	messages = append(messages, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: prompt,
	})

	return messages
}

// applyRequestParameters applies and validates optional parameters to the request.
// This centralizes parameter validation and application logic using consistent validation functions.
func (p *openAIProvider) applyRequestParameters(req *openai.ChatCompletionRequest, opts map[string]any) {
	if temp, ok := safeFloat32(opts["temperature"]); ok {
		req.Temperature = validateTemperature(temp)
	}

	if maxTokens, ok := safeInt(opts["max_tokens"]); ok {
		if validatedTokens := validateMaxTokens(maxTokens); validatedTokens > 0 {
			req.MaxTokens = validatedTokens
		}
	}

	if topP, ok := safeFloat32(opts["top_p"]); ok {
		if validatedTopP := validateTopP(topP); validatedTopP >= 0 {
			req.TopP = validatedTopP
		}
	}

	if frequencyPenalty, ok := safeFloat32(opts["frequency_penalty"]); ok {
		req.FrequencyPenalty = validatePenalty(frequencyPenalty)
	}

	if presencePenalty, ok := safeFloat32(opts["presence_penalty"]); ok {
		req.PresencePenalty = validatePenalty(presencePenalty)
	}
}

// handleOpenAIError converts OpenAI-specific errors to appropriate error types.
// This provides better error handling and debugging information while preserving context.
func (p *openAIProvider) handleOpenAIError(err error) error {
	var apiErr *openai.APIError
	if errors.As(err, &apiErr) {
		// Preserve additional context from the original error.
		baseMsg := fmt.Sprintf("OpenAI API error (HTTP %d)", apiErr.HTTPStatusCode)
		if apiErr.Code != "" {
			baseMsg += fmt.Sprintf(" [%s]", apiErr.Code)
		}
		if apiErr.Type != "" {
			baseMsg += fmt.Sprintf(" type:%s", apiErr.Type)
		}

		switch apiErr.HTTPStatusCode {
		case 401:
			return fmt.Errorf("%s: authentication failed - check API key: %s", baseMsg, apiErr.Message)
		case 429:
			return fmt.Errorf("%s: rate limit exceeded - consider retry with backoff: %s", baseMsg, apiErr.Message)
		case 500, 502, 503, 504:
			return fmt.Errorf("%s: server error - retry may succeed: %s", baseMsg, apiErr.Message)
		case 400:
			return fmt.Errorf("%s: bad request - check parameters: %s", baseMsg, apiErr.Message)
		case 404:
			return fmt.Errorf("%s: resource not found - check model name: %s", baseMsg, apiErr.Message)
		default:
			return fmt.Errorf("%s: %s", baseMsg, apiErr.Message)
		}
	}

	return fmt.Errorf("OpenAI request failed - check network connectivity: %w", err)
}

// safeFloat32 safely converts various numeric types to float32 with range validation
func safeFloat32(value any) (float32, bool) {
	switch v := value.(type) {
	case float32:
		return v, true
	case float64:
		if v > maxFloat32 || v < minFloat32 {
			return 0, false // Would overflow float32
		}
		if v != 0 && (v > -minFloat32Subnormal && v < minFloat32Subnormal) {
			return 0, false // Would underflow to zero
		}
		return float32(v), true
	case int:
		if v > maxSafeFloat32Integer || v < minSafeFloat32Integer {
			return 0, false // Would lose precision
		}
		return float32(v), true
	case int64:
		if v > maxSafeFloat32Integer || v < minSafeFloat32Integer {
			return 0, false // Would lose precision or overflow
		}
		return float32(v), true
	default:
		return 0, false
	}
}

// safeInt safely converts various numeric types to int with range validation
func safeInt(value any) (int, bool) {
	const maxInt = int(^uint(0) >> 1)
	const minInt = -maxInt - 1

	switch v := value.(type) {
	case int:
		return v, true
	case int64:
		if v > int64(maxInt) || v < int64(minInt) {
			return 0, false // Would overflow int
		}
		return int(v), true
	case float32:
		if v != v || v > float32(maxInt) || v < float32(minInt) {
			return 0, false // NaN, infinity, or would overflow
		}
		return int(v), true // Truncate fractional part
	case float64:
		if v != v || v > float64(maxInt) || v < float64(minInt) {
			return 0, false // NaN, infinity, or would overflow
		}
		return int(v), true // Truncate fractional part
	default:
		return 0, false
	}
}

// validateTemperature ensures temperature is within valid bounds
func validateTemperature(temp float32) float32 {
	if temp < minTemperature {
		return minTemperature
	}
	if temp > maxTemperature {
		return maxTemperature
	}
	return temp
}

// validateMaxTokens ensures max_tokens meets minimum requirement
func validateMaxTokens(tokens int) int {
	if tokens < minMaxTokens {
		return 0 // Let OpenAI use default
	}
	return tokens
}

// validateTopP ensures top_p is within valid bounds
func validateTopP(topP float32) float32 {
	if topP < minTopP {
		return -1 // Invalid, signals to skip parameter
	}
	if topP > maxTopP {
		return maxTopP
	}
	return topP
}

// validatePenalty ensures penalty values are within valid bounds
func validatePenalty(penalty float32) float32 {
	if penalty < minPenalty {
		return minPenalty
	}
	if penalty > maxPenalty {
		return maxPenalty
	}
	return penalty
}

// validateBaseURL ensures the BaseURL is a valid URL format
func validateBaseURL(baseURL string) error {
	parsedURL, err := url.Parse(baseURL)
	if err != nil {
		return fmt.Errorf("malformed URL: %w", err)
	}

	if parsedURL.Scheme == "" {
		return fmt.Errorf("URL must include scheme (http:// or https://)")
	}

	if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
		return fmt.Errorf("URL scheme must be http or https, got: %s", parsedURL.Scheme)
	}

	if parsedURL.Host == "" {
		return fmt.Errorf("URL must include host")
	}

	return nil
}

// validateTimeout ensures timeout is within reasonable bounds
func validateTimeout(timeout time.Duration) time.Duration {
	if timeout < minTimeout {
		return minTimeout
	}
	if timeout > maxTimeout {
		return maxTimeout
	}
	return timeout
}

// estimateTokens provides a simple token estimation for fallback scenarios.
// This uses a character-based heuristic approximating GPT tokenization.
// Note: This is only used when OpenAI doesn't provide actual token counts.
func (p *openAIProvider) estimateTokens(text string) int {
	// Rough estimation: 1 token ≈ 4 characters for English text
	// This is a conservative estimate used only as fallback
	if len(text) == 0 {
		return 0
	}
	return (len(text) + 3) / 4
}
