// Package llm provides a standardized interface for interacting with various
// large language model (LLM) providers. It abstracts provider-specific
// details, offering a unified client for making generation requests,
// handling authentication, and managing configurations. The package is designed
// for extensibility, allowing new providers to be integrated by implementing
// the core interfaces.
package llm

import (
	"fmt"
	"sync"
)

// BaseProvider provides common, thread-safe functionality for all LLM providers,
// primarily for managing the model name.
type BaseProvider struct {
	mu    sync.RWMutex
	model string
}

// GetModel returns the name of the model currently configured for the provider.
// It is safe for concurrent use.
func (b *BaseProvider) GetModel() string {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.model
}

// SetModel updates the model name for the provider.
// It is safe for concurrent use.
func (b *BaseProvider) SetModel(model string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.model = model
}

// RequestOptions represents a standardized set of configuration parameters for an LLM request.
// It consolidates common settings across different providers.
type RequestOptions struct {
	// MaxTokens specifies the maximum number of tokens to generate.
	MaxTokens int
	// Model is the identifier of the language model to use for the request.
	Model string
	// Temperature controls the randomness of the output.
	// A higher value (e.g., 0.8) results in more creative and varied responses,
	// while a lower value (e.g., 0.2) produces more deterministic and focused output.
	// A nil value indicates that the provider's default should be used.
	Temperature *float64
	// TopP is an alternative to temperature sampling, known as nucleus sampling.
	// It selects the most probable tokens whose cumulative probability mass exceeds a certain threshold.
	// A nil value indicates that the provider's default should be used.
	TopP *float64
	// System provides instructions or context to the model,
	// guiding its behavior and response style for the conversation.
	System string
	// Extra holds any provider-specific options that are not part of the standardized set.
	// This allows for flexible configuration of unique provider features.
	Extra map[string]any
}

// ParseRequestOptions extracts and validates LLM request parameters from a map.
// It populates a RequestOptions struct with standardized values,
// using provided defaults for any missing or invalid entries.
// Any unrecognized options are collected into the Extra field.
func ParseRequestOptions(opts map[string]any, defaultModel string) RequestOptions {
	options := RequestOptions{
		MaxTokens: ExtractOptionalInt(opts, "max_tokens", DefaultMaxTokens, IsPositiveInt),
		Model:     ExtractOptionalString(opts, "model", defaultModel, IsNonEmptyString),
		System:    ExtractOptionalString(opts, "system", "", nil),
		Extra:     make(map[string]any),
	}

	if temp := ExtractOptionalFloat64(opts, "temperature", -1, IsValidTemperature); temp != -1 {
		options.Temperature = &temp
	}

	if topP := ExtractOptionalFloat64(opts, "top_p", -1, IsValidTopP); topP != -1 {
		options.TopP = &topP
	}

	// Collect any provider-specific options that were not handled above.
	for k, v := range opts {
		switch k {
		case "max_tokens", "model", "system", "temperature", "top_p":
		// These are standard options and have already been processed.
		default:
			options.Extra[k] = v
		}
	}

	return options
}

// TokenCounter provides a utility for estimating token counts from text.
// This is useful when an exact tokenizer is not available for a given model.
type TokenCounter struct {
	// CharactersPerToken represents the average number of characters per token.
	// This value is an approximation and can be adjusted based on the specific model or language.
	CharactersPerToken float64
}

// NewTokenCounter creates a new TokenCounter with a default character-per-token ratio.
// The default is a general approximation suitable for English text.
func NewTokenCounter() *TokenCounter {
	return &TokenCounter{
		CharactersPerToken: 4.0, // A common approximation for English text.
	}
}

// EstimateTokens calculates an estimated token count for a given string of text.
// The estimation is based on the configured CharactersPerToken ratio.
func (tc *TokenCounter) EstimateTokens(text string) int {
	if len(text) == 0 {
		return 0
	}
	return int(float64(len(text)) / tc.CharactersPerToken)
}

// GetTokenCount returns the actual token count if it is available and positive.
// Otherwise, it falls back to estimating the count based on the provided text.
func (tc *TokenCounter) GetTokenCount(actualCount int, text string) int {
	if actualCount > 0 {
		return actualCount
	}
	return tc.EstimateTokens(text)
}

// ErrorWrapper standardizes error messages originating from different LLM providers.
// It adds consistent, provider-specific context to errors.
type ErrorWrapper struct {
	// ProviderName is the name of the LLM provider, used as a prefix in error messages.
	ProviderName string
}

// WrapError enhances a given error with additional context,
// including the provider's name.
// If the input error is nil, it returns nil.
func (ew *ErrorWrapper) WrapError(err error, context string) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("%s %s: %w", ew.ProviderName, context, err)
}

// WrapAuthError wraps an error to indicate it occurred during authentication.
func (ew *ErrorWrapper) WrapAuthError(err error) error {
	return ew.WrapError(err, "authentication failed")
}

// WrapRateLimitError wraps an error to indicate that a rate limit was exceeded.
func (ew *ErrorWrapper) WrapRateLimitError(err error) error {
	return ew.WrapError(err, "rate limit exceeded")
}

// WrapServerError wraps an error to indicate a server-side issue from the provider,
// including the HTTP status code.
func (ew *ErrorWrapper) WrapServerError(err error, code int) error {
	return ew.WrapError(err, fmt.Sprintf("server error (%d)", code))
}

// SystemPromptHandler defines an interface for processing system prompts.
// Different LLM providers handle system prompts in unique ways;
// this interface abstracts those differences.
type SystemPromptHandler interface {
	// HandleSystemPrompt processes a user prompt and a system prompt,
	// returning a modified prompt and a system parameter suitable for the provider's API.
	HandleSystemPrompt(userPrompt, systemPrompt string) (processedPrompt string, systemParam any)
}

// DefaultSystemPromptHandler provides a default strategy for handling system prompts
// by prepending the system prompt to the user prompt.
type DefaultSystemPromptHandler struct{}

// HandleSystemPrompt combines the system and user prompts into a single string.
// If the system prompt is empty, it returns the user prompt unmodified.
func (d *DefaultSystemPromptHandler) HandleSystemPrompt(userPrompt, systemPrompt string) (string, any) {
	if systemPrompt == "" {
		return userPrompt, nil
	}
	return systemPrompt + "\n\n" + userPrompt, nil
}

// SeparateSystemPromptHandler provides a strategy for providers that handle
// system prompts as a distinct parameter, separate from the user prompt.
type SeparateSystemPromptHandler struct{}

// HandleSystemPrompt returns the user prompt unmodified and passes the system prompt
// through as a separate parameter.
// If the system prompt is empty, the system parameter will be nil.
func (s *SeparateSystemPromptHandler) HandleSystemPrompt(userPrompt, systemPrompt string) (string, any) {
	if systemPrompt == "" {
		return userPrompt, nil
	}
	return userPrompt, systemPrompt
}
