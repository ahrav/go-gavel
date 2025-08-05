// Package llm provides a unified interface for interacting with various Large
// Language Model (LLM) providers. It abstracts provider-specific details,
// offering a consistent API for making requests, handling responses, and managing
// configurations. This package is designed to be extensible, allowing new
// providers to be added by implementing the CoreLLM interface.
package llm

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// Anthropic provider constants define default values and identifiers.
const (
	// AnthropicDefaultModel is the default model used for Anthropic API calls.
	// It is currently set to Claude 3.5 Sonnet.
	AnthropicDefaultModel = "claude-3-5-sonnet-20241022"
)

func init() {
	// Registers the Anthropic provider with the central provider factory.
	// This allows the factory to create instances of the Anthropic provider
	// when requested by name.
	RegisterProviderFactory("anthropic", newAnthropicProvider)
}

// AnthropicProvider implements the CoreLLM interface for Anthropic's Claude API.
// It handles Anthropic-specific request formatting, response parsing, and error
// handling, while conforming to the common interface for middleware
// compatibility.
type anthropicProvider struct {
	BaseProvider
	client          anthropic.Client
	tokenCounter    *TokenCounter
	errorClassifier *ErrorClassifier
}

// newAnthropicProvider creates a new Anthropic provider instance.
// This factory function configures the provider for Anthropic's API and
// validates that the required configuration, such as the API key, is present.
func newAnthropicProvider(config ClientConfig) (CoreLLM, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("anthropic API key cannot be empty")
	}

	model := config.Model
	if model == "" {
		model = AnthropicDefaultModel
	}

	opts := []option.RequestOption{option.WithAPIKey(config.APIKey)}
	if config.BaseURL != "" {
		validatedURL, err := ValidateBaseURL(config.BaseURL)
		if err != nil {
			return nil, fmt.Errorf("invalid BaseURL: %w", err)
		}
		opts = append(opts, option.WithBaseURL(validatedURL))
	}

	client := anthropic.NewClient(opts...)

	return &anthropicProvider{
		BaseProvider:    BaseProvider{model: model},
		client:          client,
		tokenCounter:    NewTokenCounter(),
		errorClassifier: &ErrorClassifier{Provider: "anthropic"},
	}, nil
}

// DoRequest sends a request to the Anthropic API and returns the response.
// This method formats the request, handles authentication, and parses the
// response, while also tracking token usage for both the prompt and the
// completion.
func (p *anthropicProvider) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
	options := ParseRequestOptions(opts, p.model)
	params := p.buildAnthropicParams(prompt, options)

	message, err := p.client.Messages.New(ctx, params)
	if err != nil {
		return "", 0, 0, p.handleError(err)
	}

	return p.processResponse(message, prompt)
}

// buildAnthropicParams creates the API request parameters with proper validation.
// It constructs the message list and sets model-specific options like
// temperature and max tokens.
func (p *anthropicProvider) buildAnthropicParams(prompt string, options RequestOptions) anthropic.MessageNewParams {
	messages := []anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.NewTextBlock(prompt)),
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(options.Model),
		MaxTokens: int64(options.MaxTokens),
		Messages:  messages,
	}

	// Anthropic's API requires the temperature to be between 0.0 and 1.0.
	// This check ensures we only send a valid value.
	if options.Temperature != nil && *options.Temperature >= 0.0 && *options.Temperature <= 1.0 {
		params.Temperature = anthropic.Float(*options.Temperature)
	}

	if options.System != "" {
		params.System = []anthropic.TextBlockParam{{Text: options.System}}
	}

	return params
}

// processResponse extracts the text content and token counts from the API
// response. It handles cases where the response might be empty and ensures
// consistent token counting.
func (p *anthropicProvider) processResponse(message *anthropic.Message, originalPrompt string) (string, int, int, error) {
	var responseText strings.Builder
	for _, block := range message.Content {
		switch content := block.AsAny().(type) {
		case anthropic.TextBlock:
			responseText.WriteString(content.Text)
		}
	}

	responseStr := responseText.String()
	if responseStr == "" {
		return "", 0, 0, fmt.Errorf("empty response from Anthropic API")
	}

	tokensIn := p.getTokenCount(message.Usage.InputTokens, originalPrompt)
	tokensOut := p.getTokenCount(message.Usage.OutputTokens, responseStr)

	return responseStr, tokensIn, tokensOut, nil
}

// getTokenCount returns the token count from the API if available.
// Otherwise, it falls back to a local estimation. This ensures token counts are
// always populated, even if the API does not provide them.
func (p *anthropicProvider) getTokenCount(apiTokens int64, text string) int {
	if apiTokens > 0 {
		return int(apiTokens)
	}
	return p.tokenCounter.EstimateTokens(text)
}

// handleError provides structured error handling for Anthropic API calls.
// It classifies errors into standard categories, such as context-related
// errors, HTTP status code errors, or unknown provider errors.
func (p *anthropicProvider) handleError(err error) error {
	// Check for context errors first, as they are common and should be
	// handled specifically.
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
		return p.errorClassifier.ClassifyContextError(err)
	}

	// Handle specific Anthropic SDK errors by inspecting the error type.
	var anthropicErr *anthropic.Error
	if errors.As(err, &anthropicErr) {
		message := anthropicErr.Error()
		if message == "" {
			message = "unknown error"
		}
		return p.errorClassifier.ClassifyHTTPError(anthropicErr.StatusCode, message, err)
	}

	// Fallback for any other type of error.
	return NewProviderError("anthropic", ErrorTypeUnknown, 0, "request failed", err)
}
