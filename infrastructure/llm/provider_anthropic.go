package llm

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// Anthropic provider constants
const (
	// AnthropicDefaultModel is the default Anthropic model (Claude 3.5 Sonnet)
	AnthropicDefaultModel = "claude-3-5-sonnet-20241022"
)

func init() {
	RegisterProviderFactory("anthropic", newAnthropicProvider)
}

// anthropicProvider implements the CoreLLM interface for Anthropic's Claude API.
// This provider handles Anthropic-specific request formatting and response parsing
// while maintaining compatibility with the common middleware system.
type anthropicProvider struct {
	client anthropic.Client
	model  string
}

// requestConfig holds parsed request configuration
type requestConfig struct {
	maxTokens   int
	model       string
	temperature *float64
	system      string
}

// newAnthropicProvider creates a new Anthropic provider instance.
// This factory function configures the provider for Anthropic's API
// and validates that required configuration is present.
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
		opts = append(opts, option.WithBaseURL(config.BaseURL))
	}

	client := anthropic.NewClient(opts...)

	return &anthropicProvider{
		client: client,
		model:  model,
	}, nil
}

// DoRequest sends a request to Anthropic's Claude API and returns the response.
// This method handles Anthropic-specific request formatting, authentication,
// and response parsing while tracking token usage.
func (p *anthropicProvider) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
	config := p.parseRequestOptions(opts)
	params := p.buildAnthropicParams(prompt, config)

	message, err := p.client.Messages.New(ctx, params)
	if err != nil {
		return "", 0, 0, p.wrapError(err)
	}

	return p.processResponse(message, prompt)
}

// parseRequestOptions extracts and validates request options with defaults
func (p *anthropicProvider) parseRequestOptions(opts map[string]any) requestConfig {
	config := requestConfig{
		maxTokens: ExtractOptionalInt(opts, "max_tokens", DefaultMaxTokens, IsPositiveInt),
		model:     ExtractOptionalString(opts, "model", p.model, IsNonEmptyString),
		system:    ExtractOptionalString(opts, "system", "", nil), // Empty string is valid for system
	}

	if temp := ExtractOptionalFloat64(opts, "temperature", -1, IsValidTemperature); temp != -1 {
		config.temperature = &temp
	}

	return config
}

// buildAnthropicParams creates the API request parameters
func (p *anthropicProvider) buildAnthropicParams(prompt string, config requestConfig) anthropic.MessageNewParams {
	messages := []anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.NewTextBlock(prompt)),
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(config.model),
		MaxTokens: int64(config.maxTokens),
		Messages:  messages,
	}

	if config.temperature != nil {
		params.Temperature = anthropic.Float(*config.temperature)
	}

	if config.system != "" {
		params.System = []anthropic.TextBlockParam{{Text: config.system}}
	}

	return params
}

// processResponse extracts content and token counts from the API response
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

// getTokenCount returns the actual token count from API or falls back to estimation
func (p *anthropicProvider) getTokenCount(apiTokens int64, text string) int {
	if apiTokens > 0 {
		return int(apiTokens)
	}
	return EstimateTokens(text)
}

// wrapError wraps Anthropic SDK errors with additional context and more specific error types
func (p *anthropicProvider) wrapError(err error) error {
	var anthropicErr *anthropic.Error
	if errors.As(err, &anthropicErr) {
		switch anthropicErr.StatusCode {
		case 401:
			return fmt.Errorf("anthropic authentication failed: check API key (%d): %w", anthropicErr.StatusCode, err)
		case 429:
			return fmt.Errorf("anthropic rate limit exceeded: %w", err)
		case 400:
			return fmt.Errorf("anthropic bad request: check parameters (%d): %w", anthropicErr.StatusCode, err)
		case 500, 502, 503, 504:
			return fmt.Errorf("anthropic server error (%d): %w", anthropicErr.StatusCode, err)
		default:
			return fmt.Errorf("anthropic API error (%d): %w", anthropicErr.StatusCode, err)
		}
	}

	// Handle context errors specifically
	if errors.Is(err, context.DeadlineExceeded) {
		return fmt.Errorf("anthropic request timeout: %w", err)
	}
	if errors.Is(err, context.Canceled) {
		return fmt.Errorf("anthropic request canceled: %w", err)
	}

	return fmt.Errorf("anthropic request failed: %w", err)
}

// GetModel returns the currently configured Anthropic model name.
func (p *anthropicProvider) GetModel() string { return p.model }

// SetModel updates the Anthropic model for subsequent requests.
func (p *anthropicProvider) SetModel(m string) { p.model = m }
