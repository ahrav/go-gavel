package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	openai "github.com/sashabaranov/go-openai"
)

const (
	// OpenAI provider constants
	OpenAIDefaultModel = "gpt-3.5-turbo"
)

func init() {
	RegisterProviderFactory("openai", newOpenAIProvider)
}

// openAIProvider implements the CoreLLM interface for OpenAI's API.
// This provider handles OpenAI-specific request formatting and response parsing
// while conforming to the common interface for middleware compatibility.
type openAIProvider struct {
	BaseProvider
	client          *openai.Client
	tokenCounter    *TokenCounter
	errorClassifier *ErrorClassifier
}

// newOpenAIProvider creates a new OpenAI provider instance.
// This factory function initializes the provider with configuration
// and validates required settings like API key presence.
func newOpenAIProvider(config ClientConfig) (CoreLLM, error) {
	if config.APIKey == "" {
		return nil, ErrEmptyAPIKey
	}

	model := config.Model
	if model == "" {
		model = OpenAIDefaultModel
	}

	clientConfig := openai.DefaultConfig(config.APIKey)

	if config.BaseURL != "" {
		validatedURL, err := ValidateBaseURL(config.BaseURL)
		if err != nil {
			return nil, fmt.Errorf("invalid BaseURL: %w", err)
		}
		clientConfig.BaseURL = validatedURL
	}

	if config.Timeout > 0 {
		validatedTimeout := ValidateTimeout(config.Timeout)
		clientConfig.HTTPClient = &http.Client{
			Timeout: validatedTimeout,
		}
	}

	client := openai.NewClientWithConfig(clientConfig)

	return &openAIProvider{
		BaseProvider:    BaseProvider{model: model},
		client:          client,
		tokenCounter:    NewTokenCounter(),
		errorClassifier: &ErrorClassifier{Provider: "openai"},
	}, nil
}

// DoRequest sends a request to the OpenAI API and returns the response.
// It handles OpenAI-specific request formatting, authentication, and response parsing,
// and returns the generated content along with token usage data.
func (p *openAIProvider) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
	options := ParseRequestOptions(opts, p.model)

	req := p.buildChatCompletionRequest(prompt, options)
	resp, err := p.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", 0, 0, p.handleError(err)
	}

	if len(resp.Choices) == 0 {
		return "", 0, 0, ErrNoResponseChoice
	}

	content := resp.Choices[0].Message.Content

	tokensIn := p.getTokenCount(resp.Usage.PromptTokens, prompt)
	tokensOut := p.getTokenCount(resp.Usage.CompletionTokens, content)

	return content, tokensIn, tokensOut, nil
}

// getTokenCount returns the token count for the given text.
// It prioritizes the actual count from the API response if available,
// falling back to an estimation if the count is zero.
func (p *openAIProvider) getTokenCount(actualCount int, text string) int {
	if actualCount > 0 {
		return actualCount
	}
	return p.tokenCounter.EstimateTokens(text)
}

// buildChatCompletionRequest creates an openai.ChatCompletionRequest from a prompt and options.
// This method orchestrates message building and the application of request parameters.
func (p *openAIProvider) buildChatCompletionRequest(prompt string, options RequestOptions) openai.ChatCompletionRequest {
	req := openai.ChatCompletionRequest{
		Model:    options.Model,
		Messages: p.buildMessages(prompt, options),
	}

	p.applyRequestParameters(&req, options)
	return req
}

// buildMessages creates the message slice for an OpenAI chat completion request.
// It constructs the messages from the user prompt and an optional system prompt.
func (p *openAIProvider) buildMessages(prompt string, options RequestOptions) []openai.ChatCompletionMessage {
	messages := make([]openai.ChatCompletionMessage, 0, 2)

	if options.System != "" {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: options.System,
		})
	}

	messages = append(messages, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: prompt,
	})

	return messages
}

// applyRequestParameters applies and validates optional parameters to the request.
// This method centralizes parameter validation and application logic.
func (p *openAIProvider) applyRequestParameters(req *openai.ChatCompletionRequest, options RequestOptions) {
	if options.Temperature != nil {
		// OpenAI API supports a temperature range of 0.0 to 2.0.
		temp := ClampFloat64(*options.Temperature, 0.0, 2.0)
		req.Temperature = float32(temp)
	}

	if options.MaxTokens > 0 {
		req.MaxTokens = options.MaxTokens
	}

	if options.TopP != nil {
		topP := ClampFloat64(*options.TopP, 0.0, 1.0)
		req.TopP = float32(topP)
	}

	// Handle provider-specific options.
	if frequencyPenalty, ok := options.Extra["frequency_penalty"]; ok {
		if penalty, valid := SafeFloat32(frequencyPenalty); valid {
			req.FrequencyPenalty = float32(ClampFloat64(float64(penalty), MinPenalty, MaxPenalty))
		}
	}

	if presencePenalty, ok := options.Extra["presence_penalty"]; ok {
		if penalty, valid := SafeFloat32(presencePenalty); valid {
			req.PresencePenalty = float32(ClampFloat64(float64(penalty), MinPenalty, MaxPenalty))
		}
	}
}

// handleError classifies and wraps errors from the OpenAI API.
// It distinguishes between context-related errors, API errors, and other failures,
// wrapping them in standardized error types.
func (p *openAIProvider) handleError(err error) error {
	// Check for context errors first.
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
		return p.errorClassifier.ClassifyContextError(err)
	}

	// Handle OpenAI API errors.
	var apiErr *openai.APIError
	if errors.As(err, &apiErr) {
		message := apiErr.Message
		if message == "" {
			message = "unknown error"
		}

		return p.errorClassifier.ClassifyHTTPError(apiErr.HTTPStatusCode, message, err)
	}

	// Fallback for generic or unknown errors.
	return NewProviderError("openai", ErrorTypeUnknown, 0, "request failed", err)
}
