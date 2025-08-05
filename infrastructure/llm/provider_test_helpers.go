package llm

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockProvider represents a mock implementation of the CoreLLM interface.
// It is used for testing provider-agnostic logic without making actual API calls.
type MockProvider struct {
	BaseProvider
	// DoRequestFunc allows injecting custom logic into the DoRequest method.
	// If nil, a default mock response is returned.
	DoRequestFunc func(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error)
	// CallCount tracks the number of times DoRequest has been invoked.
	CallCount int
}

// DoRequest implements the CoreLLM interface for MockProvider.
// It increments the call count and executes DoRequestFunc if it is defined.
// Otherwise, it returns a default success response.
func (m *MockProvider) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
	m.CallCount++
	if m.DoRequestFunc != nil {
		return m.DoRequestFunc(ctx, prompt, opts)
	}
	return "mock response", 10, 5, nil
}

// ProviderTestSuite defines a standardized suite of tests for any CoreLLM provider.
// It ensures that all provider implementations adhere to a common contract.
type ProviderTestSuite struct {
	t        *testing.T
	provider CoreLLM
	config   ClientConfig
}

// NewProviderTestSuite creates a new test suite for a given provider.
// It initializes the provider using its registered factory and configuration.
// The function will fail the test if the factory is not found or if provider creation fails.
func NewProviderTestSuite(t *testing.T, factoryName string, config ClientConfig) *ProviderTestSuite {
	factory, exists := GetProviderFactory(factoryName)
	if !exists {
		t.Fatalf("Provider factory %s not found", factoryName)
	}

	provider, err := factory(config)
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	return &ProviderTestSuite{
		t:        t,
		provider: provider,
		config:   config,
	}
}

// GetProviderFactory retrieves a provider factory function from the registry by name.
// It returns the factory and a boolean indicating whether the factory was found.
func GetProviderFactory(name string) (ProviderFactory, bool) {
	factory, exists := providerFactories[name]
	return factory, exists
}

// TestBasicRequest verifies that a provider can handle a simple, successful request.
// It checks for a non-empty response and positive token counts.
func (pts *ProviderTestSuite) TestBasicRequest() {
	ctx := context.Background()
	response, tokensIn, tokensOut, err := pts.provider.DoRequest(ctx, "Hello, world!", nil)

	require.NoError(pts.t, err, "Basic request should not fail")
	assert.NotEmpty(pts.t, response, "Response should not be empty")
	assert.Positive(pts.t, tokensIn, "Input token count should be positive")
	assert.Positive(pts.t, tokensOut, "Output token count should be positive")
}

// TestRequestWithOptions validates that the provider correctly handles various request options.
// It runs sub-tests for temperature, max tokens, system prompts, and combinations of options.
func (pts *ProviderTestSuite) TestRequestWithOptions() {
	testCases := []struct {
		name string
		opts map[string]any
	}{
		{
			name: "with temperature",
			opts: map[string]any{"temperature": 0.7},
		},
		{
			name: "with max tokens",
			opts: map[string]any{"max_tokens": 100},
		},
		{
			name: "with system prompt",
			opts: map[string]any{"system": "You are a helpful assistant."},
		},
		{
			name: "with all options",
			opts: map[string]any{
				"temperature": 0.5,
				"max_tokens":  150,
				"system":      "You are a helpful assistant.",
				"top_p":       0.9,
			},
		},
	}

	ctx := context.Background()
	for _, tc := range testCases {
		pts.t.Run(tc.name, func(t *testing.T) {
			response, _, _, err := pts.provider.DoRequest(ctx, "Test prompt", tc.opts)
			require.NoError(t, err, "Request with options should not fail")
			assert.NotEmpty(t, response, "Response should not be empty")
		})
	}
}

// TestErrorHandling ensures the provider behaves gracefully with invalid inputs.
// It checks scenarios like empty prompts and out-of-range option values.
// The primary goal is to ensure the provider does not panic.
func (pts *ProviderTestSuite) TestErrorHandling() {
	testCases := []struct {
		name   string
		prompt string
		opts   map[string]any
	}{
		{
			name:   "empty prompt",
			prompt: "",
			opts:   nil,
		},
		{
			name:   "invalid temperature",
			prompt: "test",
			opts:   map[string]any{"temperature": 3.0}, // Out of valid range.
		},
		{
			name:   "negative max tokens",
			prompt: "test",
			opts:   map[string]any{"max_tokens": -1},
		},
	}

	ctx := context.Background()
	for _, tc := range testCases {
		pts.t.Run(tc.name, func(t *testing.T) {
			_, _, _, _ = pts.provider.DoRequest(ctx, tc.prompt, tc.opts)
			// The test ensures that invalid inputs do not cause a panic.
			// Specific error handling is checked in other tests.
		})
	}
}

// TestContextCancellation confirms that the provider correctly handles a canceled context.
// It expects a network-type error when the request context is canceled before the call.
func (pts *ProviderTestSuite) TestContextCancellation() {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, _, _, err := pts.provider.DoRequest(ctx, "Test prompt", nil)
	require.Error(pts.t, err, "Expected error for cancelled context")

	if provErr, ok := err.(*ProviderError); ok {
		assert.Equal(pts.t, ErrorTypeNetwork, provErr.Type, "Expected network error type for cancelled context")
	}
}

// TestTimeout checks that a request correctly times out when the context deadline is exceeded.
func (pts *ProviderTestSuite) TestTimeout() {
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
	defer cancel()

	time.Sleep(2 * time.Millisecond)

	_, _, _, err := pts.provider.DoRequest(ctx, "Test prompt", nil)
	assert.Error(pts.t, err, "Expected timeout error")
}

// TestModelGetterSetter validates the provider's GetModel and SetModel methods.
// It ensures that the model can be updated and restored correctly.
func (pts *ProviderTestSuite) TestModelGetterSetter() {
	originalModel := pts.provider.GetModel()

	newModel := "test-model-123"
	pts.provider.SetModel(newModel)

	assert.Equal(pts.t, newModel, pts.provider.GetModel(), "Model should be updated correctly")

	// Restore the original model to avoid side effects in other tests.
	pts.provider.SetModel(originalModel)
}
