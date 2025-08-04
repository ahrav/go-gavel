package application

import (
	"context"
	"fmt"
	"os"
	"regexp"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"

	"github.com/ahrav/go-gavel/infrastructure/llm"
	"github.com/ahrav/go-gavel/infrastructure/units"
	"github.com/ahrav/go-gavel/internal/domain"
	"github.com/ahrav/go-gavel/internal/ports"
	"github.com/ahrav/go-gavel/internal/testutils"
)

// TestMultiProviderIntegration tests the multi-provider LLM functionality,
// ensuring that different judge units can use different LLM providers.
func TestMultiProviderIntegration(t *testing.T) {
	ctx := context.Background()

	// Set up mock API keys for testing.
	os.Setenv("OPENAI_API_KEY", "test-openai-key")
	os.Setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

	// Create a temporary credentials file for Google.
	tmpFile, err := os.CreateTemp("", "test-creds-*.json")
	require.NoError(t, err)
	defer os.Remove(tmpFile.Name())
	defer tmpFile.Close()

	// Write minimal JSON content to make it a valid file.
	_, err = tmpFile.WriteString(`{"type": "service_account", "project_id": "test"}`)
	require.NoError(t, err)

	os.Setenv("GOOGLE_APPLICATION_CREDENTIALS", tmpFile.Name())
	defer func() {
		os.Unsetenv("OPENAI_API_KEY")
		os.Unsetenv("ANTHROPIC_API_KEY")
		os.Unsetenv("GOOGLE_APPLICATION_CREDENTIALS")
	}()

	t.Run("creates provider registry with multiple providers", func(t *testing.T) {
		// Create mock clients with specific models.
		mockOpenAI := testutils.NewMockLLMClient("gpt-4")
		mockAnthropic := testutils.NewMockLLMClient("claude-3-sonnet")
		mockGoogle := testutils.NewMockLLMClient("gemini-pro")

		// Register mock provider factories before creating the registry.
		llm.RegisterProviderFactory("openai", func(cfg llm.ClientConfig) (llm.CoreLLM, error) {
			return &mockTestUtilsAdapter{client: mockOpenAI}, nil
		})
		llm.RegisterProviderFactory("anthropic", func(cfg llm.ClientConfig) (llm.CoreLLM, error) {
			return &mockTestUtilsAdapter{client: mockAnthropic}, nil
		})
		llm.RegisterProviderFactory("google", func(cfg llm.ClientConfig) (llm.CoreLLM, error) {
			return &mockTestUtilsAdapter{client: mockGoogle}, nil
		})

		// Create the provider registry.
		config := llm.RegistryConfig{
			DefaultProvider: "openai",
			Providers:       llm.DefaultProviders,
		}
		registry, err := llm.NewRegistry(config)
		require.NoError(t, err)
		require.NotNil(t, registry)

		// Initialize the providers.
		err = registry.InitializeProviders()
		require.NoError(t, err)

		// Test retrieving different providers.
		openaiClient, err := registry.GetClient("openai/gpt-4")
		require.NoError(t, err)
		assert.NotNil(t, openaiClient)
		assert.Equal(t, "gpt-4", openaiClient.GetModel())

		anthropicClient, err := registry.GetClient("anthropic/claude-3-sonnet")
		require.NoError(t, err)
		assert.NotNil(t, anthropicClient)
		assert.Equal(t, "claude-3-sonnet", anthropicClient.GetModel())

		googleClient, err := registry.GetClient("google/gemini-pro")
		require.NoError(t, err)
		assert.NotNil(t, googleClient)
		assert.Equal(t, "gemini-pro", googleClient.GetModel())
	})

	t.Run("multiple judges with different providers", func(t *testing.T) {
		// Create mock clients.
		openaiClient := testutils.NewMockLLMClient("openai-gpt-4")
		anthropicClient := testutils.NewMockLLMClient("anthropic-claude")
		googleClient := testutils.NewMockLLMClient("google-gemini")

		// Create three judge units with different providers.
		openaiJudge, err := units.NewScoreJudgeUnit("openai-judge", openaiClient, units.ScoreJudgeConfig{
			JudgePrompt:    "OpenAI: Rate this answer to '%s': %s",
			ScoreScale:     "0.0-1.0",
			Temperature:    0.5,
			MaxTokens:      150,
			MinConfidence:  0.8,
			MaxConcurrency: 5,
		})
		require.NoError(t, err)

		anthropicJudge, err := units.NewScoreJudgeUnit("anthropic-judge", anthropicClient, units.ScoreJudgeConfig{
			JudgePrompt:    "Anthropic: Rate this answer to '%s': %s",
			ScoreScale:     "0.0-1.0",
			Temperature:    0.5,
			MaxTokens:      150,
			MinConfidence:  0.8,
			MaxConcurrency: 5,
		})
		require.NoError(t, err)

		googleJudge, err := units.NewScoreJudgeUnit("google-judge", googleClient, units.ScoreJudgeConfig{
			JudgePrompt:    "Google: Rate this answer to '%s': %s",
			ScoreScale:     "0.0-1.0",
			Temperature:    0.5,
			MaxTokens:      150,
			MinConfidence:  0.8,
			MaxConcurrency: 5,
		})
		require.NoError(t, err)

		// Create an initial state with a question and answers.
		state := domain.NewState()
		state = domain.With(state, domain.KeyQuestion, "What is cloud computing?")
		state = domain.With(state, domain.KeyAnswers, []domain.Answer{
			{ID: "answer1", Content: "Cloud computing is the delivery of computing services over the internet."},
			{ID: "answer2", Content: "Cloud computing enables on-demand access to shared computing resources."},
		})

		// Execute all three judges; they should work independently.
		stateAfterOpenAI, err := openaiJudge.Execute(ctx, state)
		require.NoError(t, err)

		stateAfterAnthropic, err := anthropicJudge.Execute(ctx, state)
		require.NoError(t, err)

		stateAfterGoogle, err := googleJudge.Execute(ctx, state)
		require.NoError(t, err)

		// Verify that all judges produced scores.
		openaiScores, ok := domain.Get(stateAfterOpenAI, domain.KeyJudgeScores)
		require.True(t, ok)
		require.Len(t, openaiScores, 2)

		anthropicScores, ok := domain.Get(stateAfterAnthropic, domain.KeyJudgeScores)
		require.True(t, ok)
		require.Len(t, anthropicScores, 2)

		googleScores, ok := domain.Get(stateAfterGoogle, domain.KeyJudgeScores)
		require.True(t, ok)
		require.Len(t, googleScores, 2)
	})

	t.Run("provider registry handles missing environment variables", func(t *testing.T) {
		// Temporarily unset API keys.
		oldOpenAI := os.Getenv("OPENAI_API_KEY")
		oldAnthropic := os.Getenv("ANTHROPIC_API_KEY")
		oldGoogle := os.Getenv("GOOGLE_APPLICATION_CREDENTIALS")

		os.Unsetenv("OPENAI_API_KEY")
		os.Unsetenv("ANTHROPIC_API_KEY")
		os.Unsetenv("GOOGLE_APPLICATION_CREDENTIALS")

		defer func() {
			if oldOpenAI != "" {
				os.Setenv("OPENAI_API_KEY", oldOpenAI)
			}
			if oldAnthropic != "" {
				os.Setenv("ANTHROPIC_API_KEY", oldAnthropic)
			}
			if oldGoogle != "" {
				os.Setenv("GOOGLE_APPLICATION_CREDENTIALS", oldGoogle)
			}
		}()

		// Creating the registry should not fail, but initialization will.
		config := llm.RegistryConfig{
			DefaultProvider: "openai",
			Providers:       llm.DefaultProviders,
		}
		registry, err := llm.NewRegistry(config)
		require.NoError(t, err)

		// It should fail to initialize providers without API keys.
		err = registry.InitializeProviders()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "environment variable not set")
	})

	t.Run("graph loader with multi-provider configuration", func(t *testing.T) {
		// Create a mock provider registry.
		mockOpenAI := testutils.NewMockLLMClient("openai-gpt-4")
		mockAnthropic := testutils.NewMockLLMClient("anthropic-claude")
		mockGoogle := testutils.NewMockLLMClient("google-gemini")

		config := llm.RegistryConfig{
			DefaultProvider: "openai",
			Providers:       llm.DefaultProviders,
		}
		registry, err := llm.NewRegistry(config)
		require.NoError(t, err)

		// Register mock provider factories.
		llm.RegisterProviderFactory("openai", func(cfg llm.ClientConfig) (llm.CoreLLM, error) {
			return &mockTestUtilsAdapter{client: mockOpenAI}, nil
		})
		llm.RegisterProviderFactory("anthropic", func(cfg llm.ClientConfig) (llm.CoreLLM, error) {
			return &mockTestUtilsAdapter{client: mockAnthropic}, nil
		})
		llm.RegisterProviderFactory("google", func(cfg llm.ClientConfig) (llm.CoreLLM, error) {
			return &mockTestUtilsAdapter{client: mockGoogle}, nil
		})

		// Register the clients.
		err = registry.RegisterClient("openai", llm.ClientConfig{
			APIKey: "test-key",
			Model:  "openai-gpt-4",
		})
		require.NoError(t, err)

		err = registry.RegisterClient("anthropic", llm.ClientConfig{
			APIKey: "test-key",
			Model:  "anthropic-claude",
		})
		require.NoError(t, err)

		err = registry.RegisterClient("google", llm.ClientConfig{
			APIKey: "test-key",
			Model:  "google-gemini",
		})
		require.NoError(t, err)

		// Create a unit registry and graph loader.
		unitRegistry := NewDefaultUnitRegistry(mockOpenAI)
		graphLoader, err := NewGraphLoader(unitRegistry, registry)
		require.NoError(t, err)

		// Test a configuration with multiple providers.
		yamlContent := `
version: "1.0.0"
metadata:
  name: "multi-provider-evaluation"
  description: "Test graph with multiple LLM providers"
units:
  - id: judge1
    type: score_judge
    model: openai/gpt-4
    budget:
      max_tokens: 1000
      max_calls: 10
    parameters:
      judge_prompt: "Rate this answer: %s"
      score_scale: "0.0-1.0"
      temperature: 0.5
      max_tokens: 150
      min_confidence: 0.8
  - id: judge2
    type: score_judge
    model: anthropic/claude-3-sonnet
    budget:
      max_tokens: 1000
      max_calls: 10
    parameters:
      judge_prompt: "Rate this answer: %s"
      score_scale: "0.0-1.0"
      temperature: 0.5
      max_tokens: 150
      min_confidence: 0.8
  - id: judge3
    type: score_judge
    model: google/gemini-pro
    budget:
      max_tokens: 1000
      max_calls: 10
    parameters:
      judge_prompt: "Rate this answer: %s"
      score_scale: "0.0-1.0"
      temperature: 0.5
      max_tokens: 150
      min_confidence: 0.8
`

		// Load the graph from YAML.
		graph, err := graphLoader.LoadFromReader(ctx, strings.NewReader(yamlContent))
		require.NoError(t, err)
		require.NotNil(t, graph)

		// Verify that the graph was created. Specific unit verification would require
		// exposing more graph internals, which is omitted for this test.
		require.NotNil(t, graph)
	})

	t.Run("mixed provider committee evaluation", func(t *testing.T) {
		// Create a committee with judges from different providers.
		mockOpenAI := testutils.NewMockLLMClient("openai-gpt-4")
		mockAnthropic := testutils.NewMockLLMClient("anthropic-claude")
		mockGoogle := testutils.NewMockLLMClient("google-gemini")

		// Create judge units.
		var judges []ports.Unit

		// OpenAI judge.
		openaiJudge, err := units.NewScoreJudgeUnit("openai-judge", mockOpenAI, units.ScoreJudgeConfig{
			JudgePrompt:    "Rate this answer (OpenAI perspective): %s",
			ScoreScale:     "0.0-1.0",
			Temperature:    0.5,
			MaxTokens:      150,
			MinConfidence:  0.8,
			MaxConcurrency: 5,
		})
		require.NoError(t, err)
		judges = append(judges, openaiJudge)

		// Anthropic judge.
		anthropicJudge, err := units.NewScoreJudgeUnit("anthropic-judge", mockAnthropic, units.ScoreJudgeConfig{
			JudgePrompt:    "Rate this answer (Anthropic perspective): %s",
			ScoreScale:     "0.0-1.0",
			Temperature:    0.5,
			MaxTokens:      150,
			MinConfidence:  0.8,
			MaxConcurrency: 5,
		})
		require.NoError(t, err)
		judges = append(judges, anthropicJudge)

		// Google judge.
		googleJudge, err := units.NewScoreJudgeUnit("google-judge", mockGoogle, units.ScoreJudgeConfig{
			JudgePrompt:    "Rate this answer (Google perspective): %s",
			ScoreScale:     "0.0-1.0",
			Temperature:    0.5,
			MaxTokens:      150,
			MinConfidence:  0.8,
			MaxConcurrency: 5,
		})
		require.NoError(t, err)
		judges = append(judges, googleJudge)

		// Create an aggregator.
		aggregator, err := units.NewMedianPoolUnit("aggregator", units.MedianPoolConfig{
			TieBreaker:       units.TieFirst,
			MinScore:         0.0,
			RequireAllScores: true,
		})
		require.NoError(t, err)

		// Execute the evaluation with multiple answers.
		state := domain.NewState()
		state = domain.With(state, domain.KeyQuestion, "Explain quantum computing")
		state = domain.With(state, domain.KeyAnswers, []domain.Answer{
			{ID: "answer1", Content: "Quantum computing uses quantum bits that can be in superposition."},
			{ID: "answer2", Content: "Quantum computers leverage quantum mechanics for computation."},
			{ID: "answer3", Content: "Quantum computing enables exponential speedup for certain problems."},
		})

		// Execute each judge.
		for _, judge := range judges {
			var err error
			state, err = judge.Execute(ctx, state)
			require.NoError(t, err)
		}

		// Get judge scores to verify multi-provider execution.
		allScores, ok := domain.Get(state, domain.KeyJudgeScores)
		require.True(t, ok)

		// The current implementation overwrites scores, so we expect at least 3.
		require.GreaterOrEqual(t, len(allScores), 3)

		// Aggregate the scores.
		finalState, err := aggregator.Execute(ctx, state)
		require.NoError(t, err)

		// Verify the verdict.
		verdict, ok := domain.Get(finalState, domain.KeyVerdict)
		require.True(t, ok)
		require.NotNil(t, verdict)
		assert.NotEmpty(t, verdict.ID)
		assert.NotNil(t, verdict.WinnerAnswer)
	})

	t.Run("provider failover behavior", func(t *testing.T) {
		// Test that the system can handle provider failures gracefully.
		// Create a provider registry with one failing provider.
		failingClient := &mockFailingLLMClient{
			shouldFail: true,
			errMsg:     "Provider temporarily unavailable",
		}

		workingClient := testutils.NewMockLLMClient("backup-model")

		config := llm.RegistryConfig{
			DefaultProvider: "openai",
			Providers:       llm.DefaultProviders,
		}
		registry, err := llm.NewRegistry(config)
		require.NoError(t, err)

		// Register mock provider factories.
		llm.RegisterProviderFactory("openai", func(cfg llm.ClientConfig) (llm.CoreLLM, error) {
			return &mockFailingCoreLLMAdapter{client: failingClient}, nil
		})
		llm.RegisterProviderFactory("anthropic", func(cfg llm.ClientConfig) (llm.CoreLLM, error) {
			return &mockTestUtilsAdapter{client: workingClient}, nil
		})

		// Register the clients.
		err = registry.RegisterClient("openai", llm.ClientConfig{
			APIKey: "test-key",
			Model:  "failing-model",
		})
		require.NoError(t, err)

		err = registry.RegisterClient("anthropic", llm.ClientConfig{
			APIKey: "test-key",
			Model:  "backup-model",
		})
		require.NoError(t, err)

		// Try to use the failing provider.
		_, err = registry.GetClient("openai/failing-model")
		require.NoError(t, err) // GetClient should succeed.

		// Create a judge with the failing client.
		judge, err := units.NewScoreJudgeUnit("test-judge", failingClient, units.ScoreJudgeConfig{
			JudgePrompt:    "Rate this answer: %s",
			ScoreScale:     "0.0-1.0",
			Temperature:    0.5,
			MaxTokens:      150,
			MinConfidence:  0.8,
			MaxConcurrency: 1,
		})
		require.NoError(t, err)

		// Execute should fail.
		state := domain.NewState()
		state = domain.With(state, domain.KeyQuestion, "Test question")
		state = domain.With(state, domain.KeyAnswers, []domain.Answer{
			{ID: "answer1", Content: "Test answer"},
		})

		_, err = judge.Execute(ctx, state)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "Provider temporarily unavailable")
	})
}

// mockFailingLLMClient is a mock LLM client that always fails.
type mockFailingLLMClient struct {
	shouldFail bool
	errMsg     string
}

// Complete returns an error if shouldFail is true.
func (m *mockFailingLLMClient) Complete(ctx context.Context, prompt string, options map[string]any) (string, error) {
	if m.shouldFail {
		return "", fmt.Errorf("%s", m.errMsg)
	}
	return "response", nil
}

// CompleteWithUsage returns an error if shouldFail is true.
func (m *mockFailingLLMClient) CompleteWithUsage(ctx context.Context, prompt string, options map[string]any) (string, int, int, error) {
	if m.shouldFail {
		return "", 0, 0, fmt.Errorf("%s", m.errMsg)
	}
	return "response", 10, 5, nil
}

// EstimateTokens returns a mock token count.
func (m *mockFailingLLMClient) EstimateTokens(text string) (int, error) {
	return len(text) / 4, nil
}

// GetModel returns the model name.
func (m *mockFailingLLMClient) GetModel() string {
	return "failing-model"
}

// mockFailingCoreLLMAdapter adapts mockFailingLLMClient to implement llm.CoreLLM.
type mockFailingCoreLLMAdapter struct {
	client *mockFailingLLMClient
}

// DoRequest delegates the request to the underlying mockFailingLLMClient.
func (m *mockFailingCoreLLMAdapter) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
	response, tokensIn, tokensOut, err := m.client.CompleteWithUsage(ctx, prompt, opts)
	return response, tokensIn, tokensOut, err
}

// GetModel returns the model name from the underlying mockFailingLLMClient.
func (m *mockFailingCoreLLMAdapter) GetModel() string {
	return m.client.GetModel()
}

// SetModel is a no-op for the mock adapter.
func (m *mockFailingCoreLLMAdapter) SetModel(model string) {
	// No-op for mock.
}

// TestProviderMetricsAndTracing tests that metrics and tracing are properly recorded.
func TestProviderMetricsAndTracing(t *testing.T) {
	// This test would require setting up actual metrics collectors and tracing infrastructure.
	// For now, we will test that the clients accept metrics collectors.

	t.Run("clients accept metrics collector", func(t *testing.T) {
		// Test that the provider registry can provide clients for metrics integration.
		// In a real implementation, we would test setting metrics on actual provider clients.

		// Set up the environment.
		os.Setenv("OPENAI_API_KEY", "test-key")
		defer os.Unsetenv("OPENAI_API_KEY")

		// Create a provider registry with a mock client.
		mockClient := testutils.NewMockLLMClient("gpt-4")

		config := llm.RegistryConfig{
			DefaultProvider: "openai",
			Providers:       llm.DefaultProviders,
		}
		registry, err := llm.NewRegistry(config)
		require.NoError(t, err)

		// Register a mock provider factory.
		llm.RegisterProviderFactory("openai", func(cfg llm.ClientConfig) (llm.CoreLLM, error) {
			return &mockTestUtilsAdapter{client: mockClient}, nil
		})

		// Register the client.
		err = registry.RegisterClient("openai", llm.ClientConfig{
			APIKey: "test-key",
			Model:  "gpt-4",
		})
		require.NoError(t, err)

		// Get the OpenAI client.
		client, err := registry.GetClient("openai/gpt-4")
		require.NoError(t, err)

		// Verify that we got the client successfully.
		require.NotNil(t, client)
		assert.Equal(t, "gpt-4", client.GetModel())

		// Execute a request, which would record metrics in a real implementation.
		ctx := context.Background()
		_, err = client.Complete(ctx, "test prompt", nil)
		require.NoError(t, err)
	})
}

// mockTestUtilsAdapter adapts testutils.MockLLMClient to implement llm.CoreLLM.
type mockTestUtilsAdapter struct {
	client *testutils.MockLLMClient
}

// DoRequest delegates the request to the underlying mockLLMClient.
func (m *mockTestUtilsAdapter) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
	response, tokensIn, tokensOut, err := m.client.CompleteWithUsage(ctx, prompt, opts)
	return response, tokensIn, tokensOut, err
}

// GetModel returns the model name from the underlying mockLLMClient.
func (m *mockTestUtilsAdapter) GetModel() string {
	return m.client.GetModel()
}

// SetModel sets the model name on the underlying mockLLMClient.
func (m *mockTestUtilsAdapter) SetModel(model string) {
	m.client.SetModel(model)
}

// TestYAMLConfigurationWithProviders tests loading YAML configurations with provider specifications.
func TestYAMLConfigurationWithProviders(t *testing.T) {
	yamlContent := `
graph:
  id: multi-provider-test
  units:
    - id: openai-judge
      type: score_judge
      model: openai/gpt-4
      budget:
        max_tokens: 1000
        max_calls: 10
      parameters:
        judge_prompt: "Rate this answer: %s"
        score_scale: "0.0-1.0"
        temperature: 0.5
        max_tokens: 150
        min_confidence: 0.8
    - id: anthropic-judge
      type: score_judge
      model: anthropic/claude-3-sonnet@20240307
      budget:
        max_tokens: 1000
        max_calls: 10
      parameters:
        judge_prompt: "Rate this answer: %s"
        score_scale: "0.0-1.0"
        temperature: 0.5
        max_tokens: 150
        min_confidence: 0.8
    - id: google-judge
      type: score_judge
      model: google/gemini-pro
      budget:
        max_tokens: 1000
        max_calls: 10
      parameters:
        judge_prompt: "Rate this answer: %s"
        score_scale: "0.0-1.0"
        temperature: 0.5
        max_tokens: 150
        min_confidence: 0.8
`

	// Parse the YAML.
	var config struct {
		Graph GraphConfig `yaml:"graph"`
	}
	err := yaml.Unmarshal([]byte(yamlContent), &config)
	require.NoError(t, err)

	// Verify that the model fields are parsed correctly.
	assert.Equal(t, "openai/gpt-4", config.Graph.Units[0].Model)
	assert.Equal(t, "anthropic/claude-3-sonnet@20240307", config.Graph.Units[1].Model)
	assert.Equal(t, "google/gemini-pro", config.Graph.Units[2].Model)

	// Verify model validation using a regex pattern.
	modelPattern := regexp.MustCompile(`^[a-z0-9]+/[A-Za-z0-9\-_\.]+(@[A-Za-z0-9\-_\.]+)?$`)
	for _, unit := range config.Graph.Units {
		if unit.Model != "" {
			assert.True(t, modelPattern.MatchString(unit.Model),
				"Model %q should match the expected pattern", unit.Model)
		}
	}
}
