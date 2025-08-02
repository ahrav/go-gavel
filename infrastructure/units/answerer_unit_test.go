package units

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"

	"github.com/ahrav/go-gavel/internal/domain"
	"github.com/ahrav/go-gavel/internal/testutils"
)

func TestAnswererUnit_Execute(t *testing.T) {
	mockLLMClient := testutils.NewMockLLMClient("test-model")

	tests := []struct {
		name           string
		config         AnswererConfig
		setupState     func() domain.State
		expectedError  string
		validateResult func(t *testing.T, state domain.State)
	}{
		{
			name: "successful execution generates answers",
			config: AnswererConfig{
				NumAnswers:     2,
				Prompt:         "Please provide a comprehensive answer to: %s",
				Temperature:    0.7,
				MaxTokens:      200,
				Timeout:        30 * time.Second,
				MaxConcurrency: 5,
			},
			setupState: func() domain.State {
				state := domain.NewState()
				return domain.With(state, domain.KeyQuestion, "What is artificial intelligence?")
			},
			validateResult: func(t *testing.T, state domain.State) {
				answers, ok := domain.Get(state, domain.KeyAnswers)
				require.True(t, ok, "Answers should be present in state")
				require.Len(t, answers, 2, "Should generate 2 answers as configured")

				for i, answer := range answers {
					assert.NotEmpty(t, answer.ID, "Answer %d should have non-empty ID", i+1)
					assert.NotEmpty(t, answer.Content, "Answer %d should have non-empty content", i+1)
					assert.Contains(t, answer.ID, "test_answerer_answer_", "Answer ID should follow expected pattern")
				}
			},
		},
		{
			name: "fails when question missing from state",
			config: AnswererConfig{
				NumAnswers:     1,
				Prompt:         "Answer: %s",
				Temperature:    0.5,
				MaxTokens:      100,
				Timeout:        30 * time.Second,
				MaxConcurrency: 5,
			},
			setupState: func() domain.State {
				return domain.NewState() // No question
			},
			expectedError: "question not found in state",
		},
		{
			name: "fails when question is empty",
			config: AnswererConfig{
				NumAnswers:     1,
				Prompt:         "Answer: %s",
				Temperature:    0.5,
				MaxTokens:      100,
				Timeout:        30 * time.Second,
				MaxConcurrency: 5,
			},
			setupState: func() domain.State {
				state := domain.NewState()
				return domain.With(state, domain.KeyQuestion, "")
			},
			expectedError: "question cannot be empty",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			unit, err := NewAnswererUnit("test_answerer", mockLLMClient, tt.config)
			require.NoError(t, err)

			state := tt.setupState()
			ctx := context.Background()

			result, err := unit.Execute(ctx, state)

			if tt.expectedError != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.expectedError)
			} else {
				require.NoError(t, err)
				if tt.validateResult != nil {
					tt.validateResult(t, result)
				}
			}
		})
	}
}

func TestAnswererUnit_Validate(t *testing.T) {
	mockLLMClient := testutils.NewMockLLMClient("test-model")

	tests := []struct {
		name          string
		config        AnswererConfig
		llmClient     *testutils.MockLLMClient
		expectedError string
	}{
		{
			name: "valid configuration passes",
			config: AnswererConfig{
				NumAnswers:     3,
				Prompt:         "Please provide a detailed answer to: %s",
				Temperature:    0.7,
				MaxTokens:      200,
				Timeout:        30 * time.Second,
				MaxConcurrency: 5,
			},
			llmClient: mockLLMClient,
		},
		{
			name: "invalid number of answers fails",
			config: AnswererConfig{
				NumAnswers:     0,
				Prompt:         "Answer: %s",
				Temperature:    0.5,
				MaxTokens:      100,
				Timeout:        30 * time.Second,
				MaxConcurrency: 5,
			},
			llmClient:     mockLLMClient,
			expectedError: "configuration validation failed",
		},
		{
			name: "prompt too short fails",
			config: AnswererConfig{
				NumAnswers:     1,
				Prompt:         "Short",
				Temperature:    0.5,
				MaxTokens:      100,
				Timeout:        30 * time.Second,
				MaxConcurrency: 5,
			},
			llmClient:     mockLLMClient,
			expectedError: "configuration validation failed",
		},
		{
			name: "invalid temperature fails",
			config: AnswererConfig{
				NumAnswers:     1,
				Prompt:         "Answer the question: %s",
				Temperature:    1.5,
				MaxTokens:      100,
				Timeout:        30 * time.Second,
				MaxConcurrency: 5,
			},
			llmClient:     mockLLMClient,
			expectedError: "configuration validation failed",
		},
		{
			name: "invalid max tokens fails",
			config: AnswererConfig{
				NumAnswers:     1,
				Prompt:         "Answer the question: %s",
				Temperature:    0.5,
				MaxTokens:      5,
				Timeout:        30 * time.Second,
				MaxConcurrency: 5,
			},
			llmClient:     mockLLMClient,
			expectedError: "configuration validation failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			unit, err := NewAnswererUnit("test_answerer", tt.llmClient, tt.config)
			if tt.expectedError != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.expectedError)
			} else {
				require.NoError(t, err)
				assert.NoError(t, unit.Validate())
			}
		})
	}
}

func TestAnswererUnit_Name(t *testing.T) {
	mockLLMClient := testutils.NewMockLLMClient("test-model")
	config := AnswererConfig{
		NumAnswers:     1,
		Prompt:         "Answer the question: %s",
		Temperature:    0.5,
		MaxTokens:      100,
		Timeout:        30 * time.Second,
		MaxConcurrency: 5,
	}

	unit, err := NewAnswererUnit("custom_answerer", mockLLMClient, config)
	require.NoError(t, err)

	assert.Equal(t, "custom_answerer", unit.Name())
}

func TestNewAnswererUnit(t *testing.T) {
	mockLLMClient := testutils.NewMockLLMClient("test-model")

	t.Run("creates unit successfully with valid parameters", func(t *testing.T) {
		config := AnswererConfig{
			NumAnswers:     2,
			Prompt:         "Please answer: %s",
			Temperature:    0.7,
			MaxTokens:      150,
			Timeout:        30 * time.Second,
			MaxConcurrency: 5,
		}

		unit, err := NewAnswererUnit("test_unit", mockLLMClient, config)
		require.NoError(t, err)
		assert.Equal(t, "test_unit", unit.Name())
	})

	t.Run("fails with empty name", func(t *testing.T) {
		config := AnswererConfig{
			NumAnswers:     1,
			Prompt:         "Answer: %s",
			Temperature:    0.5,
			MaxTokens:      100,
			Timeout:        30 * time.Second,
			MaxConcurrency: 5,
		}

		_, err := NewAnswererUnit("", mockLLMClient, config)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "unit name cannot be empty")
	})

	t.Run("fails with nil LLM client", func(t *testing.T) {
		config := AnswererConfig{
			NumAnswers:     1,
			Prompt:         "Answer: %s",
			Temperature:    0.5,
			MaxTokens:      100,
			Timeout:        30 * time.Second,
			MaxConcurrency: 5,
		}

		_, err := NewAnswererUnit("test_unit", nil, config)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "LLM client cannot be nil")
	})
}

func TestAnswererUnit_UnmarshalParameters(t *testing.T) {
	mockLLMClient := testutils.NewMockLLMClient("test-model")
	config := AnswererConfig{
		NumAnswers:     2,
		Prompt:         "Answer: {{.Question}}",
		Temperature:    0.5,
		MaxTokens:      100,
		Timeout:        30 * time.Second,
		MaxConcurrency: 5,
	}

	unit, err := NewAnswererUnit("test_answerer", mockLLMClient, config)
	require.NoError(t, err)

	t.Run("successfully updates configuration", func(t *testing.T) {
		yamlData := `
num_answers: 4
prompt: "New prompt: {{.Question}}"
temperature: 0.8
max_tokens: 200
timeout: 60s
max_concurrency: 10
`
		var node yaml.Node
		err := yaml.Unmarshal([]byte(yamlData), &node)
		require.NoError(t, err)

		newUnit, err := unit.UnmarshalParameters(*node.Content[0])
		require.NoError(t, err)
		require.NotNil(t, newUnit)

		// Verify original unit is unchanged
		assert.Equal(t, 2, unit.config.NumAnswers)
		assert.Equal(t, "Answer: {{.Question}}", unit.config.Prompt)
		assert.Equal(t, 0.5, unit.config.Temperature)

		// Verify new unit has updated config
		assert.Equal(t, 4, newUnit.config.NumAnswers)
		assert.Equal(t, "New prompt: {{.Question}}", newUnit.config.Prompt)
		assert.Equal(t, 0.8, newUnit.config.Temperature)
		assert.Equal(t, 200, newUnit.config.MaxTokens)
		assert.Equal(t, 60*time.Second, newUnit.config.Timeout)
		assert.Equal(t, 10, newUnit.config.MaxConcurrency)
	})

	t.Run("fails with invalid YAML", func(t *testing.T) {
		yamlData := `
num_answers: not_a_number
prompt: "Test"
`
		var node yaml.Node
		err := yaml.Unmarshal([]byte(yamlData), &node)
		require.NoError(t, err)

		_, err = unit.UnmarshalParameters(*node.Content[0])
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to decode parameters")
	})

	t.Run("fails with invalid configuration", func(t *testing.T) {
		yamlData := `
num_answers: 0
prompt: "short"
temperature: 2.0
max_tokens: 5
timeout: 0s
`
		var node yaml.Node
		err := yaml.Unmarshal([]byte(yamlData), &node)
		require.NoError(t, err)

		_, err = unit.UnmarshalParameters(*node.Content[0])
		require.Error(t, err)
		assert.Contains(t, err.Error(), "configuration validation failed")
	})

	t.Run("returns new instance maintaining thread safety", func(t *testing.T) {
		yamlData := `
num_answers: 3
prompt: "Answer this: {{.Question}}"
temperature: 0.6
max_tokens: 150
timeout: 45s
max_concurrency: 7
`
		var node yaml.Node
		err := yaml.Unmarshal([]byte(yamlData), &node)
		require.NoError(t, err)

		newUnit1, err := unit.UnmarshalParameters(*node.Content[0])
		require.NoError(t, err)

		newUnit2, err := unit.UnmarshalParameters(*node.Content[0])
		require.NoError(t, err)

		// Verify they are different instances
		assert.NotSame(t, newUnit1, newUnit2)
		assert.NotSame(t, unit, newUnit1)
		assert.NotSame(t, unit, newUnit2)

		// But have the same configuration
		assert.Equal(t, newUnit1.config.NumAnswers, newUnit2.config.NumAnswers)
		assert.Equal(t, newUnit1.config.Prompt, newUnit2.config.Prompt)
	})
}

func TestCreateAnswererUnit(t *testing.T) {
	mockLLMClient := testutils.NewMockLLMClient("test-model")

	t.Run("creates unit with valid config", func(t *testing.T) {
		config := map[string]any{
			"llm_client":      mockLLMClient,
			"num_answers":     3,
			"prompt":          "Answer the question: {{.Question}}",
			"temperature":     0.7,
			"max_tokens":      200,
			"timeout":         "30s",
			"max_concurrency": 5,
		}

		unit, err := CreateAnswererUnit("test_id", config)
		require.NoError(t, err)
		assert.Equal(t, "test_id", unit.Name())
		assert.Equal(t, 3, unit.config.NumAnswers)
		assert.Equal(t, "Answer the question: {{.Question}}", unit.config.Prompt)
	})

	t.Run("fails without LLM client", func(t *testing.T) {
		config := map[string]any{
			"num_answers": 2,
			"prompt":      "Answer: {{.Question}}",
		}

		_, err := CreateAnswererUnit("test_id", config)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "llm_client is required")
	})

	t.Run("handles type conversions", func(t *testing.T) {
		config := map[string]any{
			"llm_client":      mockLLMClient,
			"num_answers":     "3",          // string
			"temperature":     0,            // int
			"max_tokens":      float64(250), // float64
			"timeout":         30.0,         // float64 as seconds
			"max_concurrency": "8",          // string
		}

		unit, err := CreateAnswererUnit("test_id", config)
		require.NoError(t, err)
		assert.Equal(t, 3, unit.config.NumAnswers)
		assert.Equal(t, 0.0, unit.config.Temperature)
		assert.Equal(t, 250, unit.config.MaxTokens)
		assert.Equal(t, 30*time.Second, unit.config.Timeout)
		assert.Equal(t, 8, unit.config.MaxConcurrency)
	})

	t.Run("uses defaults for missing values", func(t *testing.T) {
		config := map[string]any{
			"llm_client": mockLLMClient,
		}

		unit, err := CreateAnswererUnit("test_id", config)
		require.NoError(t, err)
		assert.Equal(t, 3, unit.config.NumAnswers)              // default
		assert.Equal(t, 0.7, unit.config.Temperature)           // default
		assert.Equal(t, 500, unit.config.MaxTokens)             // default
		assert.Equal(t, 30*time.Second, unit.config.Timeout)    // default
		assert.Equal(t, 5, unit.config.MaxConcurrency)          // default
		assert.Contains(t, unit.config.Prompt, "comprehensive") // default prompt
	})
}

func TestAnswererUnit_ContextCancellation(t *testing.T) {
	mockLLMClient := testutils.NewMockLLMClient("test-model")
	config := AnswererConfig{
		NumAnswers:     3,
		Prompt:         "Answer: %s",
		Temperature:    0.5,
		MaxTokens:      100,
		Timeout:        30 * time.Second,
		MaxConcurrency: 2,
	}

	unit, err := NewAnswererUnit("test_unit", mockLLMClient, config)
	require.NoError(t, err)

	t.Run("cancels execution when context cancelled", func(t *testing.T) {
		state := domain.NewState()
		state = domain.With(state, domain.KeyQuestion, "What is the meaning of life?")

		// Create a context that we'll cancel immediately
		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately

		_, err := unit.Execute(ctx, state)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "context canceled")
	})

	t.Run("respects timeout from config", func(t *testing.T) {
		// Create a unit with minimum allowed timeout
		shortTimeoutConfig := config
		shortTimeoutConfig.Timeout = 1 * time.Second

		shortUnit, err := NewAnswererUnit("timeout_test", mockLLMClient, shortTimeoutConfig)
		require.NoError(t, err)

		// Verify the timeout is properly configured
		assert.Equal(t, 1*time.Second, shortUnit.config.Timeout, "Unit should have configured timeout")

		// Note: We can't test actual timeout behavior with a mock that returns instantly
		// and a minimum timeout of 1s. This test verifies configuration is accepted.
	})
}
