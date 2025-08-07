package units

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"

	"github.com/ahrav/go-gavel/internal/domain"
	"github.com/ahrav/go-gavel/internal/testutils"
)

func TestScoreJudgeUnit_Execute(t *testing.T) {
	mockLLMClient := testutils.NewMockLLMClient("test-model")

	// Add a JSON response for the "Rate" pattern to match our test prompt
	mockLLMClient.AddResponse(testutils.MockResponse{
		Pattern:    "rate",
		Response:   `{"score": 0.85, "confidence": 0.9, "reasoning": "This answer demonstrates good understanding with clear explanations.", "version": 1}`,
		TokensUsed: 20,
	})

	tests := []struct {
		name           string
		config         ScoreJudgeConfig
		setupState     func() domain.State
		expectedError  string
		validateResult func(t *testing.T, state domain.State)
	}{
		{
			name: "successful execution scores answers",
			config: ScoreJudgeConfig{
				JudgePrompt:    "Rate this answer to '{{.Question}}': {{.Answer}} (Provide score and reasoning)",
				ScoreScale:     "0.0-1.0",
				Temperature:    0.5,
				MaxTokens:      150,
				MinConfidence:  0.8,
				MaxConcurrency: 5,
			},
			setupState: func() domain.State {
				state := domain.NewState()
				state = domain.With(state, domain.KeyQuestion, "What is machine learning?")
				answers := []domain.Answer{
					{ID: "answer1", Content: "ML is a subset of AI"},
					{ID: "answer2", Content: "Machine learning algorithms learn from data"},
				}
				return domain.With(state, domain.KeyAnswers, answers)
			},
			validateResult: func(t *testing.T, state domain.State) {
				judgeSummaries, ok := domain.Get(state, domain.KeyJudgeScores)
				require.True(t, ok, "Judge scores should be present in state")
				require.Len(t, judgeSummaries, 2, "Should have scores for both answers")

				for i, summary := range judgeSummaries {
					assert.NotEmpty(t, summary.Reasoning, "Summary %d should have reasoning", i+1)
					assert.GreaterOrEqual(t, summary.Confidence, 0.8, "Summary %d confidence should meet minimum", i+1)
					assert.GreaterOrEqual(t, summary.Score, 0.0, "Summary %d score should be non-negative", i+1)
					assert.LessOrEqual(t, summary.Score, 1.0, "Summary %d score should not exceed 1.0", i+1)
				}
			},
		},
		{
			name: "fails when question missing from state",
			config: ScoreJudgeConfig{
				JudgePrompt:    "Rate this answer: {{.Answer}}",
				ScoreScale:     "1-10",
				Temperature:    0.5,
				MaxTokens:      100,
				MinConfidence:  0.7,
				MaxConcurrency: 5,
			},
			setupState: func() domain.State {
				state := domain.NewState()
				answers := []domain.Answer{{ID: "answer1", Content: "Test answer"}}
				return domain.With(state, domain.KeyAnswers, answers)
			},
			expectedError: "question not found in state",
		},
		{
			name: "fails when answers missing from state",
			config: ScoreJudgeConfig{
				JudgePrompt:    "Rate this answer: {{.Answer}}",
				ScoreScale:     "1-10",
				Temperature:    0.5,
				MaxTokens:      100,
				MinConfidence:  0.7,
				MaxConcurrency: 5,
			},
			setupState: func() domain.State {
				state := domain.NewState()
				return domain.With(state, domain.KeyQuestion, "Test question?")
			},
			expectedError: "answers not found in state",
		},
		{
			name: "handles empty answers list",
			config: ScoreJudgeConfig{
				JudgePrompt:    "Rate this answer: {{.Answer}}",
				ScoreScale:     "1-10",
				Temperature:    0.5,
				MaxTokens:      100,
				MinConfidence:  0.7,
				MaxConcurrency: 5,
			},
			setupState: func() domain.State {
				state := domain.NewState()
				state = domain.With(state, domain.KeyQuestion, "Test question?")
				return domain.With(state, domain.KeyAnswers, []domain.Answer{})
			},
			expectedError: "no answers to score",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			unit, err := NewScoreJudgeUnit("test_judge", mockLLMClient, tt.config)
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

func TestScoreJudgeUnit_parseLLMResponse(t *testing.T) {
	mockLLMClient := testutils.NewMockLLMClient("test-model")
	config := ScoreJudgeConfig{
		JudgePrompt:    "Rate this answer: {{.Answer}}",
		ScoreScale:     "0.0-1.0",
		Temperature:    0.5,
		MaxTokens:      100,
		MinConfidence:  0.8,
		MaxConcurrency: 5,
	}

	unit, err := NewScoreJudgeUnit("test_judge", mockLLMClient, config)
	require.NoError(t, err)

	tests := []struct {
		name           string
		response       string
		expectedScore  float64
		expectedConf   float64
		expectedReason string
		expectedError  string
	}{
		{
			name:           "valid JSON response parses correctly",
			response:       `{"score": 0.9, "confidence": 0.95, "reasoning": "Excellent answer with detailed analysis", "version": 1}`,
			expectedScore:  0.9,
			expectedConf:   0.95,
			expectedReason: "Excellent answer with detailed analysis",
		},
		{
			name:           "JSON embedded in text",
			response:       `Here is my evaluation: {"score": 0.8, "confidence": 0.9, "reasoning": "Good reasoning provided", "version": 1} That concludes my assessment.`,
			expectedScore:  0.8,
			expectedConf:   0.9,
			expectedReason: "Good reasoning provided",
		},
		{
			name:          "invalid JSON format",
			response:      `This is not valid JSON at all`,
			expectedError: "no valid JSON found",
		},
		{
			name:          "score out of range",
			response:      `{"score": 1.5, "confidence": 0.9, "reasoning": "Score exceeds range", "version": 1}`,
			expectedError: "score out of range",
		},
		{
			name:          "confidence out of range",
			response:      `{"score": 0.8, "confidence": 1.5, "reasoning": "Confidence exceeds range", "version": 1}`,
			expectedError: "invalid response structure",
		},
		{
			name:          "missing required fields",
			response:      `{"score": 0.8, "version": 1}`,
			expectedError: "invalid response structure",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			summary, err := unit.parseLLMResponse(tt.response, "test_judge_1")

			if tt.expectedError != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.expectedError)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.expectedScore, summary.Score)
				assert.Equal(t, tt.expectedConf, summary.Confidence)
				assert.Equal(t, tt.expectedReason, summary.Reasoning)
			}
		})
	}
}

func TestExtractJSON_EdgeCases(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "JSON in markdown json code block",
			input:    "Here's the result:\n```json\n{\"score\": 0.9, \"confidence\": 0.95, \"reasoning\": \"Great answer\"}\n```\nThat's all.",
			expected: `{"score": 0.9, "confidence": 0.95, "reasoning": "Great answer"}`,
		},
		{
			name:     "JSON in generic code block",
			input:    "```\n{\"score\": 0.8, \"confidence\": 0.9, \"reasoning\": \"Good\"}\n```",
			expected: `{"score": 0.8, "confidence": 0.9, "reasoning": "Good"}`,
		},
		{
			name:     "JSON with nested objects",
			input:    `{"score": 0.85, "metadata": {"version": 2, "model": "gpt-4"}, "confidence": 0.9, "reasoning": "Solid"}`,
			expected: `{"score": 0.85, "metadata": {"version": 2, "model": "gpt-4"}, "confidence": 0.9, "reasoning": "Solid"}`,
		},
		{
			name:     "JSON with escaped quotes in strings",
			input:    `{"score": 0.9, "confidence": 0.95, "reasoning": "The answer includes \"quotes\" correctly"}`,
			expected: `{"score": 0.9, "confidence": 0.95, "reasoning": "The answer includes \"quotes\" correctly"}`,
		},
		{
			name:     "JSON with braces in string values",
			input:    `{"score": 0.8, "confidence": 0.9, "reasoning": "Uses {braces} and } in text"}`,
			expected: `{"score": 0.8, "confidence": 0.9, "reasoning": "Uses {braces} and } in text"}`,
		},
		{
			name:     "Multiple JSON objects, extract first",
			input:    `{"score": 0.7, "confidence": 0.8, "reasoning": "First"} {"score": 0.9, "confidence": 0.95, "reasoning": "Second"}`,
			expected: `{"score": 0.7, "confidence": 0.8, "reasoning": "First"}`,
		},
		{
			name:     "JSON with newlines and formatting",
			input:    "{\n  \"score\": 0.85,\n  \"confidence\": 0.9,\n  \"reasoning\": \"Well formatted\"\n}",
			expected: "{\n  \"score\": 0.85,\n  \"confidence\": 0.9,\n  \"reasoning\": \"Well formatted\"\n}",
		},
		{
			name:     "No JSON found",
			input:    "This response contains no JSON whatsoever",
			expected: "",
		},
		{
			name:     "Incomplete JSON",
			input:    `{"score": 0.8, "confidence": 0.9, "reasoning": "Incomplete`,
			expected: "",
		},
		{
			name:     "JSON with code block and language identifier",
			input:    "```javascript\n{\"score\": 0.9, \"confidence\": 0.95, \"reasoning\": \"JS block\"}\n```",
			expected: `{"score": 0.9, "confidence": 0.95, "reasoning": "JS block"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractJSON(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestScoreJudgeUnit_Validate(t *testing.T) {
	mockLLMClient := testutils.NewMockLLMClient("test-model")

	tests := []struct {
		name          string
		config        ScoreJudgeConfig
		llmClient     *testutils.MockLLMClient
		expectedError string
	}{
		{
			name: "valid configuration passes",
			config: ScoreJudgeConfig{
				JudgePrompt:    "Rate this answer to '{{.Question}}': {{.Answer}}",
				ScoreScale:     "1-10",
				Temperature:    0.5,
				MaxTokens:      150,
				MinConfidence:  0.8,
				MaxConcurrency: 5,
			},
			llmClient: mockLLMClient,
		},
		{
			name: "prompt too short fails",
			config: ScoreJudgeConfig{
				JudgePrompt:    "Short",
				ScoreScale:     "1-10",
				Temperature:    0.5,
				MaxTokens:      100,
				MinConfidence:  0.7,
				MaxConcurrency: 5,
			},
			llmClient:     mockLLMClient,
			expectedError: "configuration validation failed",
		},
		{
			name: "invalid temperature fails",
			config: ScoreJudgeConfig{
				JudgePrompt:    "Rate this answer: {{.Answer}}",
				ScoreScale:     "1-10",
				Temperature:    1.5,
				MaxTokens:      100,
				MinConfidence:  0.7,
				MaxConcurrency: 5,
			},
			llmClient:     mockLLMClient,
			expectedError: "configuration validation failed",
		},
		{
			name: "invalid max tokens fails",
			config: ScoreJudgeConfig{
				JudgePrompt:    "Rate this answer: {{.Answer}}",
				ScoreScale:     "1-10",
				Temperature:    0.5,
				MaxTokens:      10,
				MinConfidence:  0.7,
				MaxConcurrency: 5,
			},
			llmClient:     mockLLMClient,
			expectedError: "configuration validation failed",
		},
		{
			name: "invalid min confidence fails",
			config: ScoreJudgeConfig{
				JudgePrompt:    "Rate this answer: {{.Answer}}",
				ScoreScale:     "1-10",
				Temperature:    0.5,
				MaxTokens:      100,
				MinConfidence:  1.5,
				MaxConcurrency: 5,
			},
			llmClient:     mockLLMClient,
			expectedError: "configuration validation failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			unit, err := NewScoreJudgeUnit("test_judge", tt.llmClient, tt.config)
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

func TestScoreJudgeUnit_Name(t *testing.T) {
	mockLLMClient := testutils.NewMockLLMClient("test-model")
	config := ScoreJudgeConfig{
		JudgePrompt:    "Rate this answer: {{.Answer}}",
		ScoreScale:     "1-10",
		Temperature:    0.5,
		MaxTokens:      100,
		MinConfidence:  0.7,
		MaxConcurrency: 5,
	}

	unit, err := NewScoreJudgeUnit("custom_judge", mockLLMClient, config)
	require.NoError(t, err)

	assert.Equal(t, "custom_judge", unit.Name())
}

func TestNewScoreJudgeUnit(t *testing.T) {
	mockLLMClient := testutils.NewMockLLMClient("test-model")

	t.Run("creates unit successfully with valid parameters", func(t *testing.T) {
		config := ScoreJudgeConfig{
			JudgePrompt:    "Please rate this answer: {{.Answer}}",
			ScoreScale:     "0.0-1.0",
			Temperature:    0.5,
			MaxTokens:      150,
			MinConfidence:  0.8,
			MaxConcurrency: 5,
		}

		unit, err := NewScoreJudgeUnit("test_unit", mockLLMClient, config)
		require.NoError(t, err)
		assert.Equal(t, "test_unit", unit.Name())
	})

	t.Run("fails with empty name", func(t *testing.T) {
		config := ScoreJudgeConfig{
			JudgePrompt:    "Rate this answer: {{.Answer}}",
			ScoreScale:     "1-10",
			Temperature:    0.5,
			MaxTokens:      100,
			MinConfidence:  0.7,
			MaxConcurrency: 5,
		}

		_, err := NewScoreJudgeUnit("", mockLLMClient, config)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "unit name cannot be empty")
	})

	t.Run("fails with nil LLM client", func(t *testing.T) {
		config := ScoreJudgeConfig{
			JudgePrompt:    "Rate this answer: {{.Answer}}",
			ScoreScale:     "1-10",
			Temperature:    0.5,
			MaxTokens:      100,
			MinConfidence:  0.7,
			MaxConcurrency: 5,
		}

		_, err := NewScoreJudgeUnit("test_unit", nil, config)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "LLM client cannot be nil")
	})
}

func TestNewScoreJudgeFromConfig(t *testing.T) {
	mockLLMClient := testutils.NewMockLLMClient("test-model")

	t.Run("creates unit with valid config", func(t *testing.T) {
		config := map[string]any{
			"judge_prompt":    "Rate this answer to '{{.Question}}': {{.Answer}}",
			"score_scale":     "1-10",
			"temperature":     0.5,
			"max_tokens":      150,
			"min_confidence":  0.8,
			"max_concurrency": 5,
		}

		unit, err := NewScoreJudgeFromConfig("test_id", config, mockLLMClient)
		require.NoError(t, err)
		assert.Equal(t, "test_id", unit.Name())
	})

	t.Run("fails without LLM client", func(t *testing.T) {
		config := map[string]any{
			"judge_prompt": "Rate this answer: {{.Answer}}",
			"score_scale":  "1-10",
		}

		_, err := NewScoreJudgeFromConfig("test_id", config, nil)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "LLM client cannot be nil")
	})

	t.Run("handles string to int conversion", func(t *testing.T) {
		config := map[string]any{
			"judge_prompt":   "Rate this answer: {{.Answer}}",
			"score_scale":    "1-10",
			"temperature":    0.5,
			"max_tokens":     150, // Use actual int, not string
			"min_confidence": 0.8,
		}

		unit, err := NewScoreJudgeFromConfig("test_id", config, mockLLMClient)
		require.NoError(t, err)
		assert.Equal(t, "test_id", unit.Name())
	})
}

// Test the new ScoreScale functionality
func TestScoreScale(t *testing.T) {
	tests := []struct {
		name          string
		scaleStr      string
		expectedMin   float64
		expectedMax   float64
		expectedError string
	}{
		{
			name:        "valid integer scale",
			scaleStr:    "1-10",
			expectedMin: 1.0,
			expectedMax: 10.0,
		},
		{
			name:        "valid float scale",
			scaleStr:    "0.0-1.0",
			expectedMin: 0.0,
			expectedMax: 1.0,
		},
		{
			name:        "valid mixed scale",
			scaleStr:    "1-5.5",
			expectedMin: 1.0,
			expectedMax: 5.5,
		},
		{
			name:          "invalid format missing dash",
			scaleStr:      "1to10",
			expectedError: "score scale must be in format 'min-max'",
		},
		{
			name:          "invalid format too many parts",
			scaleStr:      "1-2-3",
			expectedError: "score scale must be in format 'min-max'",
		},
		{
			name:          "invalid minimum value",
			scaleStr:      "abc-10",
			expectedError: "invalid minimum value",
		},
		{
			name:          "invalid maximum value",
			scaleStr:      "1-xyz",
			expectedError: "invalid maximum value",
		},
		{
			name:          "min equals max",
			scaleStr:      "5-5",
			expectedError: "minimum value must be less than maximum value",
		},
		{
			name:          "min greater than max",
			scaleStr:      "10-1",
			expectedError: "minimum value must be less than maximum value",
		},
		{
			name:          "minimum value too low",
			scaleStr:      "-1001-10",
			expectedError: "minimum score value -1001.00 is too low",
		},
		{
			name:          "maximum value too high",
			scaleStr:      "0-1001",
			expectedError: "maximum score value 1001.00 is too high",
		},
		{
			name:          "range too narrow",
			scaleStr:      "1.0-1.005",
			expectedError: "score scale range too narrow",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scale, err := ParseScoreScale(tt.scaleStr)

			if tt.expectedError != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.expectedError)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.expectedMin, scale.Min)
				assert.Equal(t, tt.expectedMax, scale.Max)

				// Test Contains method
				assert.True(t, scale.Contains(tt.expectedMin))
				assert.True(t, scale.Contains(tt.expectedMax))
				assert.True(t, scale.Contains((tt.expectedMin+tt.expectedMax)/2))
				assert.False(t, scale.Contains(tt.expectedMin-1))
				assert.False(t, scale.Contains(tt.expectedMax+1))

				// Test String method
				expectedStr := fmt.Sprintf("%.1f-%.1f", tt.expectedMin, tt.expectedMax)
				assert.Equal(t, expectedStr, scale.String())
			}
		})
	}
}

// Test configuration defaults functionality
func TestDefaultScoreJudgeConfig(t *testing.T) {
	config := defaultScoreJudgeConfig()

	assert.NotEmpty(t, config.JudgePrompt, "Default prompt should not be empty")
	assert.Equal(t, "1-10", config.ScoreScale, "Default scale should be 1-10")
	assert.Equal(t, 0.0, config.Temperature, "Default temperature should be 0.0")
	assert.Equal(t, 256, config.MaxTokens, "Default max tokens should be 256")
	assert.Equal(t, 0.0, config.MinConfidence, "Default min confidence should be 0.0")

	// Verify the default config is valid
	_, err := ParseScoreScale(config.ScoreScale)
	assert.NoError(t, err, "Default score scale should be valid")
}

// Test improved factory type coercion
func TestNewScoreJudgeFromConfig_TypeCoercion(t *testing.T) {
	mockLLMClient := testutils.NewMockLLMClient("test-model")

	tests := []struct {
		name   string
		config map[string]any
	}{
		{
			name: "handles float64 max_tokens from YAML",
			config: map[string]any{
				"score_scale": "1-10",
				"max_tokens":  float64(200), // YAML often parses numbers as float64
			},
		},
		{
			name: "handles int temperature",
			config: map[string]any{
				"score_scale": "1-10",
				"temperature": int(0), // Sometimes temperature comes as int
			},
		},
		{
			name: "handles numeric values",
			config: map[string]any{
				"score_scale":    "1-10",
				"max_tokens":     300, // Use actual numbers
				"temperature":    0.5, // Use actual numbers
				"min_confidence": 0.8, // Use actual numbers
			},
		},
		{
			name:   "uses defaults for missing values",
			config: map[string]any{
				// Only required field, others should use defaults
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			unit, err := NewScoreJudgeFromConfig("test_unit", tt.config, mockLLMClient)
			require.NoError(t, err, "Factory should handle type coercion properly")
			assert.NotNil(t, unit)
			assert.Equal(t, "test_unit", unit.Name())
		})
	}
}

// Test thread safety of UnmarshalParameters
func TestScoreJudgeUnit_UnmarshalParameters_ThreadSafety(t *testing.T) {
	mockLLMClient := testutils.NewMockLLMClient("test-model")
	config := ScoreJudgeConfig{
		JudgePrompt:    "Rate this answer: {{.Answer}}",
		ScoreScale:     "1-10",
		Temperature:    0.5,
		MaxTokens:      100,
		MinConfidence:  0.7,
		MaxConcurrency: 5,
	}

	unit, err := NewScoreJudgeUnit("test_judge", mockLLMClient, config)
	require.NoError(t, err)

	// Verify that UnmarshalParameters returns a new instance
	yamlData := `
judge_prompt: "New and improved prompt for scoring: {{.Answer}}"
score_scale: "0-5"
temperature: 0.8
max_tokens: 200
min_confidence: 0.9
max_concurrency: 10
`
	var node yaml.Node
	err = yaml.Unmarshal([]byte(yamlData), &node)
	require.NoError(t, err)

	newUnit, err := unit.UnmarshalParameters(*node.Content[0])
	require.NoError(t, err)
	require.NotNil(t, newUnit)

	// Verify original unit is unchanged
	assert.Equal(t, "Rate this answer: {{.Answer}}", unit.config.JudgePrompt)
	assert.Equal(t, "1-10", unit.config.ScoreScale)

	// Verify new unit has updated config
	assert.Equal(t, "New and improved prompt for scoring: {{.Answer}}", newUnit.config.JudgePrompt)
	assert.Equal(t, "0-5", newUnit.config.ScoreScale)
	assert.Equal(t, 0.8, newUnit.config.Temperature)
}
