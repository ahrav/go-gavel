package units

import (
	"context"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"

	"github.com/ahrav/go-gavel/internal/domain"
)

func TestArithmeticMeanUnit_Aggregate(t *testing.T) {
	tests := []struct {
		name             string
		config           ArithmeticMeanConfig
		scores           []float64
		candidates       []domain.Answer
		expectedWinnerID string
		expectedScore    float64
		expectedError    string
	}{
		{
			name: "calculates arithmetic mean with highest score winner",
			config: ArithmeticMeanConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			scores: []float64{0.7, 0.9, 0.8}, // mean = 2.4/3 = 0.8
			candidates: []domain.Answer{
				{ID: "answer1", Content: "First answer"},
				{ID: "answer2", Content: "Second answer"},
				{ID: "answer3", Content: "Third answer"},
			},
			expectedWinnerID: "answer2", // highest individual score
			expectedScore:    0.8,       // arithmetic mean
		},
		{
			name: "handles equal scores with first tie breaker and returns mean",
			config: ArithmeticMeanConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			scores: []float64{0.8, 0.8, 0.7}, // mean = 2.3/3 ≈ 0.7667
			candidates: []domain.Answer{
				{ID: "answer1", Content: "First answer"},
				{ID: "answer2", Content: "Second answer"},
				{ID: "answer3", Content: "Third answer"},
			},
			expectedWinnerID: "answer1",          // first of tied winners
			expectedScore:    0.7666666666666667, // arithmetic mean
		},
		{
			name: "fails with tie breaker error",
			config: ArithmeticMeanConfig{
				TieBreaker:       "error",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			scores:        []float64{0.8, 0.8, 0.7},
			candidates:    []domain.Answer{{ID: "a1"}, {ID: "a2"}, {ID: "a3"}},
			expectedError: "multiple answers tied with highest score",
		},
		{
			name: "enforces minimum score requirement against mean",
			config: ArithmeticMeanConfig{
				TieBreaker:       "first",
				MinScore:         0.8,
				RequireAllScores: true,
			},
			scores:        []float64{0.8, 0.7, 0.85}, // mean = 2.35/3 ≈ 0.783 < 0.8
			candidates:    []domain.Answer{{ID: "a1"}, {ID: "a2"}, {ID: "a3"}},
			expectedError: "mean=0.783, minimum=0.800",
		},
		{
			name: "handles empty scores",
			config: ArithmeticMeanConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			scores:        []float64{},
			candidates:    []domain.Answer{},
			expectedError: "no scores provided for aggregation",
		},
		{
			name: "validates score-candidate length mismatch",
			config: ArithmeticMeanConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			scores:        []float64{0.8, 0.9},
			candidates:    []domain.Answer{{ID: "a1"}},
			expectedError: "scores and candidates length mismatch",
		},
		{
			name: "rejects NaN scores",
			config: ArithmeticMeanConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			scores:        []float64{0.8, math.NaN(), 0.9},
			candidates:    []domain.Answer{{ID: "a1"}, {ID: "a2"}, {ID: "a3"}},
			expectedError: "invalid score at index 1",
		},
		{
			name: "rejects infinite scores",
			config: ArithmeticMeanConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			scores:        []float64{0.8, math.Inf(1), 0.9},
			candidates:    []domain.Answer{{ID: "a1"}, {ID: "a2"}, {ID: "a3"}},
			expectedError: "invalid score at index 1",
		},
		{
			name: "single score returns itself as mean",
			config: ArithmeticMeanConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			scores:           []float64{0.75},
			candidates:       []domain.Answer{{ID: "single", Content: "Only answer"}},
			expectedWinnerID: "single",
			expectedScore:    0.75, // mean of single value is itself
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			unit, err := NewArithmeticMeanUnit("test_arithmetic_mean", tt.config)
			require.NoError(t, err)

			winner, score, err := unit.Aggregate(tt.scores, tt.candidates)

			if tt.expectedError != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.expectedError)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.expectedWinnerID, winner.ID)
				assert.InDelta(t, tt.expectedScore, score, 0.0001) // Allow for floating point precision
			}
		})
	}
}

func TestArithmeticMeanUnit_Execute(t *testing.T) {
	tests := []struct {
		name           string
		config         ArithmeticMeanConfig
		setupState     func() domain.State
		expectedError  string
		validateResult func(t *testing.T, state domain.State)
	}{
		{
			name: "successful execution with valid data",
			config: ArithmeticMeanConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			setupState: func() domain.State {
				state := domain.NewState()
				answers := []domain.Answer{
					{ID: "answer1", Content: "First answer"},
					{ID: "answer2", Content: "Second answer"},
				}
				judgeSummaries := []domain.JudgeSummary{
					{Score: 0.8, Reasoning: "Good answer", Confidence: 0.9},
					{Score: 0.9, Reasoning: "Better answer", Confidence: 0.95},
				}
				state = domain.With(state, domain.KeyAnswers, answers)
				state = domain.With(state, domain.KeyJudgeScores, judgeSummaries)
				return state
			},
			validateResult: func(t *testing.T, state domain.State) {
				verdict, ok := domain.Get(state, domain.KeyVerdict)
				require.True(t, ok, "Verdict should be present in state")
				require.NotNil(t, verdict, "Verdict should not be nil")

				assert.Equal(t, "answer2", verdict.WinnerAnswer.ID)
				assert.InDelta(t, 0.85, verdict.AggregateScore, 0.0001) // mean = (0.8 + 0.9) / 2 = 0.85
				assert.Contains(t, verdict.ID, "test_arithmetic_mean_verdict")
			},
		},
		{
			name: "fails when answers missing from state",
			config: ArithmeticMeanConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			setupState: func() domain.State {
				state := domain.NewState()
				// Missing answers
				judgeSummaries := []domain.JudgeSummary{
					{Score: 0.8, Reasoning: "Good", Confidence: 0.9},
				}
				state = domain.With(state, domain.KeyJudgeScores, judgeSummaries)
				return state
			},
			expectedError: "answers not found in state",
		},
		{
			name: "fails when judge scores missing from state",
			config: ArithmeticMeanConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			setupState: func() domain.State {
				state := domain.NewState()
				answers := []domain.Answer{
					{ID: "answer1", Content: "First answer"},
				}
				state = domain.With(state, domain.KeyAnswers, answers)
				// Missing judge scores
				return state
			},
			expectedError: "judge scores not found in state",
		},
		{
			name: "handles length mismatch when RequireAllScores is false",
			config: ArithmeticMeanConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: false,
			},
			setupState: func() domain.State {
				state := domain.NewState()
				answers := []domain.Answer{
					{ID: "answer1", Content: "First answer"},
					{ID: "answer2", Content: "Second answer"},
					{ID: "answer3", Content: "Third answer"},
				}
				judgeSummaries := []domain.JudgeSummary{
					{Score: 0.8, Reasoning: "Good", Confidence: 0.9},
					{Score: 0.9, Reasoning: "Better", Confidence: 0.95},
				}
				state = domain.With(state, domain.KeyAnswers, answers)
				state = domain.With(state, domain.KeyJudgeScores, judgeSummaries)
				return state
			},
			validateResult: func(t *testing.T, state domain.State) {
				verdict, ok := domain.Get(state, domain.KeyVerdict)
				require.True(t, ok)
				require.NotNil(t, verdict)

				// Should work with truncated data (first 2 answers and scores).
				assert.Equal(t, "answer2", verdict.WinnerAnswer.ID)
				assert.InDelta(t, 0.85, verdict.AggregateScore, 0.0001) // mean = (0.8 + 0.9) / 2 = 0.85
			},
		},
		{
			name: "fails length mismatch when RequireAllScores is true",
			config: ArithmeticMeanConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			setupState: func() domain.State {
				state := domain.NewState()
				answers := []domain.Answer{
					{ID: "answer1", Content: "First answer"},
					{ID: "answer2", Content: "Second answer"},
				}
				judgeSummaries := []domain.JudgeSummary{
					{Score: 0.8, Reasoning: "Good", Confidence: 0.9},
				}
				state = domain.With(state, domain.KeyAnswers, answers)
				state = domain.With(state, domain.KeyJudgeScores, judgeSummaries)
				return state
			},
			expectedError: "mismatch between answers (2) and judge scores (1)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			unit, err := NewArithmeticMeanUnit("test_arithmetic_mean", tt.config)
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

func TestArithmeticMeanUnit_Validate(t *testing.T) {
	tests := []struct {
		name          string
		config        ArithmeticMeanConfig
		expectedError string
	}{
		{
			name: "valid configuration passes",
			config: ArithmeticMeanConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: true,
			},
		},
		{
			name: "invalid tie breaker fails",
			config: ArithmeticMeanConfig{
				TieBreaker:       "invalid",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			expectedError: "configuration validation failed",
		},
		{
			name: "negative min score fails",
			config: ArithmeticMeanConfig{
				TieBreaker:       "first",
				MinScore:         -0.1,
				RequireAllScores: true,
			},
			expectedError: "configuration validation failed",
		},
		{
			name: "min score above 1.0 fails",
			config: ArithmeticMeanConfig{
				TieBreaker:       "first",
				MinScore:         1.1,
				RequireAllScores: true,
			},
			expectedError: "configuration validation failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			unit, err := NewArithmeticMeanUnit("test_arithmetic_mean", tt.config)
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

func TestArithmeticMeanUnit_Name(t *testing.T) {
	config := ArithmeticMeanConfig{
		TieBreaker:       "first",
		MinScore:         0.0,
		RequireAllScores: true,
	}

	unit, err := NewArithmeticMeanUnit("test_aggregator", config)
	require.NoError(t, err)

	assert.Equal(t, "test_aggregator", unit.Name())
}

func TestCreateArithmeticMeanUnit(t *testing.T) {
	t.Run("creates unit with default config", func(t *testing.T) {
		config := map[string]any{}

		unit, err := CreateArithmeticMeanUnit("test_id", config)
		require.NoError(t, err)
		assert.Equal(t, "test_id", unit.Name())

		// Verify default config was applied
		assert.Equal(t, TieFirst, unit.config.TieBreaker)
		assert.Equal(t, 0.0, unit.config.MinScore)
		assert.True(t, unit.config.RequireAllScores)
	})

	t.Run("creates unit with custom config", func(t *testing.T) {
		config := map[string]any{
			"tie_breaker":        "random",
			"min_score":          0.5,
			"require_all_scores": false,
		}

		unit, err := CreateArithmeticMeanUnit("test_id", config)
		require.NoError(t, err)
		assert.Equal(t, "test_id", unit.Name())

		// Verify custom config was applied
		assert.Equal(t, TieRandom, unit.config.TieBreaker)
		assert.Equal(t, 0.5, unit.config.MinScore)
		assert.False(t, unit.config.RequireAllScores)
	})

	t.Run("fails with empty id", func(t *testing.T) {
		config := map[string]any{}

		unit, err := CreateArithmeticMeanUnit("", config)
		require.Error(t, err)
		assert.Nil(t, unit)
		assert.Contains(t, err.Error(), "unit name cannot be empty")
	})
}

func TestArithmeticMeanUnit_UnmarshalParameters(t *testing.T) {
	tests := []struct {
		name          string
		yamlContent   string
		expectedError string
		validate      func(t *testing.T, unit *ArithmeticMeanUnit)
	}{
		{
			name: "valid YAML parameters",
			yamlContent: `
tie_breaker: random
min_score: 0.7
require_all_scores: false
`,
			validate: func(t *testing.T, unit *ArithmeticMeanUnit) {
				assert.Equal(t, TieRandom, unit.config.TieBreaker)
				assert.Equal(t, 0.7, unit.config.MinScore)
				assert.False(t, unit.config.RequireAllScores)
			},
		},
		{
			name: "invalid tie breaker",
			yamlContent: `
tie_breaker: invalid
min_score: 0.5
`,
			expectedError: "parameter validation failed",
		},
		{
			name: "invalid min score",
			yamlContent: `
tie_breaker: first
min_score: 1.5
`,
			expectedError: "parameter validation failed",
		},
		{
			name:          "empty YAML fails validation",
			yamlContent:   ``,
			expectedError: "parameter validation failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var node yaml.Node
			err := yaml.Unmarshal([]byte(tt.yamlContent), &node)
			require.NoError(t, err)

			unit := &ArithmeticMeanUnit{name: "test"}

			// Handle empty YAML case
			if len(node.Content) == 0 {
				// Create an empty node for empty YAML
				emptyNode := yaml.Node{Kind: yaml.MappingNode}
				err = unit.UnmarshalParameters(emptyNode)
			} else {
				err = unit.UnmarshalParameters(*node.Content[0])
			}

			if tt.expectedError != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.expectedError)
			} else {
				require.NoError(t, err)
				if tt.validate != nil {
					tt.validate(t, unit)
				}
			}
		})
	}
}

func TestDefaultArithmeticMeanConfig(t *testing.T) {
	config := DefaultArithmeticMeanConfig()

	assert.Equal(t, TieFirst, config.TieBreaker)
	assert.Equal(t, 0.0, config.MinScore)
	assert.True(t, config.RequireAllScores)
}
