package units

import (
	"context"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

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
			name: "selects highest score winner with winner's score as aggregate",
			config: ArithmeticMeanConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			scores: []float64{0.7, 0.9, 0.8}, // winner score = 0.9
			candidates: []domain.Answer{
				{ID: "answer1", Content: "First answer"},
				{ID: "answer2", Content: "Second answer"},
				{ID: "answer3", Content: "Third answer"},
			},
			expectedWinnerID: "answer2",
			expectedScore:    0.9, // winner's score
		},
		{
			name: "handles equal scores with first tie breaker and returns winner's score",
			config: ArithmeticMeanConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			scores: []float64{0.8, 0.8, 0.7}, // winner score = 0.8
			candidates: []domain.Answer{
				{ID: "answer1", Content: "First answer"},
				{ID: "answer2", Content: "Second answer"},
				{ID: "answer3", Content: "Third answer"},
			},
			expectedWinnerID: "answer1",
			expectedScore:    0.8, // winner's score (first of tied winners)
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
			name: "enforces minimum score requirement against winner's score",
			config: ArithmeticMeanConfig{
				TieBreaker:       "first",
				MinScore:         0.9,
				RequireAllScores: true,
			},
			scores:        []float64{0.8, 0.7, 0.85}, // winner score = 0.85 < 0.9
			candidates:    []domain.Answer{{ID: "a1"}, {ID: "a2"}, {ID: "a3"}},
			expectedError: "highest score below minimum threshold",
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
				assert.InDelta(t, 0.9, verdict.AggregateScore, 0.0001) // winner's score = 0.9
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
				assert.InDelta(t, 0.9, verdict.AggregateScore, 0.0001) // winner's score = 0.9
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
	})
}
