package units

import (
	"context"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ahrav/go-gavel/internal/domain"
)

func TestMedianPoolUnit_calculateMedian(t *testing.T) {
	unit, err := NewMedianPoolUnit("test", DefaultMedianPoolConfig())
	require.NoError(t, err)

	tests := []struct {
		name     string
		scores   []float64
		expected float64
	}{
		{
			name:     "odd number of scores returns middle value",
			scores:   []float64{0.1, 0.5, 0.9},
			expected: 0.5,
		},
		{
			name:     "even number of scores returns mathematical median",
			scores:   []float64{0.1, 0.3, 0.7, 0.9}, // sorted: [0.1, 0.3, 0.7, 0.9], median = (0.3 + 0.7) / 2 = 0.5
			expected: 0.5,
		},
		{
			name:     "unsorted scores are handled correctly",
			scores:   []float64{0.9, 0.1, 0.5}, // will be sorted to [0.1, 0.5, 0.9]
			expected: 0.5,
		},
		{
			name:     "single score returns that score",
			scores:   []float64{0.75},
			expected: 0.75,
		},
		{
			name:     "two scores returns mathematical median (average)",
			scores:   []float64{0.3, 0.7},
			expected: 0.5, // (0.3 + 0.7) / 2 = 0.5
		},
		{
			name:     "empty slice returns zero",
			scores:   []float64{},
			expected: 0.0,
		},
		// NEW TESTS: Mathematical median behavior (these will fail initially)
		{
			name:     "mathematical median for even length - simple case",
			scores:   []float64{0.3, 0.7},
			expected: 0.5, // (0.3 + 0.7) / 2 = 0.5, not 0.7
		},
		{
			name:     "mathematical median for even length - four values",
			scores:   []float64{0.1, 0.3, 0.7, 0.9},
			expected: 0.5, // (0.3 + 0.7) / 2 = 0.5, not 0.7
		},
		{
			name:     "mathematical median for even length - larger spread",
			scores:   []float64{0.2, 0.6, 0.7, 0.9},
			expected: 0.65, // (0.6 + 0.7) / 2 = 0.65, not 0.7
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := unit.calculateMedian(tt.scores)
			assert.InDelta(t, tt.expected, result, 0.0001, "Expected median %f, got %f", tt.expected, result)
		})
	}
}

func TestMedianPoolUnit_Aggregate(t *testing.T) {
	tests := []struct {
		name             string
		config           MedianPoolConfig
		scores           []float64
		candidates       []domain.Answer
		expectedWinnerID string
		expectedScore    float64 // This should be the median of all scores
		expectedError    string
	}{
		{
			name: "selects candidate closest to median with odd number of scores",
			config: MedianPoolConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			scores: []float64{0.3, 0.7, 0.9}, // sorted: [0.3, 0.7, 0.9], median = 0.7
			candidates: []domain.Answer{
				{ID: "answer1", Content: "First answer"},  // score 0.3, distance = |0.3-0.7| = 0.4
				{ID: "answer2", Content: "Second answer"}, // score 0.7, distance = |0.7-0.7| = 0.0 (closest)
				{ID: "answer3", Content: "Third answer"},  // score 0.9, distance = |0.9-0.7| = 0.2
			},
			expectedWinnerID: "answer2", // closest to median
			expectedScore:    0.7,       // median of [0.3, 0.7, 0.9]
		},
		{
			name: "selects candidate closest to mathematical median with even number of scores",
			config: MedianPoolConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			scores: []float64{0.2, 0.6, 0.7, 0.9}, // sorted: [0.2, 0.6, 0.7, 0.9], median = (0.6 + 0.7) / 2 = 0.65
			candidates: []domain.Answer{
				{ID: "answer1", Content: "First answer"},  // score 0.2, distance = |0.2-0.65| = 0.45
				{ID: "answer2", Content: "Second answer"}, // score 0.6, distance = |0.6-0.65| = 0.05
				{ID: "answer3", Content: "Third answer"},  // score 0.7, distance = |0.7-0.65| = 0.05
				{ID: "answer4", Content: "Fourth answer"}, // score 0.9, distance = |0.9-0.65| = 0.25
			},
			expectedWinnerID: "answer2", // first of the tied closest candidates (0.05 distance)
			expectedScore:    0.65,      // mathematical median of [0.2, 0.6, 0.7, 0.9]
		},
		{
			name: "handles ties in distance to median with first tie breaker",
			config: MedianPoolConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			scores: []float64{0.4, 0.6, 0.8}, // median = 0.6
			candidates: []domain.Answer{
				{ID: "answer1", Content: "First answer"},  // score 0.4, distance = |0.4-0.6| = 0.2
				{ID: "answer2", Content: "Second answer"}, // score 0.6, distance = |0.6-0.6| = 0.0 (closest)
				{ID: "answer3", Content: "Third answer"},  // score 0.8, distance = |0.8-0.6| = 0.2 (tied with answer1)
			},
			expectedWinnerID: "answer2", // closest to median (no tie)
			expectedScore:    0.6,       // median
		},
		{
			name: "handles exact ties in distance with first tie breaker",
			config: MedianPoolConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			scores: []float64{0.5, 0.7, 0.5}, // median = 0.5 (sorted: [0.5, 0.5, 0.7])
			candidates: []domain.Answer{
				{ID: "answer1", Content: "First answer"},  // score 0.5, distance = |0.5-0.5| = 0.0 (tied)
				{ID: "answer2", Content: "Second answer"}, // score 0.7, distance = |0.7-0.5| = 0.2
				{ID: "answer3", Content: "Third answer"},  // score 0.5, distance = |0.5-0.5| = 0.0 (tied)
			},
			expectedWinnerID: "answer1", // first tied candidate
			expectedScore:    0.5,       // median
		},
		{
			name: "fails with tie breaker error",
			config: MedianPoolConfig{
				TieBreaker:       "error",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			scores:        []float64{0.5, 0.7, 0.5}, // Two candidates tied for closest to median
			candidates:    []domain.Answer{{ID: "a1"}, {ID: "a2"}, {ID: "a3"}},
			expectedError: "multiple answers tied with",
		},
		{
			name: "enforces minimum score requirement against median",
			config: MedianPoolConfig{
				TieBreaker:       "first",
				MinScore:         0.8,
				RequireAllScores: true,
			},
			scores:        []float64{0.6, 0.7, 0.75}, // median = 0.7 < 0.8
			candidates:    []domain.Answer{{ID: "a1"}, {ID: "a2"}, {ID: "a3"}},
			expectedError: "highest score below minimum threshold",
		},
		{
			name: "handles empty scores",
			config: MedianPoolConfig{
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
			config: MedianPoolConfig{
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
			config: MedianPoolConfig{
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
			config: MedianPoolConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			scores:        []float64{0.8, math.Inf(1), 0.9},
			candidates:    []domain.Answer{{ID: "a1"}, {ID: "a2"}, {ID: "a3"}},
			expectedError: "invalid score at index 1",
		},
		{
			name: "single candidate returns that candidate with its score as median",
			config: MedianPoolConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			scores: []float64{0.75},
			candidates: []domain.Answer{
				{ID: "single", Content: "Only answer"},
			},
			expectedWinnerID: "single",
			expectedScore:    0.75, // median of [0.75] = 0.75
		},
		{
			name: "mathematical median calculation with four scores",
			config: MedianPoolConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			scores: []float64{0.1, 0.3, 0.7, 0.9}, // sorted: [0.1, 0.3, 0.7, 0.9], median = (0.3 + 0.7) / 2 = 0.5
			candidates: []domain.Answer{
				{ID: "answer1", Content: "First answer"},  // score 0.1, distance = |0.1-0.5| = 0.4
				{ID: "answer2", Content: "Second answer"}, // score 0.3, distance = |0.3-0.5| = 0.2 (but 0.20000000000000001 due to fp)
				{ID: "answer3", Content: "Third answer"},  // score 0.7, distance = |0.7-0.5| = 0.2 (but 0.19999999999999996 due to fp, closer!)
				{ID: "answer4", Content: "Fourth answer"}, // score 0.9, distance = |0.9-0.5| = 0.4
			},
			expectedWinnerID: "answer3", // closest candidate due to floating point precision
			expectedScore:    0.5,       // mathematical median
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			unit, err := NewMedianPoolUnit("test_median_pool", tt.config)
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

func TestMedianPoolUnit_Execute(t *testing.T) {
	tests := []struct {
		name           string
		config         MedianPoolConfig
		setupState     func() domain.State
		expectedError  string
		validateResult func(t *testing.T, state domain.State)
	}{
		{
			name: "successful execution with valid data",
			config: MedianPoolConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			setupState: func() domain.State {
				state := domain.NewState()
				answers := []domain.Answer{
					{ID: "answer1", Content: "First answer"},
					{ID: "answer2", Content: "Second answer"},
					{ID: "answer3", Content: "Third answer"},
				}
				judgeSummaries := []domain.JudgeSummary{
					{Score: 0.3, Reasoning: "Weak answer", Confidence: 0.8},
					{Score: 0.7, Reasoning: "Good answer", Confidence: 0.9},
					{Score: 0.9, Reasoning: "Great answer", Confidence: 0.95},
				}
				state = domain.With(state, domain.KeyAnswers, answers)
				state = domain.With(state, domain.KeyJudgeScores, judgeSummaries)
				return state
			},
			validateResult: func(t *testing.T, state domain.State) {
				verdict, ok := domain.Get(state, domain.KeyVerdict)
				require.True(t, ok, "Verdict should be present in state")
				require.NotNil(t, verdict, "Verdict should not be nil")

				// median of [0.3, 0.7, 0.9] = 0.7, answer2 has score 0.7 (exact match)
				assert.Equal(t, "answer2", verdict.WinnerAnswer.ID)
				assert.Equal(t, 0.7, verdict.AggregateScore)
				assert.Contains(t, verdict.ID, "test_median_pool_verdict")
			},
		},
		{
			name: "fails when answers missing from state",
			config: MedianPoolConfig{
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
			config: MedianPoolConfig{
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
			config: MedianPoolConfig{
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
					{Score: 0.6, Reasoning: "Good", Confidence: 0.9},
					{Score: 0.8, Reasoning: "Better", Confidence: 0.95},
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
				// median of [0.6, 0.8] = (0.6 + 0.8) / 2 = 0.7, answer1 (score 0.6) is closer to 0.7
				assert.Equal(t, "answer1", verdict.WinnerAnswer.ID)
				assert.Equal(t, 0.7, verdict.AggregateScore)
			},
		},
		{
			name: "fails length mismatch when RequireAllScores is true",
			config: MedianPoolConfig{
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
			unit, err := NewMedianPoolUnit("test_median_pool", tt.config)
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

func TestMedianPoolUnit_Validate(t *testing.T) {
	tests := []struct {
		name          string
		config        MedianPoolConfig
		expectedError string
	}{
		{
			name: "valid configuration passes",
			config: MedianPoolConfig{
				TieBreaker:       "first",
				MinScore:         0.0,
				RequireAllScores: true,
			},
		},
		{
			name: "invalid tie breaker fails",
			config: MedianPoolConfig{
				TieBreaker:       "invalid",
				MinScore:         0.0,
				RequireAllScores: true,
			},
			expectedError: "configuration validation failed",
		},
		{
			name: "negative min score fails",
			config: MedianPoolConfig{
				TieBreaker:       "first",
				MinScore:         -0.1,
				RequireAllScores: true,
			},
			expectedError: "configuration validation failed",
		},
		{
			name: "min score above 1.0 fails",
			config: MedianPoolConfig{
				TieBreaker:       "first",
				MinScore:         1.1,
				RequireAllScores: true,
			},
			expectedError: "configuration validation failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			unit, err := NewMedianPoolUnit("test_median_pool", tt.config)
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

func TestMedianPoolUnit_Name(t *testing.T) {
	config := MedianPoolConfig{
		TieBreaker:       "first",
		MinScore:         0.0,
		RequireAllScores: true,
	}

	unit, err := NewMedianPoolUnit("test_median_aggregator", config)
	require.NoError(t, err)

	assert.Equal(t, "test_median_aggregator", unit.Name())
}

func TestCreateMedianPoolUnit(t *testing.T) {
	t.Run("creates unit with default config", func(t *testing.T) {
		config := map[string]any{}

		unit, err := CreateMedianPoolUnit("test_id", config)
		require.NoError(t, err)
		assert.Equal(t, "test_id", unit.Name())
	})

	t.Run("creates unit with custom config", func(t *testing.T) {
		config := map[string]any{
			"tie_breaker":        "random",
			"min_score":          0.5,
			"require_all_scores": false,
		}

		unit, err := CreateMedianPoolUnit("test_id", config)
		require.NoError(t, err)
		assert.Equal(t, "test_id", unit.Name())
	})

	t.Run("fails with invalid tie_breaker type", func(t *testing.T) {
		config := map[string]any{
			"tie_breaker": 123, // should be string
		}

		unit, err := CreateMedianPoolUnit("test_id", config)
		require.Error(t, err)
		assert.Nil(t, unit)
		assert.Contains(t, err.Error(), "tie_breaker must be a string")
	})

	t.Run("fails with invalid tie_breaker value", func(t *testing.T) {
		config := map[string]any{
			"tie_breaker": "invalid_value",
		}

		unit, err := CreateMedianPoolUnit("test_id", config)
		require.Error(t, err)
		assert.Nil(t, unit)
		assert.Contains(t, err.Error(), "invalid tie_breaker value: invalid_value")
	})

	t.Run("fails with invalid min_score type", func(t *testing.T) {
		config := map[string]any{
			"min_score": "0.5", // should be float64
		}

		unit, err := CreateMedianPoolUnit("test_id", config)
		require.Error(t, err)
		assert.Nil(t, unit)
		assert.Contains(t, err.Error(), "min_score must be a float64")
	})

	t.Run("fails with invalid require_all_scores type", func(t *testing.T) {
		config := map[string]any{
			"require_all_scores": "true", // should be bool
		}

		unit, err := CreateMedianPoolUnit("test_id", config)
		require.Error(t, err)
		assert.Nil(t, unit)
		assert.Contains(t, err.Error(), "require_all_scores must be a bool")
	})
}
