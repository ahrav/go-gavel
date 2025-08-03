package middleware

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ahrav/go-gavel/internal/domain"
)

// biasedMockJudge implements a judge that consistently scores the first answer
// higher than subsequent answers, demonstrating clear positional bias.
// This mock is essential for testing bias mitigation effectiveness.
type biasedMockJudge struct {
	name        string
	firstScore  float64 // Score always given to first answer
	otherScore  float64 // Score given to non-first answers
	executeFunc func(ctx context.Context, state domain.State) (domain.State, error)
	validateErr error
	callCount   int // Track number of times Execute was called
	mu          sync.Mutex
}

func newBiasedMockJudge(name string, firstScore, otherScore float64) *biasedMockJudge {
	return &biasedMockJudge{
		name:       name,
		firstScore: firstScore,
		otherScore: otherScore,
	}
}

func (bmj *biasedMockJudge) Name() string {
	return bmj.name
}

func (bmj *biasedMockJudge) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	bmj.mu.Lock()
	bmj.callCount++
	bmj.mu.Unlock()

	// Use custom execute function if provided
	if bmj.executeFunc != nil {
		return bmj.executeFunc(ctx, state)
	}

	// Extract answers from state
	answers, ok := domain.Get(state, domain.KeyAnswers)
	if !ok {
		return state, errors.New("answers not found in state")
	}

	// Create biased scores - first answer gets higher score
	scores := make([]domain.JudgeSummary, len(answers))
	for i := range answers {
		var score float64
		var reasoning string

		if i == 0 {
			score = bmj.firstScore
			reasoning = fmt.Sprintf("First answer bias: scoring %.2f", score)
		} else {
			score = bmj.otherScore
			reasoning = fmt.Sprintf("Non-first answer: scoring %.2f", score)
		}

		scores[i] = domain.JudgeSummary{
			Reasoning:  reasoning,
			Confidence: 0.8,
			Score:      score,
		}
	}

	// Return state with biased scores
	return domain.With(state, domain.KeyJudgeScores, scores), nil
}

func (bmj *biasedMockJudge) Validate() error {
	return bmj.validateErr
}

func (bmj *biasedMockJudge) getCallCount() int {
	bmj.mu.Lock()
	defer bmj.mu.Unlock()
	return bmj.callCount
}

// neutralMockJudge provides consistent scoring without positional bias
// for comparison and control testing scenarios.
type neutralMockJudge struct {
	name        string
	score       float64
	executeFunc func(ctx context.Context, state domain.State) (domain.State, error)
	validateErr error
}

func newNeutralMockJudge(name string, score float64) *neutralMockJudge {
	return &neutralMockJudge{
		name:  name,
		score: score,
	}
}

func (nmj *neutralMockJudge) Name() string {
	return nmj.name
}

func (nmj *neutralMockJudge) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	if nmj.executeFunc != nil {
		return nmj.executeFunc(ctx, state)
	}

	answers, ok := domain.Get(state, domain.KeyAnswers)
	if !ok {
		return state, errors.New("answers not found in state")
	}

	// Create neutral scores - all answers get same score
	scores := make([]domain.JudgeSummary, len(answers))
	for i := range answers {
		scores[i] = domain.JudgeSummary{
			Reasoning:  fmt.Sprintf("Neutral scoring: %.2f", nmj.score),
			Confidence: 0.8,
			Score:      nmj.score,
		}
	}

	return domain.With(state, domain.KeyJudgeScores, scores), nil
}

func (nmj *neutralMockJudge) Validate() error {
	return nmj.validateErr
}

func TestNewPositionSwapMiddleware(t *testing.T) {
	mockJudge := newBiasedMockJudge("test-judge", 0.8, 0.3)

	middleware := NewPositionSwapMiddleware(mockJudge, "position-swap-test")

	assert.Equal(t, "position-swap-test", middleware.Name())
	assert.Equal(t, mockJudge, middleware.next)
}

func TestNewPositionSwapMiddleware_PanicsWithNilUnit(t *testing.T) {
	assert.Panics(t, func() {
		NewPositionSwapMiddleware(nil, "test")
	})
}

func TestNewPositionSwapMiddleware_PanicsWithEmptyName(t *testing.T) {
	mockJudge := newBiasedMockJudge("test-judge", 0.8, 0.3)

	assert.Panics(t, func() {
		NewPositionSwapMiddleware(mockJudge, "")
	})
}

func TestPositionSwapMiddleware_Validate(t *testing.T) {
	tests := []struct {
		name        string
		judge       *biasedMockJudge
		expectedErr string
	}{
		{
			name:        "valid configuration",
			judge:       newBiasedMockJudge("test-judge", 0.8, 0.3),
			expectedErr: "",
		},
		{
			name: "wrapped unit validation fails",
			judge: func() *biasedMockJudge {
				judge := newBiasedMockJudge("test-judge", 0.8, 0.3)
				judge.validateErr = errors.New("wrapped unit error")
				return judge
			}(),
			expectedErr: "wrapped unit validation failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			middleware := NewPositionSwapMiddleware(tt.judge, "test-middleware")
			err := middleware.Validate()

			if tt.expectedErr == "" {
				assert.NoError(t, err)
			} else {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.expectedErr)
			}
		})
	}
}

func TestPositionSwapMiddleware_Execute_TwoAnswers_BiasEffect(t *testing.T) {
	// Create biased judge that prefers first answer
	biasedJudge := newBiasedMockJudge("biased-judge", 0.9, 0.3)
	middleware := NewPositionSwapMiddleware(biasedJudge, "position-swap-test")

	// Create test state with two answers
	answers := []domain.Answer{
		{ID: "answer1", Content: "First answer"},
		{ID: "answer2", Content: "Second answer"},
	}
	state := domain.NewState()
	state = domain.With(state, domain.KeyAnswers, answers)

	// Execute middleware
	result, err := middleware.Execute(context.Background(), state)
	require.NoError(t, err)

	// Verify judge was called twice (dual execution)
	assert.Equal(t, 2, biasedJudge.getCallCount())

	// Extract combined scores
	combinedScores, ok := domain.Get(result, domain.KeyJudgeScores)
	require.True(t, ok, "combined scores should be present")
	require.Len(t, combinedScores, 2)

	// Without position swap: answer1=0.9, answer2=0.3 (bias toward first)
	// With position swap:
	// - Run 1 (original order): answer1=0.9, answer2=0.3
	// - Run 2 (reversed order): answer1=0.3, answer2=0.9 (now second gets first-position bias)
	// - Combined: answer1=(0.9+0.3)/2=0.6, answer2=(0.3+0.9)/2=0.6 (bias mitigated)

	expectedScore1 := (0.9 + 0.3) / 2.0 // 0.6
	expectedScore2 := (0.3 + 0.9) / 2.0 // 0.6

	assert.InDelta(t, expectedScore1, combinedScores[0].Score, 0.001,
		"first answer score should be arithmetic mean")
	assert.InDelta(t, expectedScore2, combinedScores[1].Score, 0.001,
		"second answer score should be arithmetic mean")

	// Verify bias mitigation - scores should now be equal
	assert.InDelta(t, combinedScores[0].Score, combinedScores[1].Score, 0.001,
		"bias mitigation should result in equal scores for identical content")

	// Verify reasoning contains bias mitigation information
	assert.Contains(t, combinedScores[0].Reasoning, "Position swap:")
	assert.Contains(t, combinedScores[1].Reasoning, "Position swap:")
}

func TestPositionSwapMiddleware_Execute_SingleAnswer(t *testing.T) {
	mockJudge := newNeutralMockJudge("neutral-judge", 0.7)
	middleware := NewPositionSwapMiddleware(mockJudge, "position-swap-test")

	// Create test state with single answer
	answers := []domain.Answer{
		{ID: "answer1", Content: "Only answer"},
	}
	state := domain.NewState()
	state = domain.With(state, domain.KeyAnswers, answers)

	// Execute middleware
	result, err := middleware.Execute(context.Background(), state)
	require.NoError(t, err)

	// Single answer should result in single execution (no bias possible)
	scores, ok := domain.Get(result, domain.KeyJudgeScores)
	require.True(t, ok)
	require.Len(t, scores, 1)

	assert.Equal(t, 0.7, scores[0].Score)
}

func TestPositionSwapMiddleware_Execute_ThreeAnswers_BiasEffect(t *testing.T) {
	// Create biased judge that strongly prefers first answer
	biasedJudge := newBiasedMockJudge("biased-judge", 0.95, 0.2)
	middleware := NewPositionSwapMiddleware(biasedJudge, "position-swap-test")

	// Create test state with three answers
	answers := []domain.Answer{
		{ID: "answer1", Content: "First answer"},
		{ID: "answer2", Content: "Second answer"},
		{ID: "answer3", Content: "Third answer"},
	}
	state := domain.NewState()
	state = domain.With(state, domain.KeyAnswers, answers)

	// Execute middleware
	result, err := middleware.Execute(context.Background(), state)
	require.NoError(t, err)

	// Verify dual execution occurred
	assert.Equal(t, 2, biasedJudge.getCallCount())

	// Extract combined scores
	combinedScores, ok := domain.Get(result, domain.KeyJudgeScores)
	require.True(t, ok)
	require.Len(t, combinedScores, 3)

	// Calculate expected scores after bias mitigation
	// Run 1 (original order): [0.95, 0.2, 0.2]
	// Run 2 (reversed order): [0.2, 0.2, 0.95]
	// Combined: [(0.95+0.2)/2, (0.2+0.2)/2, (0.2+0.95)/2] = [0.575, 0.2, 0.575]

	expectedScores := []float64{
		(0.95 + 0.2) / 2.0, // 0.575
		(0.2 + 0.2) / 2.0,  // 0.2
		(0.2 + 0.95) / 2.0, // 0.575
	}

	for i, expected := range expectedScores {
		assert.InDelta(t, expected, combinedScores[i].Score, 0.001,
			"answer %d score should match expected after bias mitigation", i+1)
	}

	// Verify that first and last answers now have equal scores (bias mitigation)
	assert.InDelta(t, combinedScores[0].Score, combinedScores[2].Score, 0.001,
		"first and last answers should have equal scores after bias mitigation")
}

func TestPositionSwapMiddleware_Execute_ErrorScenarios(t *testing.T) {
	tests := []struct {
		name        string
		setupState  func() domain.State
		setupJudge  func() *neutralMockJudge
		expectedErr string
	}{
		{
			name: "missing answers in state",
			setupState: func() domain.State {
				return domain.NewState() // No answers
			},
			setupJudge: func() *neutralMockJudge {
				return newNeutralMockJudge("test-judge", 0.5)
			},
			expectedErr: "answers not found in state",
		},
		{
			name: "empty answers list",
			setupState: func() domain.State {
				state := domain.NewState()
				return domain.With(state, domain.KeyAnswers, []domain.Answer{})
			},
			setupJudge: func() *neutralMockJudge {
				return newNeutralMockJudge("test-judge", 0.5)
			},
			expectedErr: "answers cannot be empty",
		},
		{
			name: "first execution fails",
			setupState: func() domain.State {
				answers := []domain.Answer{
					{ID: "answer1", Content: "First"},
					{ID: "answer2", Content: "Second"},
				}
				state := domain.NewState()
				return domain.With(state, domain.KeyAnswers, answers)
			},
			setupJudge: func() *neutralMockJudge {
				judge := newNeutralMockJudge("failing-judge", 0.5)
				judge.executeFunc = func(ctx context.Context, state domain.State) (domain.State, error) {
					return state, errors.New("first execution failed")
				}
				return judge
			},
			expectedErr: "first execution failed",
		},
		{
			name: "second execution fails",
			setupState: func() domain.State {
				answers := []domain.Answer{
					{ID: "answer1", Content: "First"},
					{ID: "answer2", Content: "Second"},
				}
				state := domain.NewState()
				return domain.With(state, domain.KeyAnswers, answers)
			},
			setupJudge: func() *neutralMockJudge {
				judge := newNeutralMockJudge("partially-failing-judge", 0.5)
				callCount := 0
				judge.executeFunc = func(ctx context.Context, state domain.State) (domain.State, error) {
					callCount++
					if callCount == 1 {
						// First call succeeds
						answers, _ := domain.Get(state, domain.KeyAnswers)
						scores := make([]domain.JudgeSummary, len(answers))
						for i := range scores {
							scores[i] = domain.JudgeSummary{Score: 0.5, Confidence: 0.8}
						}
						return domain.With(state, domain.KeyJudgeScores, scores), nil
					}
					// Second call fails
					return state, errors.New("second execution failed")
				}
				return judge
			},
			expectedErr: "second execution failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			judge := tt.setupJudge()
			middleware := NewPositionSwapMiddleware(judge, "test-middleware")
			state := tt.setupState()

			result, err := middleware.Execute(context.Background(), state)

			assert.Error(t, err)
			assert.Contains(t, err.Error(), tt.expectedErr)
			assert.Equal(t, state, result) // State should be unchanged on error
		})
	}
}

func TestPositionSwapMiddleware_Execute_ScoreCombinationErrors(t *testing.T) {
	judge := newNeutralMockJudge("inconsistent-judge", 0.5)

	// Make judge return inconsistent score counts between executions
	callCount := 0
	judge.executeFunc = func(ctx context.Context, state domain.State) (domain.State, error) {
		callCount++
		answers, _ := domain.Get(state, domain.KeyAnswers)

		var scores []domain.JudgeSummary
		if callCount == 1 {
			// First execution returns correct number of scores
			scores = make([]domain.JudgeSummary, len(answers))
		} else {
			// Second execution returns wrong number of scores
			scores = make([]domain.JudgeSummary, len(answers)-1)
		}

		for i := range scores {
			scores[i] = domain.JudgeSummary{Score: 0.5, Confidence: 0.8}
		}

		return domain.With(state, domain.KeyJudgeScores, scores), nil
	}

	middleware := NewPositionSwapMiddleware(judge, "test-middleware")

	answers := []domain.Answer{
		{ID: "answer1", Content: "First"},
		{ID: "answer2", Content: "Second"},
	}
	state := domain.NewState()
	state = domain.With(state, domain.KeyAnswers, answers)

	result, err := middleware.Execute(context.Background(), state)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "score count mismatch")
	assert.Equal(t, state, result)
}

func TestPositionSwapMiddleware_ConcurrentExecution(t *testing.T) {
	// Test thread safety with concurrent executions
	biasedJudge := newBiasedMockJudge("concurrent-judge", 0.8, 0.4)
	middleware := NewPositionSwapMiddleware(biasedJudge, "concurrent-test")

	const numGoroutines = 50
	var wg sync.WaitGroup
	results := make([]domain.State, numGoroutines)
	errors := make([]error, numGoroutines)

	// Run multiple executions concurrently
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()

			// Each goroutine uses slightly different answers
			answers := []domain.Answer{
				{ID: fmt.Sprintf("answer1-%d", index), Content: "First answer"},
				{ID: fmt.Sprintf("answer2-%d", index), Content: "Second answer"},
			}
			state := domain.NewState()
			state = domain.With(state, domain.KeyAnswers, answers)

			result, err := middleware.Execute(context.Background(), state)
			results[index] = result
			errors[index] = err
		}(i)
	}

	wg.Wait()

	// Verify all executions succeeded
	for i := 0; i < numGoroutines; i++ {
		assert.NoError(t, errors[i], "execution %d should not fail", i)

		scores, ok := domain.Get(results[i], domain.KeyJudgeScores)
		require.True(t, ok, "execution %d should have scores", i)
		require.Len(t, scores, 2, "execution %d should have 2 scores", i)

		// Verify bias mitigation occurred
		expectedScore := (0.8 + 0.4) / 2.0 // 0.6
		assert.InDelta(t, expectedScore, scores[0].Score, 0.001, "execution %d first score", i)
		assert.InDelta(t, expectedScore, scores[1].Score, 0.001, "execution %d second score", i)
	}

	// Verify judge was called the correct number of times (2 per execution)
	totalCalls := biasedJudge.getCallCount()
	assert.Equal(t, numGoroutines*2, totalCalls, "judge should be called twice per execution")
}

func TestPositionSwapMiddleware_BiasDetectionAndMitigation(t *testing.T) {
	// This integration test demonstrates the actual bias detection and mitigation
	// Create a strongly biased judge
	stronglyBiasedJudge := newBiasedMockJudge("strongly-biased", 1.0, 0.1)

	// Test WITHOUT position swap middleware (control)
	answers := []domain.Answer{
		{ID: "answer1", Content: "Identical content"},
		{ID: "answer2", Content: "Identical content"},
	}
	state := domain.NewState()
	state = domain.With(state, domain.KeyAnswers, answers)

	// Direct execution without middleware
	controlResult, err := stronglyBiasedJudge.Execute(context.Background(), state)
	require.NoError(t, err)

	controlScores, ok := domain.Get(controlResult, domain.KeyJudgeScores)
	require.True(t, ok)

	// Verify bias exists in control case
	assert.Equal(t, 1.0, controlScores[0].Score, "first answer should get high score due to bias")
	assert.Equal(t, 0.1, controlScores[1].Score, "second answer should get low score due to bias")

	biasAmount := controlScores[0].Score - controlScores[1].Score
	assert.Equal(t, 0.9, biasAmount, "significant bias should exist in control case")

	// Test WITH position swap middleware
	middleware := NewPositionSwapMiddleware(
		newBiasedMockJudge("mitigated-biased", 1.0, 0.1),
		"bias-mitigation-test",
	)

	mitigatedResult, err := middleware.Execute(context.Background(), state)
	require.NoError(t, err)

	mitigatedScores, ok := domain.Get(mitigatedResult, domain.KeyJudgeScores)
	require.True(t, ok)

	// Verify bias mitigation
	expectedMitigatedScore := (1.0 + 0.1) / 2.0 // 0.55
	assert.InDelta(t, expectedMitigatedScore, mitigatedScores[0].Score, 0.001)
	assert.InDelta(t, expectedMitigatedScore, mitigatedScores[1].Score, 0.001)

	mitigatedBiasAmount := mitigatedScores[0].Score - mitigatedScores[1].Score
	assert.InDelta(t, 0.0, mitigatedBiasAmount, 0.001, "bias should be eliminated by position swap")

	// Verify bias reduction percentage
	biasReduction := ((biasAmount - mitigatedBiasAmount) / biasAmount) * 100
	assert.Greater(t, biasReduction, 99.0, "bias reduction should be >99%")
}

func TestPositionSwapMiddleware_AnswerOrderPreservation(t *testing.T) {
	// Ensure original answer order is preserved in final output
	judge := newBiasedMockJudge("order-test-judge", 0.9, 0.3)
	middleware := NewPositionSwapMiddleware(judge, "order-preservation-test")

	originalAnswers := []domain.Answer{
		{ID: "first", Content: "First answer content"},
		{ID: "second", Content: "Second answer content"},
		{ID: "third", Content: "Third answer content"},
	}

	state := domain.NewState()
	state = domain.With(state, domain.KeyAnswers, originalAnswers)

	result, err := middleware.Execute(context.Background(), state)
	require.NoError(t, err)

	// Verify answer order is preserved
	finalAnswers, ok := domain.Get(result, domain.KeyAnswers)
	require.True(t, ok)
	require.Len(t, finalAnswers, len(originalAnswers))

	for i, answer := range finalAnswers {
		assert.Equal(t, originalAnswers[i].ID, answer.ID, "answer ID order should be preserved")
		assert.Equal(t, originalAnswers[i].Content, answer.Content, "answer content should be preserved")
	}

	// Verify scores are provided for each answer in correct order
	scores, ok := domain.Get(result, domain.KeyJudgeScores)
	require.True(t, ok)
	require.Len(t, scores, len(originalAnswers))
}

// BenchmarkPositionSwapMiddleware measures performance impact of dual execution
func BenchmarkPositionSwapMiddleware(b *testing.B) {
	judge := newNeutralMockJudge("benchmark-judge", 0.5)
	middleware := NewPositionSwapMiddleware(judge, "benchmark-test")

	answers := []domain.Answer{
		{ID: "answer1", Content: "First benchmark answer"},
		{ID: "answer2", Content: "Second benchmark answer"},
	}
	state := domain.NewState()
	state = domain.With(state, domain.KeyAnswers, answers)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := middleware.Execute(context.Background(), state)
		if err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
	}
}
