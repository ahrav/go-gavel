// Package middleware_test contains the unit tests for the middleware package.
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
// This mock is essential for testing the effectiveness of bias mitigation strategies.
type biasedMockJudge struct {
	name        string
	firstScore  float64 // The score always given to the first answer presented.
	otherScore  float64 // The score given to all non-first answers.
	executeFunc func(ctx context.Context, state domain.State) (domain.State, error)
	validateErr error
	callCount   int // Tracks the number of times Execute was called.
	mu          sync.Mutex
}

// newBiasedMockJudge creates a new judge with a configurable positional bias.
func newBiasedMockJudge(name string, firstScore, otherScore float64) *biasedMockJudge {
	return &biasedMockJudge{
		name:       name,
		firstScore: firstScore,
		otherScore: otherScore,
	}
}

// Name returns the mock judge's name.
func (bmj *biasedMockJudge) Name() string {
	return bmj.name
}

// Execute applies a biased scoring model to the answers in the state.
// The first answer receives a consistently higher score.
func (bmj *biasedMockJudge) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	bmj.mu.Lock()
	bmj.callCount++
	bmj.mu.Unlock()

	// Use a custom execute function if one is provided for the test.
	if bmj.executeFunc != nil {
		return bmj.executeFunc(ctx, state)
	}

	// Extract answers from the current state.
	answers, ok := domain.Get(state, domain.KeyAnswers)
	if !ok {
		return state, errors.New("answers not found in state")
	}

	// Create biased scores where the first answer is always scored higher.
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

	// Return the state updated with the biased scores.
	return domain.With(state, domain.KeyJudgeScores, scores), nil
}

// Validate returns a predefined validation error for testing purposes.
func (bmj *biasedMockJudge) Validate() error {
	return bmj.validateErr
}

// getCallCount returns the number of times the Execute method has been called.
func (bmj *biasedMockJudge) getCallCount() int {
	bmj.mu.Lock()
	defer bmj.mu.Unlock()
	return bmj.callCount
}

// neutralMockJudge provides consistent scoring without any positional bias,
// serving as a control for comparison in testing scenarios.
type neutralMockJudge struct {
	name        string
	score       float64
	executeFunc func(ctx context.Context, state domain.State) (domain.State, error)
	validateErr error
}

// newNeutralMockJudge creates a judge that assigns the same score to all answers.
func newNeutralMockJudge(name string, score float64) *neutralMockJudge {
	return &neutralMockJudge{
		name:  name,
		score: score,
	}
}

// Name returns the mock judge's name.
func (nmj *neutralMockJudge) Name() string {
	return nmj.name
}

// Execute applies a neutral scoring model, giving all answers the same score.
func (nmj *neutralMockJudge) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	if nmj.executeFunc != nil {
		return nmj.executeFunc(ctx, state)
	}

	answers, ok := domain.Get(state, domain.KeyAnswers)
	if !ok {
		return state, errors.New("answers not found in state")
	}

	// Create neutral scores where all answers receive the same score.
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

// Validate returns a predefined validation error.
func (nmj *neutralMockJudge) Validate() error {
	return nmj.validateErr
}

// TestNewPositionSwapMiddleware tests the successful creation of the middleware.
func TestNewPositionSwapMiddleware(t *testing.T) {
	mockJudge := newBiasedMockJudge("test-judge", 0.8, 0.3)

	middleware := NewPositionSwapMiddleware(mockJudge, "position-swap-test")

	assert.Equal(t, "position-swap-test", middleware.Name())
	assert.Equal(t, mockJudge, middleware.next)
}

// TestNewPositionSwapMiddleware_PanicsWithNilUnit tests that creation panics
// if the wrapped unit (judge) is nil.
func TestNewPositionSwapMiddleware_PanicsWithNilUnit(t *testing.T) {
	assert.Panics(t, func() {
		NewPositionSwapMiddleware(nil, "test")
	})
}

// TestNewPositionSwapMiddleware_PanicsWithEmptyName tests that creation panics
// if an empty name is provided for the middleware instance.
func TestNewPositionSwapMiddleware_PanicsWithEmptyName(t *testing.T) {
	mockJudge := newBiasedMockJudge("test-judge", 0.8, 0.3)

	assert.Panics(t, func() {
		NewPositionSwapMiddleware(mockJudge, "")
	})
}

// TestPositionSwapMiddleware_Validate tests the validation logic of the middleware.
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

// TestPositionSwapMiddleware_Execute_TwoAnswers_BiasEffect tests the core bias
// mitigation logic with two answers.
func TestPositionSwapMiddleware_Execute_TwoAnswers_BiasEffect(t *testing.T) {
	// Create a biased judge that prefers the first answer.
	biasedJudge := newBiasedMockJudge("biased-judge", 0.9, 0.3)
	middleware := NewPositionSwapMiddleware(biasedJudge, "position-swap-test")

	// Create a test state with two answers.
	answers := []domain.Answer{
		{ID: "answer1", Content: "First answer"},
		{ID: "answer2", Content: "Second answer"},
	}
	state := domain.NewState()
	state = domain.With(state, domain.KeyAnswers, answers)

	// Execute the middleware.
	result, err := middleware.Execute(context.Background(), state)
	require.NoError(t, err)

	// The judge should be called twice for dual execution.
	assert.Equal(t, 2, biasedJudge.getCallCount())

	// Extract the combined scores after mitigation.
	combinedScores, ok := domain.Get(result, domain.KeyJudgeScores)
	require.True(t, ok, "combined scores should be present")
	require.Len(t, combinedScores, 2)

	// The expected behavior is that the bias is averaged out.
	// Run 1 (original order): answer1=0.9, answer2=0.3
	// Run 2 (reversed order): answer2=0.9, answer1=0.3
	// Combined: answer1=(0.9+0.3)/2=0.6, answer2=(0.3+0.9)/2=0.6
	expectedScore := (0.9 + 0.3) / 2.0 // 0.6

	assert.InDelta(t, expectedScore, combinedScores[0].Score, 0.001,
		"first answer score should be the arithmetic mean")
	assert.InDelta(t, expectedScore, combinedScores[1].Score, 0.001,
		"second answer score should be the arithmetic mean")

	// After mitigation, the scores for identical content should be equal.
	assert.InDelta(t, combinedScores[0].Score, combinedScores[1].Score, 0.001,
		"bias mitigation should result in equal scores")

	// The reasoning should indicate that position swapping occurred.
	assert.Contains(t, combinedScores[0].Reasoning, "Position swap:")
	assert.Contains(t, combinedScores[1].Reasoning, "Position swap:")
}

// TestPositionSwapMiddleware_Execute_SingleAnswer tests that the middleware
// bypasses dual execution when there is only one answer.
func TestPositionSwapMiddleware_Execute_SingleAnswer(t *testing.T) {
	mockJudge := newNeutralMockJudge("neutral-judge", 0.7)
	middleware := NewPositionSwapMiddleware(mockJudge, "position-swap-test")

	// Create a test state with a single answer.
	answers := []domain.Answer{
		{ID: "answer1", Content: "Only answer"},
	}
	state := domain.NewState()
	state = domain.With(state, domain.KeyAnswers, answers)

	// Execute the middleware.
	result, err := middleware.Execute(context.Background(), state)
	require.NoError(t, err)

	// A single answer means no positional bias is possible, so it should pass through.
	scores, ok := domain.Get(result, domain.KeyJudgeScores)
	require.True(t, ok)
	require.Len(t, scores, 1)

	assert.Equal(t, 0.7, scores[0].Score)
}

// TestPositionSwapMiddleware_Execute_ThreeAnswers_BiasEffect tests bias mitigation
// with an odd number of answers.
func TestPositionSwapMiddleware_Execute_ThreeAnswers_BiasEffect(t *testing.T) {
	// Create a judge with a strong bias for the first position.
	biasedJudge := newBiasedMockJudge("biased-judge", 0.95, 0.2)
	middleware := NewPositionSwapMiddleware(biasedJudge, "position-swap-test")

	// Create a test state with three answers.
	answers := []domain.Answer{
		{ID: "answer1", Content: "First answer"},
		{ID: "answer2", Content: "Second answer"},
		{ID: "answer3", Content: "Third answer"},
	}
	state := domain.NewState()
	state = domain.With(state, domain.KeyAnswers, answers)

	// Execute the middleware.
	result, err := middleware.Execute(context.Background(), state)
	require.NoError(t, err)

	// Dual execution should still occur.
	assert.Equal(t, 2, biasedJudge.getCallCount())

	// Extract the combined scores.
	combinedScores, ok := domain.Get(result, domain.KeyJudgeScores)
	require.True(t, ok)
	require.Len(t, combinedScores, 3)

	// Calculate the expected scores after mitigation.
	// Run 1 (original): [0.95, 0.2, 0.2]
	// Run 2 (reversed): [0.2, 0.2, 0.95]
	// Combined: [(0.95+0.2)/2, (0.2+0.2)/2, (0.2+0.95)/2]
	expectedScores := []float64{
		(0.95 + 0.2) / 2.0, // 0.575
		(0.2 + 0.2) / 2.0,  // 0.2
		(0.2 + 0.95) / 2.0, // 0.575
	}

	for i, expected := range expectedScores {
		assert.InDelta(t, expected, combinedScores[i].Score, 0.001,
			"answer %d score should match expected after bias mitigation", i+1)
	}

	// The first and last answers should now have equal scores.
	assert.InDelta(t, combinedScores[0].Score, combinedScores[2].Score, 0.001,
		"first and last answers should have equal scores after bias mitigation")
}

// TestPositionSwapMiddleware_Execute_ErrorScenarios tests various failure modes
// during middleware execution.
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
				return domain.NewState() // No answers are set.
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
				answers := []domain.Answer{{ID: "a"}, {ID: "b"}}
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
				answers := []domain.Answer{{ID: "a"}, {ID: "b"}}
				state := domain.NewState()
				return domain.With(state, domain.KeyAnswers, answers)
			},
			setupJudge: func() *neutralMockJudge {
				judge := newNeutralMockJudge("partially-failing-judge", 0.5)
				callCount := 0
				judge.executeFunc = func(ctx context.Context, state domain.State) (domain.State, error) {
					callCount++
					if callCount == 1 {
						// The first call succeeds.
						answers, _ := domain.Get(state, domain.KeyAnswers)
						scores := make([]domain.JudgeSummary, len(answers))
						return domain.With(state, domain.KeyJudgeScores, scores), nil
					}
					// The second call fails.
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
			assert.Equal(t, state, result) // State should be unchanged on error.
		})
	}
}

// TestPositionSwapMiddleware_Execute_ScoreCombinationErrors tests for errors when
// combining scores from the two judge executions.
func TestPositionSwapMiddleware_Execute_ScoreCombinationErrors(t *testing.T) {
	judge := newNeutralMockJudge("inconsistent-judge", 0.5)

	// This judge is rigged to return a different number of scores on the second call.
	callCount := 0
	judge.executeFunc = func(ctx context.Context, state domain.State) (domain.State, error) {
		callCount++
		answers, _ := domain.Get(state, domain.KeyAnswers)

		var scores []domain.JudgeSummary
		if callCount == 1 {
			// The first execution returns the correct number of scores.
			scores = make([]domain.JudgeSummary, len(answers))
		} else {
			// The second execution returns the wrong number of scores.
			scores = make([]domain.JudgeSummary, len(answers)-1)
		}

		return domain.With(state, domain.KeyJudgeScores, scores), nil
	}

	middleware := NewPositionSwapMiddleware(judge, "test-middleware")

	answers := []domain.Answer{{ID: "a"}, {ID: "b"}}
	state := domain.NewState()
	state = domain.With(state, domain.KeyAnswers, answers)

	result, err := middleware.Execute(context.Background(), state)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "score count mismatch")
	assert.Equal(t, state, result)
}

// TestPositionSwapMiddleware_ConcurrentExecution tests the thread safety of the
// middleware when handling concurrent requests.
func TestPositionSwapMiddleware_ConcurrentExecution(t *testing.T) {
	biasedJudge := newBiasedMockJudge("concurrent-judge", 0.8, 0.4)
	middleware := NewPositionSwapMiddleware(biasedJudge, "concurrent-test")

	const numGoroutines = 50
	var wg sync.WaitGroup
	results := make([]domain.State, numGoroutines)
	errors := make([]error, numGoroutines)

	// Run multiple executions concurrently.
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()

			// Each goroutine uses a unique set of answers.
			answers := []domain.Answer{
				{ID: fmt.Sprintf("answer1-%d", index)},
				{ID: fmt.Sprintf("answer2-%d", index)},
			}
			state := domain.NewState()
			state = domain.With(state, domain.KeyAnswers, answers)

			result, err := middleware.Execute(context.Background(), state)
			results[index] = result
			errors[index] = err
		}(i)
	}

	wg.Wait()

	// Verify that all executions succeeded and mitigated bias correctly.
	for i := 0; i < numGoroutines; i++ {
		assert.NoError(t, errors[i], "execution %d should not fail", i)

		scores, ok := domain.Get(results[i], domain.KeyJudgeScores)
		require.True(t, ok, "execution %d should have scores", i)
		require.Len(t, scores, 2, "execution %d should have 2 scores", i)

		// Verify that bias was mitigated as expected.
		expectedScore := (0.8 + 0.4) / 2.0 // 0.6
		assert.InDelta(t, expectedScore, scores[0].Score, 0.001, "execution %d first score", i)
		assert.InDelta(t, expectedScore, scores[1].Score, 0.001, "execution %d second score", i)
	}

	// The judge should have been called twice for each goroutine.
	totalCalls := biasedJudge.getCallCount()
	assert.Equal(t, numGoroutines*2, totalCalls, "judge should be called twice per execution")
}

// TestPositionSwapMiddleware_BiasDetectionAndMitigation provides an integration-style
// test to clearly show the "before and after" effect of the middleware.
func TestPositionSwapMiddleware_BiasDetectionAndMitigation(t *testing.T) {
	// Create a strongly biased judge to serve as the control.
	stronglyBiasedJudge := newBiasedMockJudge("strongly-biased", 1.0, 0.1)

	// First, test the scenario WITHOUT the position swap middleware.
	answers := []domain.Answer{
		{ID: "answer1", Content: "Identical content"},
		{ID: "answer2", Content: "Identical content"},
	}
	state := domain.NewState()
	state = domain.With(state, domain.KeyAnswers, answers)

	// Directly execute the biased judge.
	controlResult, err := stronglyBiasedJudge.Execute(context.Background(), state)
	require.NoError(t, err)

	controlScores, ok := domain.Get(controlResult, domain.KeyJudgeScores)
	require.True(t, ok)

	// Verify that significant bias exists in the control case.
	assert.Equal(t, 1.0, controlScores[0].Score, "first answer should get high score due to bias")
	assert.Equal(t, 0.1, controlScores[1].Score, "second answer should get low score due to bias")
	biasAmount := controlScores[0].Score - controlScores[1].Score
	assert.Equal(t, 0.9, biasAmount, "significant bias should exist in the control case")

	// Now, test the scenario WITH the position swap middleware.
	middleware := NewPositionSwapMiddleware(
		newBiasedMockJudge("mitigated-biased", 1.0, 0.1),
		"bias-mitigation-test",
	)

	mitigatedResult, err := middleware.Execute(context.Background(), state)
	require.NoError(t, err)

	mitigatedScores, ok := domain.Get(mitigatedResult, domain.KeyJudgeScores)
	require.True(t, ok)

	// Verify that the bias has been successfully mitigated.
	expectedMitigatedScore := (1.0 + 0.1) / 2.0 // 0.55
	assert.InDelta(t, expectedMitigatedScore, mitigatedScores[0].Score, 0.001)
	assert.InDelta(t, expectedMitigatedScore, mitigatedScores[1].Score, 0.001)

	mitigatedBiasAmount := mitigatedScores[0].Score - mitigatedScores[1].Score
	assert.InDelta(t, 0.0, mitigatedBiasAmount, 0.001, "bias should be eliminated by position swap")

	// The bias reduction should be nearly 100%.
	biasReduction := ((biasAmount - mitigatedBiasAmount) / biasAmount) * 100
	assert.Greater(t, biasReduction, 99.0, "bias reduction should be >99%")
}

// TestPositionSwapMiddleware_AnswerOrderPreservation ensures that the original
// order of answers is preserved in the final output state.
func TestPositionSwapMiddleware_AnswerOrderPreservation(t *testing.T) {
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

	// Verify that the answer order in the output state matches the input.
	finalAnswers, ok := domain.Get(result, domain.KeyAnswers)
	require.True(t, ok)
	require.Len(t, finalAnswers, len(originalAnswers))

	for i, answer := range finalAnswers {
		assert.Equal(t, originalAnswers[i].ID, answer.ID, "answer ID order should be preserved")
		assert.Equal(t, originalAnswers[i].Content, answer.Content, "answer content should be preserved")
	}

	// Verify that scores are provided for each answer in the correct order.
	scores, ok := domain.Get(result, domain.KeyJudgeScores)
	require.True(t, ok)
	require.Len(t, scores, len(originalAnswers))
}

// BenchmarkPositionSwapMiddleware measures the performance overhead introduced
// by the dual execution strategy of the position swap middleware.
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
