// Package application provides the core business logic and orchestration for
// the evaluation engine.
package application

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ahrav/go-gavel/infrastructure/units"
	"github.com/ahrav/go-gavel/internal/domain"
	"github.com/ahrav/go-gavel/internal/testutils"
)

// TestEndToEndEvaluationPipeline tests the complete evaluation flow, from question
// to verdict. It ensures that the answerer, scoring, and aggregation units
// work together correctly to produce a final result.
func TestEndToEndEvaluationPipeline(t *testing.T) {
	ctx := context.Background()

	// Setup: Create a mock LLM client for deterministic testing.
	mockLLMClient := testutils.NewMockLLMClient("test-model-v1")

	// Create the units for the pipeline.
	answererUnit, err := units.NewAnswererUnit("answerer1", mockLLMClient, units.AnswererConfig{
		NumAnswers:     3,
		Prompt:         "Please provide a comprehensive answer to: %s",
		Temperature:    0.7,
		MaxTokens:      200,
		Timeout:        30 * time.Second,
		MaxConcurrency: 5,
	})
	require.NoError(t, err)

	scoreJudgeUnit, err := units.NewScoreJudgeUnit("judge1", mockLLMClient, units.ScoreJudgeConfig{
		JudgePrompt:    "Rate this answer to '%s': %s (Provide score and reasoning)",
		ScoreScale:     "0.0-1.0",
		Temperature:    0.5,
		MaxTokens:      150,
		MinConfidence:  0.8,
		MaxConcurrency: 5,
	})
	require.NoError(t, err)

	maxPoolUnit, err := units.NewArithmeticMeanUnit("aggregator1", units.ArithmeticMeanConfig{
		TieBreaker:       units.TieFirst,
		MinScore:         0.0,
		RequireAllScores: true,
	})
	require.NoError(t, err)

	// Test the complete pipeline.
	t.Run("complete evaluation pipeline", func(t *testing.T) {
		// Step 1: Start with an initial state containing a question.
		initialState := domain.NewState()
		question := "What are the key benefits of using microservices architecture?"
		initialState = domain.With(initialState, domain.KeyQuestion, question)

		// Step 2: Generate candidate answers.
		stateWithAnswers, err := answererUnit.Execute(ctx, initialState)
		require.NoError(t, err)

		// Verify that the answers were generated correctly.
		answers, ok := domain.Get(stateWithAnswers, domain.KeyAnswers)
		require.True(t, ok, "Answers should be present in the state")
		require.Len(t, answers, 3, "Should generate 3 answers as configured")

		// Verify the structure of the generated answers.
		for i, answer := range answers {
			assert.NotEmpty(t, answer.ID, "Answer %d should have a non-empty ID", i+1)
			assert.NotEmpty(t, answer.Content, "Answer %d should have non-empty content", i+1)
			assert.Contains(t, answer.ID, "answerer1_answer_", "Answer ID should follow the expected pattern")
		}

		// Step 3: Score the candidate answers.
		stateWithScores, err := scoreJudgeUnit.Execute(ctx, stateWithAnswers)
		require.NoError(t, err)

		// Verify that the scores were generated.
		judgeSummaries, ok := domain.Get(stateWithScores, domain.KeyJudgeScores)
		require.True(t, ok, "Judge scores should be present in the state")
		require.Len(t, judgeSummaries, 3, "Should have scores for all 3 answers")

		// Verify the score structure and requirements.
		for i, summary := range judgeSummaries {
			assert.NotEmpty(t, summary.Reasoning, "Summary %d should have reasoning", i+1)
			assert.GreaterOrEqual(t, summary.Confidence, 0.8, "Summary %d confidence should meet the minimum", i+1)
			assert.GreaterOrEqual(t, summary.Score, 0.0, "Summary %d score should be non-negative", i+1)
			assert.LessOrEqual(t, summary.Score, 1.0, "Summary %d score should not exceed 1.0", i+1)
		}

		// Step 4: Aggregate the scores to determine a winner.
		finalState, err := maxPoolUnit.Execute(ctx, stateWithScores)
		require.NoError(t, err)

		// Verify that a verdict was generated.
		verdict, ok := domain.Get(finalState, domain.KeyVerdict)
		require.True(t, ok, "A verdict should be present in the final state")
		require.NotNil(t, verdict, "The verdict should not be nil")

		// Verify the verdict structure.
		assert.NotEmpty(t, verdict.ID, "The verdict should have a non-empty ID")
		assert.NotNil(t, verdict.WinnerAnswer, "The verdict should have a winner answer")
		assert.GreaterOrEqual(t, verdict.AggregateScore, 0.0, "The aggregate score should be non-negative")
		assert.LessOrEqual(t, verdict.AggregateScore, 1.0, "The aggregate score should not exceed 1.0")

		// Verify that the winner answer is one of the generated answers.
		winnerFound := false
		for _, answer := range answers {
			if answer.ID == verdict.WinnerAnswer.ID {
				winnerFound = true
				assert.Equal(t, answer.Content, verdict.WinnerAnswer.Content,
					"Winner answer content should match the original")
				break
			}
		}
		assert.True(t, winnerFound, "The winner should be one of the original candidate answers")

		// Verify that the winner has the highest score among the judge summaries.
		// Since we are using max pooling, the winner should correspond to the highest individual score.
		maxScore := 0.0
		for _, summary := range judgeSummaries {
			if summary.Score > maxScore {
				maxScore = summary.Score
			}
		}
		assert.Equal(t, maxScore, verdict.AggregateScore,
			"The aggregate score should equal the highest individual score for max pooling")
	})

	t.Run("pipeline handles edge cases", func(t *testing.T) {
		// Test with a very short question.
		shortState := domain.NewState()
		shortState = domain.With(shortState, domain.KeyQuestion, "Why?")

		// The pipeline should still work with short inputs.
		stateAfterAnswers, err := answererUnit.Execute(ctx, shortState)
		require.NoError(t, err)

		stateAfterScores, err := scoreJudgeUnit.Execute(ctx, stateAfterAnswers)
		require.NoError(t, err)

		finalState, err := maxPoolUnit.Execute(ctx, stateAfterScores)
		require.NoError(t, err)

		// Verify that a final verdict exists.
		_, ok := domain.Get(finalState, domain.KeyVerdict)
		assert.True(t, ok, "Should produce a verdict even for short questions")
	})

	t.Run("pipeline maintains state immutability", func(t *testing.T) {
		// Test that the original state is not modified.
		originalState := domain.NewState()
		originalState = domain.With(originalState, domain.KeyQuestion, "Test question")

		// Store the original keys for comparison.
		originalKeys := originalState.Keys()

		// Execute the pipeline.
		stateAfterAnswers, err := answererUnit.Execute(ctx, originalState)
		require.NoError(t, err)

		// Verify that the original state is unchanged.
		newOriginalKeys := originalState.Keys()
		assert.Equal(t, originalKeys, newOriginalKeys, "The original state should be immutable")

		// Verify that the new state has additional data.
		newKeys := stateAfterAnswers.Keys()
		assert.Greater(t, len(newKeys), len(originalKeys), "The new state should have additional keys")
	})
}

// TestUnitRegistryIntegration tests the unit registry with the new unit types.
func TestUnitRegistryIntegration(t *testing.T) {
	ctx := context.Background()

	// Create a registry with a mock LLM client.
	mockLLMClient := testutils.NewMockLLMClient("mock-model-v1")
	registry := NewDefaultUnitRegistry(mockLLMClient)

	t.Run("registry supports new unit types", func(t *testing.T) {
		supportedTypes := registry.GetSupportedTypes()

		// Verify that the new unit types are supported.
		assert.Contains(t, supportedTypes, "answerer", "Registry should support answerer units")
		assert.Contains(t, supportedTypes, "score_judge", "Registry should support score judge units")
		assert.Contains(t, supportedTypes, "max_pool", "Registry should support max pool units")
	})

	t.Run("creates answerer unit successfully", func(t *testing.T) {
		config := map[string]any{
			"num_answers":     2,
			"prompt":          "Answer: %s",
			"temperature":     0.7,
			"max_tokens":      100,
			"timeout":         "30s",
			"max_concurrency": 5,
		}

		unit, err := registry.CreateUnit("answerer", "test_answerer", config)
		require.NoError(t, err)
		assert.Equal(t, "test_answerer", unit.Name())
		assert.NoError(t, unit.Validate())
	})

	t.Run("creates score judge unit successfully", func(t *testing.T) {
		config := map[string]any{
			"judge_prompt":   "Rate this answer for question '%s': %s (Provide detailed reasoning)",
			"score_scale":    "1-10",
			"temperature":    0.5,
			"max_tokens":     150,
			"min_confidence": 0.8,
		}

		unit, err := registry.CreateUnit("score_judge", "test_judge", config)
		require.NoError(t, err)
		assert.Equal(t, "test_judge", unit.Name())
		assert.NoError(t, unit.Validate())
	})

	t.Run("creates max pool unit successfully", func(t *testing.T) {
		config := map[string]any{
			"tie_breaker":        "first",
			"min_score":          0.0,
			"require_all_scores": true,
		}

		unit, err := registry.CreateUnit("max_pool", "test_pool", config)
		require.NoError(t, err)
		assert.Equal(t, "test_pool", unit.Name())
		assert.NoError(t, unit.Validate())
	})

	t.Run("creates units through registry and executes pipeline", func(t *testing.T) {
		// Create units through the registry.
		answerer, err := registry.CreateUnit("answerer", "pipeline_answerer", map[string]any{
			"num_answers":     2,
			"prompt":          "Provide answer to: %s",
			"temperature":     0.7,
			"max_tokens":      150,
			"timeout":         "30s",
			"max_concurrency": 5,
		})
		require.NoError(t, err)

		judge, err := registry.CreateUnit("score_judge", "pipeline_judge", map[string]any{
			"judge_prompt":   "Score this answer to '%s': %s",
			"score_scale":    "0.0-1.0",
			"temperature":    0.5,
			"max_tokens":     100,
			"min_confidence": 0.7,
		})
		require.NoError(t, err)

		aggregator, err := registry.CreateUnit("max_pool", "pipeline_aggregator", map[string]any{
			"tie_breaker": "first",
			"min_score":   0.0,
		})
		require.NoError(t, err)

		// Execute a mini-pipeline.
		state := domain.NewState()
		state = domain.With(state, domain.KeyQuestion, "What is the best programming language?")

		state, err = answerer.Execute(ctx, state)
		require.NoError(t, err)

		state, err = judge.Execute(ctx, state)
		require.NoError(t, err)

		state, err = aggregator.Execute(ctx, state)
		require.NoError(t, err)

		// Verify that the pipeline completed successfully.
		_, ok := domain.Get(state, domain.KeyVerdict)
		assert.True(t, ok, "Pipeline should produce a verdict")
	})

	t.Run("handles unit creation errors gracefully", func(t *testing.T) {
		// Test with an unsupported unit type.
		_, err := registry.CreateUnit("unsupported_type", "test_id", map[string]any{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "unsupported unit type")

		// Test with an empty unit ID.
		_, err = registry.CreateUnit("answerer", "", map[string]any{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "unit ID cannot be empty")

		// Test with an invalid configuration.
		_, err = registry.CreateUnit("answerer", "test_id", map[string]any{
			"num_answers": -1, // Invalid value.
		})
		require.Error(t, err)
	})
}

// TestDeterministicBehavior verifies that the mock LLM client produces
// consistent results for testing reliability.
func TestDeterministicBehavior(t *testing.T) {
	ctx := context.Background()

	// Create multiple instances and verify consistent behavior.
	client1 := testutils.NewMockLLMClient("test-model")
	client2 := testutils.NewMockLLMClient("test-model")

	prompt := "Generate a comprehensive response"
	options := map[string]any{"temperature": 0.5}

	// Execute the same prompt multiple times.
	result1, err1 := client1.Complete(ctx, prompt, options)
	result2, err2 := client2.Complete(ctx, prompt, options)
	result3, err3 := client1.Complete(ctx, prompt, options)

	require.NoError(t, err1)
	require.NoError(t, err2)
	require.NoError(t, err3)

	// Verify deterministic behavior.
	assert.Equal(t, result1, result2, "Different client instances should produce the same result")
	assert.Equal(t, result1, result3, "The same client should produce the same result for the same input")

	// Verify that token estimates are consistent.
	tokens1, _ := client1.EstimateTokens("test text")
	tokens2, _ := client2.EstimateTokens("test text")
	assert.Equal(t, tokens1, tokens2, "Token estimates should be consistent")
}
