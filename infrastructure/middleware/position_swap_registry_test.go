// Package middleware_test contains the unit tests for the middleware package.
package middleware

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ahrav/go-gavel/internal/domain"
	"github.com/ahrav/go-gavel/internal/testutils"
)

// mockPosSwapUnit implements the ports.Unit interface for testing.
type mockPosSwapUnit struct {
	name string
}

func (m *mockPosSwapUnit) Name() string { return m.name }

func (m *mockPosSwapUnit) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	// Add mock judge scores for the PositionSwap middleware to combine
	answers, _ := domain.Get(state, domain.KeyAnswers)
	judgeScores := make([]domain.JudgeSummary, len(answers))
	for i := range answers {
		judgeScores[i] = domain.JudgeSummary{
			Score:      0.8, // Mock score
			Confidence: 0.9,
			Reasoning:  "Mock reasoning",
		}
	}

	// Add judge scores and mark execution
	result := domain.With(state, domain.KeyJudgeScores, judgeScores)
	return domain.With(result, domain.NewKey[bool]("executed_"+m.name), true), nil
}

func (m *mockPosSwapUnit) Validate() error { return nil }

// TestPositionSwapMiddleware_DirectCreation tests that the PositionSwap middleware
// can be created directly with our new simplified registry pattern.
func TestPositionSwapMiddleware_DirectCreation(t *testing.T) {
	// Create a properly configured mock LLM client for the registry.
	mockLLMClient := testutils.NewMockLLMClient("gpt-4")

	// Create a mock unit to wrap
	mockUnit := &mockPosSwapUnit{
		name: "mock_judge",
	}

	// Create the middleware directly using the factory function
	config := map[string]any{
		"wrapped_unit": mockUnit, // Pass the unit instance directly
	}

	// Create the PositionSwap wrapper using the factory.
	unit, err := NewPositionSwapFromConfig("judge_with_position_swap", config, mockLLMClient)
	require.NoError(t, err, "should create PositionSwap wrapper successfully")
	require.NotNil(t, unit, "created unit should not be nil")

	// Verify that the created unit is indeed a PositionSwapMiddleware instance.
	positionSwapMiddleware, ok := unit.(*PositionSwapMiddleware)
	require.True(t, ok, "created unit should be PositionSwapMiddleware")
	assert.Equal(t, "judge_with_position_swap", positionSwapMiddleware.Name())

	// Test that the created middleware instance is valid.
	err = positionSwapMiddleware.Validate()
	assert.NoError(t, err, "middleware should validate successfully")
}

// TestPositionSwapMiddleware_ConfigurationErrors tests the error handling
// for invalid configurations when creating a PositionSwap wrapper.
func TestPositionSwapMiddleware_ConfigurationErrors(t *testing.T) {
	mockLLMClient := testutils.NewMockLLMClient("gpt-4")

	tests := []struct {
		name        string
		config      map[string]any
		expectedErr string
	}{
		{
			name:        "missing wrapped_unit",
			config:      map[string]any{},
			expectedErr: "requires 'wrapped_unit'",
		},
		{
			name: "wrapped_unit is not a Unit",
			config: map[string]any{
				"wrapped_unit": "invalid",
			},
			expectedErr: "requires 'wrapped_unit' as a Unit instance",
		},
		{
			name: "wrapped_unit is a map (old style)",
			config: map[string]any{
				"wrapped_unit": map[string]any{
					"id":   "test",
					"type": "score_judge",
				},
			},
			expectedErr: "requires 'wrapped_unit' as a Unit instance",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			unit, err := NewPositionSwapFromConfig("test_wrapper", tt.config, mockLLMClient)

			assert.Error(t, err)
			assert.Contains(t, err.Error(), tt.expectedErr)
			assert.Nil(t, unit)
		})
	}
}

// TestPositionSwapMiddleware_WrappedUnitExecution tests that the middleware
// correctly wraps and executes the underlying unit.
func TestPositionSwapMiddleware_WrappedUnitExecution(t *testing.T) {
	// Create a mock unit to wrap
	mockUnit := &mockPosSwapUnit{
		name: "mock_judge",
	}

	// Create the middleware with the mock unit
	config := map[string]any{
		"wrapped_unit": mockUnit,
	}

	middleware, err := NewPositionSwapFromConfig("wrapper", config, nil)
	require.NoError(t, err)

	// Create a state with some test answers
	state := domain.NewState()
	answers := []domain.Answer{
		{ID: "answer1", Content: "First answer"},
		{ID: "answer2", Content: "Second answer"},
	}
	state = domain.With(state, domain.KeyAnswers, answers)

	// Execute the middleware (it will execute the wrapped unit twice)
	ctx := context.Background()
	result, err := middleware.Execute(ctx, state)
	require.NoError(t, err)

	// Verify that the wrapped unit was executed
	executed, ok := domain.Get(result, domain.NewKey[bool]("executed_mock_judge"))
	assert.True(t, ok, "wrapped unit should have been executed")
	assert.True(t, executed, "execution marker should be true")
}
