// Package middleware_test contains the unit tests for the middleware package.
package middleware

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ahrav/go-gavel/internal/application"
	"github.com/ahrav/go-gavel/internal/testutils"
)

// TestPositionSwapMiddleware_RegistryIntegration tests that the PositionSwap middleware
// can be successfully registered with and created by the UnitRegistry.
func TestPositionSwapMiddleware_RegistryIntegration(t *testing.T) {
	// Create a properly configured mock LLM client for the registry.
	mockLLMClient := testutils.NewMockLLMClient("gpt-4")

	// Create a new unit registry.
	registry := application.NewDefaultUnitRegistry(mockLLMClient)

	// Register the PositionSwap middleware factory with the registry.
	err := RegisterPositionSwapMiddleware(registry)
	require.NoError(t, err, "middleware registration should succeed")

	// Verify that position_swap_wrapper is now a supported type.
	supportedTypes := registry.GetSupportedTypes()
	assert.Contains(t, supportedTypes, "position_swap_wrapper",
		"position_swap_wrapper should be in supported types after registration")

	// Define a valid configuration for creating a PositionSwap wrapper.
	config := map[string]any{
		"wrapped_unit": map[string]any{
			"id":   "score_judge",
			"type": "score_judge",
			"params": map[string]any{
				"model":           "gpt-4",
				"prompt_template": "judge_correctness.tmpl",
			},
		},
	}

	// Create the PositionSwap wrapper through the registry.
	unit, err := registry.CreateUnit("position_swap_wrapper", "judge_with_position_swap", config)
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

// TestPositionSwapMiddleware_RegistryConfigurationErrors tests the error handling
// for invalid configurations when creating a PositionSwap wrapper from the registry.
func TestPositionSwapMiddleware_RegistryConfigurationErrors(t *testing.T) {
	mockLLMClient := testutils.NewMockLLMClient("gpt-4")
	registry := application.NewDefaultUnitRegistry(mockLLMClient)

	// Register the middleware to enable testing its creation.
	err := RegisterPositionSwapMiddleware(registry)
	require.NoError(t, err)

	tests := []struct {
		name        string
		config      map[string]any
		expectedErr string
	}{
		{
			name:        "missing wrapped_unit",
			config:      map[string]any{},
			expectedErr: "requires 'wrapped_unit' configuration",
		},
		{
			name: "wrapped_unit is not a map",
			config: map[string]any{
				"wrapped_unit": "invalid",
			},
			expectedErr: "wrapped_unit must be a configuration object",
		},
		{
			name: "wrapped_unit missing type",
			config: map[string]any{
				"wrapped_unit": map[string]any{
					"id": "test",
				},
			},
			expectedErr: "wrapped_unit must have a 'type' field",
		},
		{
			name: "wrapped_unit missing id",
			config: map[string]any{
				"wrapped_unit": map[string]any{
					"type": "score_judge",
				},
			},
			expectedErr: "wrapped_unit must have an 'id' field",
		},
		{
			name: "invalid wrapped unit type",
			config: map[string]any{
				"wrapped_unit": map[string]any{
					"id":   "test",
					"type": "nonexistent_unit_type",
				},
			},
			expectedErr: "unsupported unit type",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			unit, err := registry.CreateUnit("position_swap_wrapper", "test_wrapper", tt.config)

			assert.Error(t, err)
			assert.Contains(t, err.Error(), tt.expectedErr)
			assert.Nil(t, unit)
		})
	}
}

// TestPositionSwapMiddleware_RegistryDoubleRegistration tests that registering
// the middleware factory multiple times does not cause issues.
func TestPositionSwapMiddleware_RegistryDoubleRegistration(t *testing.T) {
	mockLLMClient := testutils.NewMockLLMClient("gpt-4")
	registry := application.NewDefaultUnitRegistry(mockLLMClient)

	// Register the middleware twice in a row.
	err1 := RegisterPositionSwapMiddleware(registry)
	require.NoError(t, err1, "first registration should succeed")

	err2 := RegisterPositionSwapMiddleware(registry)
	require.NoError(t, err2, "second registration should also succeed (idempotent)")

	// Verify that the type is still supported.
	supportedTypes := registry.GetSupportedTypes()
	assert.Contains(t, supportedTypes, "position_swap_wrapper")

	// Verify that creating the wrapper still works correctly.
	config := map[string]any{
		"wrapped_unit": map[string]any{
			"id":   "score_judge",
			"type": "score_judge",
		},
	}

	unit, err := registry.CreateUnit("position_swap_wrapper", "test", config)
	require.NoError(t, err)
	require.NotNil(t, unit)

	_, ok := unit.(*PositionSwapMiddleware)
	assert.True(t, ok, "should still create a PositionSwapMiddleware instance")
}
