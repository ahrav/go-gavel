package domain

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestStateError(t *testing.T) {
	tests := []struct {
		name      string
		key       string
		operation string
		err       error
		wantMsg   string
	}{
		{
			name:      "basic state error",
			key:       KeyQuestion.name,
			operation: "Get",
			err:       ErrKeyNotFound,
			wantMsg:   "state error: operation=Get, key=question, err=key not found",
		},
		{
			name:      "with wrapped error",
			key:       KeyAnswers.name,
			operation: "With",
			err:       ErrTypeMismatch,
			wantMsg:   "state error: operation=With, key=answers, err=type mismatch",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := NewStateError(tt.key, tt.operation, tt.err)

			assert.Equal(t, tt.wantMsg, err.Error(), "Error message mismatch")
			assert.Equal(t, tt.key, err.Key, "Key mismatch")
			assert.Equal(t, tt.operation, err.Operation, "Operation mismatch")

			// Test error unwrapping
			assert.True(t, errors.Is(err, tt.err), "Should unwrap to underlying error")
		})
	}
}

func TestValidationError(t *testing.T) {
	t.Run("single error", func(t *testing.T) {
		err := NewValidationError("Unit")
		err.AddError("missing configuration")

		assert.Equal(t, "validation error for Unit: missing configuration", err.Error())
		assert.True(t, err.HasErrors(), "Should have errors")
		assert.Len(t, err.Errors, 1, "Should have one error")
	})

	t.Run("multiple errors", func(t *testing.T) {
		err := NewValidationError("Pipeline")
		err.AddError("invalid units")
		err.AddError("missing dependencies")
		err.AddError("circular reference detected")

		assert.Contains(t, err.Error(), "validation errors for Pipeline")
		assert.True(t, err.HasErrors(), "Should have errors")
		assert.Len(t, err.Errors, 3, "Should have three errors")
	})

	t.Run("no errors", func(t *testing.T) {
		err := NewValidationError("Config")

		assert.False(t, err.HasErrors(), "Should not have errors")
		assert.Empty(t, err.Errors, "Errors slice should be empty")
	})
}

func TestCommonDomainErrors(t *testing.T) {
	// Test that common errors are defined and have expected messages
	tests := []struct {
		err     error
		message string
	}{
		{ErrInvalidState, "invalid state"},
		{ErrKeyNotFound, "key not found"},
		{ErrTypeMismatch, "type mismatch"},
		{ErrEmptyValue, "empty value"},
		{ErrInvalidConfiguration, "invalid configuration"},
	}

	for _, tt := range tests {
		t.Run(tt.message, func(t *testing.T) {
			assert.Equal(t, tt.message, tt.err.Error(), "Error message mismatch")
		})
	}
}

func TestErrorWrapping(t *testing.T) {
	// Test Go 1.13+ error wrapping features
	baseErr := errors.New("base error")
	stateErr := NewStateError(KeyQuestion.name, "Test", baseErr)

	// Test Is functionality
	assert.True(t, errors.Is(stateErr, baseErr), "Should match base error with Is")

	// Test Unwrap functionality
	unwrapped := errors.Unwrap(stateErr)
	assert.Equal(t, baseErr, unwrapped, "Should unwrap to base error")

	// Test wrapping with standard library errors
	wrappedErr := NewStateError(KeyAnswers.name, "Process", ErrKeyNotFound)
	assert.True(t, errors.Is(wrappedErr, ErrKeyNotFound), "Should match domain error")
}

func TestValidationErrorAccumulation(t *testing.T) {
	err := NewValidationError("TestEntity")

	// Add errors incrementally
	assert.False(t, err.HasErrors(), "Should start with no errors")

	err.AddError("first error")
	assert.True(t, err.HasErrors(), "Should have errors after adding one")
	assert.Len(t, err.Errors, 1, "Should have one error")

	err.AddError("second error")
	assert.Len(t, err.Errors, 2, "Should have two errors")

	// Verify all errors are preserved
	assert.Equal(t, "first error", err.Errors[0], "First error should be preserved")
	assert.Equal(t, "second error", err.Errors[1], "Second error should be preserved")
}
