package ports

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ahrav/go-gavel/internal/domain"
)

// testKey is a domain key used for setting and retrieving values in unit tests.
var testKey = domain.NewKey[string]("test")

// mockUnit is a mock implementation of the Unit interface for testing purposes.
// It allows injecting custom logic for Execute and Validate methods.
type mockUnit struct {
	name        string
	executeFunc func(context.Context, domain.State) (domain.State, error)
	validateErr error
}

// Name returns the configured name of the mock unit.
func (m *mockUnit) Name() string {
	return m.name
}

// Execute runs the custom execute function if provided, otherwise returns the state unmodified.
func (m *mockUnit) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	if m.executeFunc != nil {
		return m.executeFunc(ctx, state)
	}
	return state, nil
}

// Validate returns the predefined validation error.
func (m *mockUnit) Validate() error {
	return m.validateErr
}

// TestUnit_Interface verifies that the mockUnit correctly implements the Unit interface.
// It also tests the basic behavior of a unit, including name retrieval, validation, and execution.
func TestUnit_Interface(t *testing.T) {
	var _ Unit = (*mockUnit)(nil)

	unit := &mockUnit{
		name: "test-unit",
		executeFunc: func(ctx context.Context, state domain.State) (domain.State, error) {
			return domain.With(state, testKey, "value"), nil
		},
		validateErr: nil,
	}

	assert.Equal(t, "test-unit", unit.Name(), "Name() mismatch")
	assert.NoError(t, unit.Validate(), "Validate() should not return an error")

	ctx := context.Background()
	initialState := domain.NewState()

	newState, err := unit.Execute(ctx, initialState)
	require.NoError(t, err, "Execute() should not return an error")

	val, ok := domain.Get(newState, testKey)
	require.True(t, ok, "Execute() should add the test key to the state")
	assert.Equal(t, "value", val, "Execute() state value mismatch")

	_, ok = domain.Get(initialState, testKey)
	assert.False(t, ok, "Execute() should not modify the original state")
}

// TestUnit_ValidationFailure tests that the Validate method correctly returns a predefined error.
// This ensures that validation failures are properly propagated.
func TestUnit_ValidationFailure(t *testing.T) {
	validationErr := errors.New("invalid configuration")
	unit := &mockUnit{
		name:        "failing-unit",
		validateErr: validationErr,
	}

	err := unit.Validate()
	assert.Equal(t, validationErr, err, "Validate() error mismatch")
}

// TestUnit_ExecutionFailure tests that the Execute method correctly returns an error from the custom execute function.
// This verifies that execution errors are handled as expected.
func TestUnit_ExecutionFailure(t *testing.T) {
	execErr := errors.New("execution failed")
	unit := &mockUnit{
		name: "failing-unit",
		executeFunc: func(ctx context.Context, state domain.State) (domain.State, error) {
			return domain.State{}, execErr
		},
	}

	ctx := context.Background()
	state := domain.NewState()

	_, err := unit.Execute(ctx, state)
	assert.Equal(t, execErr, err, "Execute() error mismatch")
}

// TestUnit_ContextCancellation ensures that a unit's Execute method respects context cancellation.
// It verifies that if the context is canceled, the execution is aborted and the appropriate error is returned.
func TestUnit_ContextCancellation(t *testing.T) {
	unit := &mockUnit{
		name: "context-aware-unit",
		executeFunc: func(ctx context.Context, state domain.State) (domain.State, error) {
			select {
			case <-ctx.Done():
				return domain.State{}, ctx.Err()
			default:
				return state, nil
			}
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	state := domain.NewState()
	_, err := unit.Execute(ctx, state)
	assert.Equal(t, context.Canceled, err, "Execute() with a canceled context should return context.Canceled")
}
