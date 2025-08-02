package ports

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ahrav/go-gavel/internal/domain"
)

// mockUnit is a test implementation of the Unit interface
type mockUnit struct {
	name        string
	executeFunc func(context.Context, domain.State) (domain.State, error)
	validateErr error
}

func (m *mockUnit) Name() string {
	return m.name
}

func (m *mockUnit) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	if m.executeFunc != nil {
		return m.executeFunc(ctx, state)
	}
	return state, nil
}

func (m *mockUnit) Validate() error {
	return m.validateErr
}

func TestUnit_Interface(t *testing.T) {
	// Verify mockUnit implements Unit interface
	var _ Unit = (*mockUnit)(nil)

	// Test basic unit behavior
	unit := &mockUnit{
		name: "test-unit",
		executeFunc: func(ctx context.Context, state domain.State) (domain.State, error) {
			// Add a test value to state
			return state.With(domain.StateKey("test"), "value"), nil
		},
		validateErr: nil,
	}

	// Test Name()
	assert.Equal(t, "test-unit", unit.Name(), "Name() mismatch")

	// Test Validate()
	assert.NoError(t, unit.Validate(), "Validate() should not return error")

	// Test Execute()
	ctx := context.Background()
	initialState := domain.NewState()

	newState, err := unit.Execute(ctx, initialState)
	require.NoError(t, err, "Execute() should not return error")

	// Verify state was modified
	val, ok := newState.Get(domain.StateKey("test"))
	require.True(t, ok, "Execute() should add test key to state")
	v, ok := val.(string)
	require.True(t, ok, "Value should be string")
	assert.Equal(t, "value", v, "Execute() state value mismatch")

	// Verify original state unchanged
	_, ok = initialState.Get(domain.StateKey("test"))
	assert.False(t, ok, "Execute() should not modify original state")
}

func TestUnit_ValidationFailure(t *testing.T) {
	validationErr := errors.New("invalid configuration")
	unit := &mockUnit{
		name:        "failing-unit",
		validateErr: validationErr,
	}

	err := unit.Validate()
	assert.Equal(t, validationErr, err, "Validate() error mismatch")
}

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

	// Test with cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	state := domain.NewState()
	_, err := unit.Execute(ctx, state)
	assert.Equal(t, context.Canceled, err, "Execute() with cancelled context should return context.Canceled")
}
