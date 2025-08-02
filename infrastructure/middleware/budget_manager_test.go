package middleware

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ahrav/go-gavel/internal/application"
	"github.com/ahrav/go-gavel/internal/domain"
)

// mockUnit implements ports.Unit for testing middleware functionality.
type mockUnit struct {
	name        string
	executeFunc func(ctx context.Context, state domain.State) (domain.State, error)
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

// mockBudgetObserver implements BudgetObserver for testing.
type mockBudgetObserver struct {
	preCheckCalls  []preCheckCall
	postCheckCalls []postCheckCall
	mu             sync.Mutex
}

type preCheckCall struct {
	usage  domain.Usage
	budget Budget
}

type postCheckCall struct {
	usage   domain.Usage
	budget  Budget
	elapsed time.Duration
	err     error
}

func (m *mockBudgetObserver) PreCheck(ctx context.Context, usage domain.Usage, budget Budget) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.preCheckCalls = append(m.preCheckCalls, preCheckCall{usage: usage, budget: budget})
}

func (m *mockBudgetObserver) PostCheck(ctx context.Context, usage domain.Usage, budget Budget, elapsed time.Duration, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.postCheckCalls = append(m.postCheckCalls, postCheckCall{
		usage:   usage,
		budget:  budget,
		elapsed: elapsed,
		err:     err,
	})
}

func (m *mockBudgetObserver) getCalls() ([]preCheckCall, []postCheckCall) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]preCheckCall(nil), m.preCheckCalls...), append([]postCheckCall(nil), m.postCheckCalls...)
}

func TestNewBudgetManager(t *testing.T) {
	budget := Budget{MaxTokens: 1000, MaxCalls: 10}
	nextUnit := &mockUnit{name: "test-unit"}
	observer := &mockBudgetObserver{}

	manager := NewBudgetManager(budget, nextUnit, observer)

	assert.Equal(t, budget, manager.budget)
	assert.Equal(t, nextUnit, manager.next)
	assert.Equal(t, observer, manager.observer)
}

func TestNewBudgetManager_PanicsWithNilUnit(t *testing.T) {
	budget := Budget{MaxTokens: 1000, MaxCalls: 10}

	assert.Panics(t, func() {
		NewBudgetManager(budget, nil, nil)
	})
}

func TestBudgetManager_Name(t *testing.T) {
	budget := Budget{MaxTokens: 1000, MaxCalls: 10}
	nextUnit := &mockUnit{name: "test-unit"}
	manager := NewBudgetManager(budget, nextUnit, nil)

	assert.Equal(t, "BudgetManager", manager.Name())
}

func TestBudgetManager_Validate(t *testing.T) {
	tests := []struct {
		name        string
		budget      Budget
		nextUnit    *mockUnit
		expectedErr string
	}{
		{
			name:   "valid configuration",
			budget: Budget{MaxTokens: 1000, MaxCalls: 10},
			nextUnit: &mockUnit{
				name:        "test-unit",
				validateErr: nil,
			},
			expectedErr: "",
		},
		{
			name:        "negative max tokens",
			budget:      Budget{MaxTokens: -1, MaxCalls: 10},
			nextUnit:    &mockUnit{name: "test-unit"},
			expectedErr: "max_tokens cannot be negative",
		},
		{
			name:        "negative max calls",
			budget:      Budget{MaxTokens: 1000, MaxCalls: -1},
			nextUnit:    &mockUnit{name: "test-unit"},
			expectedErr: "max_calls cannot be negative",
		},
		{
			name:   "next unit validation fails",
			budget: Budget{MaxTokens: 1000, MaxCalls: 10},
			nextUnit: &mockUnit{
				name:        "test-unit",
				validateErr: errors.New("next unit error"),
			},
			expectedErr: "next unit error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			manager := NewBudgetManager(tt.budget, tt.nextUnit, nil)
			err := manager.Validate()

			if tt.expectedErr == "" {
				assert.NoError(t, err)
			} else {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.expectedErr)
			}
		})
	}
}

func TestBudgetManager_Execute_WithinLimits(t *testing.T) {
	budget := Budget{MaxTokens: 1000, MaxCalls: 10}
	nextUnit := &mockUnit{
		name: "test-unit",
		executeFunc: func(ctx context.Context, state domain.State) (domain.State, error) {
			// Simulate token usage during execution
			return state.UpdateBudgetUsage(100, 1), nil
		},
	}
	observer := &mockBudgetObserver{}
	manager := NewBudgetManager(budget, nextUnit, observer)

	// Create state with some existing usage
	state := domain.NewState().UpdateBudgetUsage(200, 2)

	result, err := manager.Execute(context.Background(), state)

	require.NoError(t, err)

	// Verify final usage (existing + new)
	finalUsage := result.GetBudgetUsage()
	assert.Equal(t, int64(300), finalUsage.Tokens)
	assert.Equal(t, int64(3), finalUsage.Calls)

	// Verify observer calls
	preCalls, postCalls := observer.getCalls()
	assert.Len(t, preCalls, 1)
	assert.Len(t, postCalls, 1)

	// Verify pre-check called with initial usage
	assert.Equal(t, int64(200), preCalls[0].usage.Tokens)
	assert.Equal(t, int64(2), preCalls[0].usage.Calls)

	// Verify post-check called with final usage
	assert.Equal(t, int64(300), postCalls[0].usage.Tokens)
	assert.Equal(t, int64(3), postCalls[0].usage.Calls)
	assert.NoError(t, postCalls[0].err)
}

func TestBudgetManager_Execute_ExceedsTokenLimit(t *testing.T) {
	budget := Budget{MaxTokens: 100, MaxCalls: 10}
	nextUnit := &mockUnit{name: "test-unit"}
	manager := NewBudgetManager(budget, nextUnit, nil)

	// Create state that exceeds token limit
	state := domain.NewState().UpdateBudgetUsage(200, 2)

	result, err := manager.Execute(context.Background(), state)

	// Should fail before executing next unit
	assert.Error(t, err)
	var budgetErr *domain.BudgetExceededError
	assert.ErrorAs(t, err, &budgetErr)
	assert.Equal(t, "tokens", budgetErr.LimitType)
	assert.Equal(t, 100, budgetErr.Limit)
	assert.Equal(t, 200, budgetErr.Used)

	// State should be unchanged
	assert.Equal(t, state, result)
}

func TestBudgetManager_Execute_ExceedsCallLimit(t *testing.T) {
	budget := Budget{MaxTokens: 1000, MaxCalls: 5}
	nextUnit := &mockUnit{name: "test-unit"}
	manager := NewBudgetManager(budget, nextUnit, nil)

	// Create state that exceeds call limit
	state := domain.NewState().UpdateBudgetUsage(100, 10)

	result, err := manager.Execute(context.Background(), state)

	// Should fail before executing next unit
	assert.Error(t, err)
	var budgetErr *domain.BudgetExceededError
	assert.ErrorAs(t, err, &budgetErr)
	assert.Equal(t, "calls", budgetErr.LimitType)
	assert.Equal(t, 5, budgetErr.Limit)
	assert.Equal(t, 10, budgetErr.Used)

	// State should be unchanged
	assert.Equal(t, state, result)
}

func TestBudgetManager_Execute_ExceedsLimitAfterExecution(t *testing.T) {
	budget := Budget{MaxTokens: 150, MaxCalls: 10}
	nextUnit := &mockUnit{
		name: "test-unit",
		executeFunc: func(ctx context.Context, state domain.State) (domain.State, error) {
			// This will push us over the limit
			return state.UpdateBudgetUsage(100, 1), nil
		},
	}
	manager := NewBudgetManager(budget, nextUnit, nil)

	// Start with usage just under the limit
	state := domain.NewState().UpdateBudgetUsage(100, 2)

	result, err := manager.Execute(context.Background(), state)

	// Should fail after execution
	assert.Error(t, err)
	var budgetErr *domain.BudgetExceededError
	assert.ErrorAs(t, err, &budgetErr)
	assert.Equal(t, "tokens", budgetErr.LimitType)

	// State should contain the updates from execution
	finalUsage := result.GetBudgetUsage()
	assert.Equal(t, int64(200), finalUsage.Tokens)
	assert.Equal(t, int64(3), finalUsage.Calls)
}

func TestBudgetManager_Execute_WithoutObserver(t *testing.T) {
	budget := Budget{MaxTokens: 1000, MaxCalls: 10}
	nextUnit := &mockUnit{
		name: "test-unit",
		executeFunc: func(ctx context.Context, state domain.State) (domain.State, error) {
			return state.UpdateBudgetUsage(100, 1), nil
		},
	}
	manager := NewBudgetManager(budget, nextUnit, nil) // No observer

	state := domain.NewState().UpdateBudgetUsage(200, 2)

	result, err := manager.Execute(context.Background(), state)

	require.NoError(t, err)

	// Verify execution still works without observer
	finalUsage := result.GetBudgetUsage()
	assert.Equal(t, int64(300), finalUsage.Tokens)
	assert.Equal(t, int64(3), finalUsage.Calls)
}

func TestBudgetManager_Execute_NextUnitError(t *testing.T) {
	budget := Budget{MaxTokens: 1000, MaxCalls: 10}
	expectedErr := errors.New("unit execution failed")
	nextUnit := &mockUnit{
		name: "test-unit",
		executeFunc: func(ctx context.Context, state domain.State) (domain.State, error) {
			return state, expectedErr
		},
	}
	observer := &mockBudgetObserver{}
	manager := NewBudgetManager(budget, nextUnit, observer)

	state := domain.NewState().UpdateBudgetUsage(100, 1)

	result, err := manager.Execute(context.Background(), state)

	// Should return the error from next unit
	assert.Equal(t, expectedErr, err)

	// Observer should be called with the error
	_, postCalls := observer.getCalls()
	assert.Len(t, postCalls, 1)
	assert.Equal(t, expectedErr, postCalls[0].err)

	// Budget limits should not be checked after failure
	assert.Equal(t, state, result)
}

func TestBudgetManager_Execute_UnlimitedBudget(t *testing.T) {
	budget := Budget{MaxTokens: 0, MaxCalls: 0} // Unlimited
	nextUnit := &mockUnit{
		name: "test-unit",
		executeFunc: func(ctx context.Context, state domain.State) (domain.State, error) {
			return state.UpdateBudgetUsage(999999, 999999), nil
		},
	}
	manager := NewBudgetManager(budget, nextUnit, nil)

	state := domain.NewState().UpdateBudgetUsage(100000, 100000)

	result, err := manager.Execute(context.Background(), state)

	require.NoError(t, err)

	// Should allow unlimited usage
	finalUsage := result.GetBudgetUsage()
	assert.Equal(t, int64(1099999), finalUsage.Tokens)
	assert.Equal(t, int64(1099999), finalUsage.Calls)
}

func TestBudgetFromConfig(t *testing.T) {
	config := application.BudgetConfig{
		MaxTokens: 1000,
		MaxCalls:  50,
		MaxCost:   10.0, // This should be ignored
	}

	budget := BudgetFromConfig(config)

	assert.Equal(t, int64(1000), budget.MaxTokens)
	assert.Equal(t, int64(50), budget.MaxCalls)
}

// TestBudgetManager_ConcurrentExecution verifies that the budget manager
// is thread-safe and doesn't have race conditions.
func TestBudgetManager_ConcurrentExecution(t *testing.T) {
	budget := Budget{MaxTokens: 10000, MaxCalls: 1000}
	nextUnit := &mockUnit{
		name: "test-unit",
		executeFunc: func(ctx context.Context, state domain.State) (domain.State, error) {
			// Simulate some work and token usage
			time.Sleep(1 * time.Millisecond)
			return state.UpdateBudgetUsage(10, 1), nil
		},
	}
	manager := NewBudgetManager(budget, nextUnit, nil)

	const numGoroutines = 100
	var wg sync.WaitGroup
	results := make([]domain.Usage, numGoroutines)
	errors := make([]error, numGoroutines)

	// Run multiple executions concurrently
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()

			// Each goroutine starts with different initial usage
			state := domain.NewState().UpdateBudgetUsage(int64(index), int64(index))
			result, err := manager.Execute(context.Background(), state)

			results[index] = result.GetBudgetUsage()
			errors[index] = err
		}(i)
	}

	wg.Wait()

	// Verify all executions succeeded
	for i := 0; i < numGoroutines; i++ {
		assert.NoError(t, errors[i], "execution %d should not fail", i)

		// Each execution should have its own isolated budget tracking
		expectedTokens := int64(i + 10) // initial + added
		expectedCalls := int64(i + 1)   // initial + added

		assert.Equal(t, expectedTokens, results[i].Tokens, "execution %d tokens", i)
		assert.Equal(t, expectedCalls, results[i].Calls, "execution %d calls", i)
	}
}

// TestBudgetManager_ConcurrentExecutionWithLimits verifies that budget limits
// are correctly enforced even under concurrent access.
func TestBudgetManager_ConcurrentExecutionWithLimits(t *testing.T) {
	budget := Budget{MaxTokens: 100, MaxCalls: 10}
	nextUnit := &mockUnit{
		name: "test-unit",
		executeFunc: func(ctx context.Context, state domain.State) (domain.State, error) {
			return state.UpdateBudgetUsage(10, 1), nil
		},
	}
	manager := NewBudgetManager(budget, nextUnit, nil)

	const numGoroutines = 50
	var wg sync.WaitGroup
	successCount := int64(0)
	errorCount := int64(0)
	var mu sync.Mutex

	// Run multiple executions concurrently with different initial states
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()

			// Some will exceed limits, some won't
			initialTokens := int64(index * 10)
			initialCalls := int64(index)

			state := domain.NewState().UpdateBudgetUsage(initialTokens, initialCalls)
			_, err := manager.Execute(context.Background(), state)

			mu.Lock()
			if err != nil {
				errorCount++
			} else {
				successCount++
			}
			mu.Unlock()
		}(i)
	}

	wg.Wait()

	// Verify that some succeeded and some failed based on their limits
	assert.Greater(t, successCount, int64(0), "some executions should succeed")
	assert.Greater(t, errorCount, int64(0), "some executions should fail due to limits")
	assert.Equal(t, int64(numGoroutines), successCount+errorCount, "all executions should complete")
}
