// Package middleware provides cross-cutting concerns for the evaluation engine.
// It implements the middleware/wrapper pattern to keep business logic clean
// while adding security, budget enforcement, and resilience capabilities.
package middleware

import (
	"context"
	"fmt"
	"time"

	"github.com/ahrav/go-gavel/internal/application"
	"github.com/ahrav/go-gavel/internal/domain"
	"github.com/ahrav/go-gavel/internal/ports"
)

// Budget defines resource consumption limits for evaluation units.
// It specifies maximum allowed tokens and API calls to prevent runaway costs.
type Budget struct {
	// MaxTokens limits the total number of tokens that can be consumed.
	// Zero means unlimited token usage.
	MaxTokens int64

	// MaxCalls limits the total number of API calls that can be made.
	// Zero means unlimited API calls.
	MaxCalls int64
}

// BudgetObserver provides observability hooks for budget operations.
// Implementations can add tracing, metrics, and logging without
// coupling observability concerns to core budget logic.
type BudgetObserver interface {
	// PreCheck is called before budget limit validation.
	PreCheck(ctx context.Context, usage domain.Usage, budget Budget)

	// PostCheck is called after unit execution with usage and timing information.
	PostCheck(ctx context.Context, usage domain.Usage, budget Budget, elapsed time.Duration, err error)
}

// BudgetManager enforces token and API call limits during graph execution.
// It reads budget usage from request-scoped state and validates against
// configured limits without maintaining any shared mutable state.
type BudgetManager struct {
	// budget holds the immutable budget limits for this manager.
	budget Budget

	// next holds the next middleware or unit in the execution chain.
	next ports.Unit

	// observer provides optional observability hooks for tracing and metrics.
	observer BudgetObserver
}

// NewBudgetManager creates a new BudgetManager middleware instance
// with the specified budget limits, next unit, and optional observer.
// The manager is stateless and thread-safe by design.
func NewBudgetManager(budget Budget, next ports.Unit, observer BudgetObserver) *BudgetManager {
	if next == nil {
		panic("budget manager: next unit is required")
	}
	return &BudgetManager{
		budget:   budget,
		next:     next,
		observer: observer,
	}
}

// Name returns the unique identifier for this middleware.
// It is used for logging, debugging, and configuration.
func (bm *BudgetManager) Name() string { return "BudgetManager" }

// Execute performs budget enforcement around unit execution.
// It validates budget limits before and after execution while maintaining
// complete thread safety through stateless design.
func (bm *BudgetManager) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	usage := state.GetBudgetUsage()
	if err := bm.checkBudgetLimits(usage); err != nil {
		return state, err
	}

	if bm.observer != nil {
		bm.observer.PreCheck(ctx, usage, bm.budget)
	}

	start := time.Now()
	newState, err := bm.next.Execute(ctx, state)
	elapsed := time.Since(start)

	finalUsage := newState.GetBudgetUsage()
	if bm.observer != nil {
		bm.observer.PostCheck(ctx, finalUsage, bm.budget, elapsed, err)
	}

	// We check the budget again after execution to catch any usage that
	// occurred within the unit itself. This is crucial for units that
	// make multiple API calls or have complex token consumption logic.
	if err == nil {
		if budgetErr := bm.checkBudgetLimits(finalUsage); budgetErr != nil {
			return newState, budgetErr
		}
	}

	return newState, err
}

// Validate checks if the BudgetManager is properly configured.
// It verifies that budget limits are reasonable and the next unit is valid.
func (bm *BudgetManager) Validate() error {
	if bm.next == nil {
		return fmt.Errorf("budget manager: next unit is required")
	}

	if bm.budget.MaxTokens < 0 {
		return fmt.Errorf("budget manager: max_tokens cannot be negative, got %d", bm.budget.MaxTokens)
	}

	if bm.budget.MaxCalls < 0 {
		return fmt.Errorf("budget manager: max_calls cannot be negative, got %d", bm.budget.MaxCalls)
	}

	return bm.next.Validate()
}

// checkBudgetLimits verifies that current usage is within configured limits.
// It returns a BudgetExceededError if any limit is violated.
func (bm *BudgetManager) checkBudgetLimits(usage domain.Usage) error {
	if bm.budget.MaxTokens > 0 && usage.Tokens > bm.budget.MaxTokens {
		return domain.NewBudgetExceededError(
			"tokens",
			int(bm.budget.MaxTokens),
			int(usage.Tokens),
			bm.next.Name(),
		)
	}

	if bm.budget.MaxCalls > 0 && usage.Calls > bm.budget.MaxCalls {
		return domain.NewBudgetExceededError(
			"calls",
			int(bm.budget.MaxCalls),
			int(usage.Calls),
			bm.next.Name(),
		)
	}

	return nil
}

// BudgetFromConfig converts an application.BudgetConfig to a middleware.Budget.
// It simplifies creating Budget instances from loaded application configuration.
func BudgetFromConfig(config application.BudgetConfig) Budget {
	return Budget{
		MaxTokens: config.MaxTokens,
		MaxCalls:  config.MaxCalls,
	}
}
