// Package ports defines the core interfaces that form the contract between
// the domain/application layers and the infrastructure layer.
// These interfaces enable dependency inversion and make the system testable.
package ports

import (
	"context"

	"github.com/ahrav/go-gavel/internal/domain"
)

// Unit represents the fundamental building block of the evaluation pipeline.
// Each Unit performs a specific transformation on the evaluation State,
// enabling composable and reusable evaluation logic.
// Units should be stateless and thread-safe for concurrent execution.
type Unit interface {
	// Name returns a unique identifier for this unit.
	// The name is used for logging, debugging, and configuration.
	Name() string

	// Execute performs the unit's transformation on the provided State.
	// It returns a new State containing the results of the transformation.
	// The original State should not be modified (immutability principle).
	// Any errors during execution should be returned rather than panicking.
	//
	// The context parameter allows for cancellation and deadline propagation.
	// Units should respect context cancellation and return promptly.
	//
	// Example:
	//
	//	newState, err := unit.Execute(ctx, state)
	//	if err != nil {
	//	    return nil, fmt.Errorf("unit %s failed: %w", unit.Name(), err)
	//	}
	Execute(ctx context.Context, state domain.State) (domain.State, error)

	// Validate checks if the unit is properly configured and ready for execution.
	// This method should verify all required dependencies and configuration.
	// It is typically called during pipeline construction or before execution.
	// Return nil if validation passes, or an error describing what is invalid.
	//
	// Example validations:
	// - Required configuration parameters are set
	// - Dependencies (like LLM clients) are available
	// - Resource limits are reasonable
	Validate() error
}
