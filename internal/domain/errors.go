package domain

import (
	"errors"
	"fmt"
)

// Common domain errors that can occur during evaluation operations.
var (
	// ErrInvalidState indicates that a State operation received invalid input.
	ErrInvalidState = errors.New("invalid state")

	// ErrKeyNotFound indicates that a requested StateKey does not exist.
	ErrKeyNotFound = errors.New("key not found")

	// ErrTypeMismatch indicates that a value's type doesn't match the expected type.
	ErrTypeMismatch = errors.New("type mismatch")

	// ErrEmptyValue indicates that a required value is empty or nil.
	ErrEmptyValue = errors.New("empty value")

	// ErrInvalidConfiguration indicates that configuration is invalid or incomplete.
	ErrInvalidConfiguration = errors.New("invalid configuration")

	// ErrBudgetExceeded indicates that a budget limit has been exceeded.
	ErrBudgetExceeded = errors.New("budget exceeded")
)

// StateError represents an error that occurred during State operations.
// It provides context about which key and operation caused the error.
type StateError struct {
	// Key is the state key name that was involved in the failed operation.
	Key string

	// Operation describes what operation was being performed when the error occurred.
	Operation string

	// Err is the underlying error that caused the operation to fail.
	Err error
}

// Error implements the error interface for StateError.
func (e *StateError) Error() string {
	return fmt.Sprintf("state error: operation=%s, key=%s, err=%v", e.Operation, e.Key, e.Err)
}

// Unwrap returns the underlying error, supporting Go 1.13+ error unwrapping.
func (e *StateError) Unwrap() error { return e.Err }

// NewStateError creates a new StateError with the given details.
func NewStateError(key string, operation string, err error) *StateError {
	return &StateError{
		Key:       key,
		Operation: operation,
		Err:       err,
	}
}

// ValidationError represents an error that occurred during validation.
// It can contain multiple validation failures.
type ValidationError struct {
	// Entity is the name of the entity that failed validation.
	Entity string

	// Errors contains the list of validation error messages.
	Errors []string
}

// Error implements the error interface for ValidationError.
func (e *ValidationError) Error() string {
	if len(e.Errors) == 1 {
		return fmt.Sprintf("validation error for %s: %s", e.Entity, e.Errors[0])
	}
	return fmt.Sprintf("validation errors for %s: %v", e.Entity, e.Errors)
}

// AddError adds a new error message to the validation error.
func (e *ValidationError) AddError(msg string) { e.Errors = append(e.Errors, msg) }

// HasErrors returns true if there are any validation errors.
func (e *ValidationError) HasErrors() bool { return len(e.Errors) > 0 }

// NewValidationError creates a new ValidationError for the given entity.
func NewValidationError(entity string) *ValidationError {
	return &ValidationError{
		Entity: entity,
		Errors: make([]string, 0),
	}
}

// BudgetExceededError represents an error that occurred when a budget limit
// was exceeded during evaluation execution.
// It provides detailed information about which budget was violated and by how much.
type BudgetExceededError struct {
	// LimitType indicates which type of budget was exceeded (tokens or calls).
	LimitType string

	// Limit is the configured budget limit that was exceeded.
	Limit int

	// Used is the actual amount that was attempted to be used.
	Used int

	// UnitID identifies which unit was executing when the budget was exceeded.
	UnitID string
}

// Error implements the error interface for BudgetExceededError.
func (e *BudgetExceededError) Error() string {
	return fmt.Sprintf("budget exceeded: %s limit=%d, used=%d, unit=%s",
		e.LimitType, e.Limit, e.Used, e.UnitID)
}

// Is implements error comparison for Go 1.13+ error handling.
func (e *BudgetExceededError) Is(target error) bool {
	return target == ErrBudgetExceeded
}

// NewBudgetExceededError creates a new BudgetExceededError with the given details.
func NewBudgetExceededError(limitType string, limit, used int, unitID string) *BudgetExceededError {
	return &BudgetExceededError{
		LimitType: limitType,
		Limit:     limit,
		Used:      used,
		UnitID:    unitID,
	}
}
