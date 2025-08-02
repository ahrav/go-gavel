package testutils

import (
	"github.com/go-playground/validator/v10"
)

// NewTestValidator creates a new validator instance for testing.
// This provides a consistent validator configuration across all tests.
func NewTestValidator() *validator.Validate {
	return validator.New()
}
