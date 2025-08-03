// Package units provides domain-specific evaluation units that implement
// the ports.Unit interface for the go-gavel evaluation engine.
package units

import (
	"errors"

	"github.com/go-playground/validator/v10"
)

// TieBreaker represents the strategy for handling equal scores when multiple
// candidates have the same score during aggregation.
type TieBreaker string

// Supported tie-breaking strategies for aggregator units.
const (
	// TieFirst selects the first candidate with the tied score.
	// This provides deterministic behavior for reproducible results.
	TieFirst TieBreaker = "first"

	// TieRandom randomly selects among candidates with tied scores.
	// Uses cryptographically secure randomization for fairness.
	TieRandom TieBreaker = "random"

	// TieError returns an error when multiple candidates have tied scores.
	// Useful when tie-breaking strategy must be explicitly handled by caller.
	TieError TieBreaker = "error"
)

// Common errors returned by aggregator units.
// These errors provide consistent error handling across all aggregator implementations.
var (
	// ErrTie is returned when multiple candidates have tied scores and TieError is configured.
	ErrTie = errors.New("multiple answers tied with highest score")

	// ErrBelowMinScore is returned when the aggregate score is below the minimum threshold.
	ErrBelowMinScore = errors.New("aggregate score below minimum threshold")

	// ErrNoScores is returned when no scores are provided for aggregation.
	ErrNoScores = errors.New("no scores provided for aggregation")

	// ErrEmptyUnitName is returned when attempting to create a unit with an empty name.
	ErrEmptyUnitName = errors.New("unit name cannot be empty")

	// ErrScoreMismatch is returned when the number of scores doesn't match the number of candidates.
	ErrScoreMismatch = errors.New("scores and candidates length mismatch")
)

// Package-level validator instance for configuration validation.
// Uses go-playground/validator v10 for struct tag-based validation.
var validate = validator.New()
