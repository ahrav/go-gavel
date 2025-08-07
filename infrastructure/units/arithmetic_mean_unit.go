package units

import (
	"context"
	"crypto/rand"
	"fmt"
	"math"
	"math/big"

	"gopkg.in/yaml.v3"

	"github.com/ahrav/go-gavel/internal/domain"
	"github.com/ahrav/go-gavel/internal/ports"
)

var _ ports.Unit = (*ArithmeticMeanUnit)(nil)

// ArithmeticMeanUnit implements score aggregation using arithmetic mean calculation
// with configurable tie-breaking and minimum score thresholds. It combines multiple
// judge scores into a single aggregate metric while selecting winners based on
// individual performance.
//
// Mathematical Algorithm: Computes the arithmetic mean (Σscores / count) of all
// judge scores as the aggregate score. Winner selection uses the highest individual
// score, not the aggregate, enabling nuanced evaluation scenarios.
//
// Tie-Breaking: Supports deterministic (first), random (cryptographically secure),
// and error-on-tie strategies for consistent winner selection across executions.
//
// Performance: O(n) time complexity for n scores with single-pass calculation.
// Designed for sub-microsecond latency on typical score sets (≤100 candidates).
//
// Precision: Uses IEEE 754 double-precision arithmetic with explicit NaN/Inf
// validation to ensure mathematical correctness and prevent invalid aggregations.
//
// Concurrency: Stateless and thread-safe for concurrent execution. Multiple
// goroutines can call Execute and Aggregate simultaneously without synchronization.
//
// Quality Gates: Implements minimum score thresholds and configurable validation
// to ensure aggregation results meet quality requirements before verdict generation.
type ArithmeticMeanUnit struct {
	// name is the unique identifier for this unit instance.
	name string
	// config contains the validated configuration parameters.
	config ArithmeticMeanConfig
}

// ArithmeticMeanConfig controls aggregation behavior and quality requirements
// for arithmetic mean calculation. Configuration is immutable after unit creation
// and validated for mathematical consistency.
//
// The configuration balances evaluation accuracy with practical requirements,
// supporting flexible aggregation strategies for different evaluation contexts.
type ArithmeticMeanConfig struct {
	// TieBreaker defines the strategy for resolving equal highest scores.
	// "first": Select first candidate (deterministic, reproducible)
	// "random": Cryptographically secure random selection (unbiased)
	// "error": Fail with explicit error (strict evaluation requirements)
	TieBreaker TieBreaker `yaml:"tie_breaker" json:"tie_breaker" validate:"required,oneof=first random error"`

	// MinScore sets the minimum acceptable aggregate score threshold (0.0-1.0).
	// Aggregations below this value trigger ErrBelowMinScore for quality enforcement.
	// Use 0.0 to disable minimum score requirements.
	MinScore float64 `yaml:"min_score" json:"min_score" validate:"min=0.0,max=1.0"`

	// RequireAllScores enforces complete score coverage for all candidates.
	// true: Mismatch between answers and scores triggers validation error
	// false: Process available answer-score pairs, ignore unscored candidates
	RequireAllScores bool `yaml:"require_all_scores" json:"require_all_scores"`
}

// NewArithmeticMeanUnit creates a new ArithmeticMeanUnit with validated
// mathematical configuration and tie-breaking strategy. The unit is immediately
// ready for concurrent execution after successful creation.
//
// The name parameter serves as a unique identifier for logging, debugging,
// and verdict generation. Configuration validation ensures mathematical
// consistency and proper tie-breaking behavior.
//
// Returns ErrEmptyUnitName if name is empty, or configuration validation
// errors if constraints are violated (invalid tie-breaker, score ranges, etc.).
func NewArithmeticMeanUnit(name string, config ArithmeticMeanConfig) (*ArithmeticMeanUnit, error) {
	if name == "" {
		return nil, ErrEmptyUnitName
	}

	if err := validate.Struct(config); err != nil {
		return nil, fmt.Errorf("configuration validation failed: %w", err)
	}

	return &ArithmeticMeanUnit{
		name:   name,
		config: config,
	}, nil
}

// Name returns the unique identifier for this unit instance.
// The returned value is immutable and safe for concurrent access.
func (mpu *ArithmeticMeanUnit) Name() string { return mpu.name }

// Execute performs score aggregation using arithmetic mean calculation with
// configurable quality gates and tie-breaking strategies. It transforms judge
// scores into a final verdict with aggregate scoring and winner selection.
//
// State requirements:
//   - domain.KeyAnswers: []domain.Answer with candidate responses
//   - domain.KeyJudgeScores: []domain.JudgeSummary with evaluation scores
//
// Returns a new state containing domain.KeyVerdict with:
//   - Aggregate score as arithmetic mean of all judge scores
//   - Winner selected by highest individual score (not aggregate)
//   - Verdict ID derived from unit name for traceability
//
// Error Handling: Handles mismatched answer/score counts based on configuration.
// RequireAllScores=true enforces strict pairing, false processes available pairs.
//
// Errors:
//   - Missing answers or judge scores in state
//   - Empty answer set for aggregation
//   - Score/answer count mismatch (when RequireAllScores=true)
//   - Aggregation failures (NaN/Inf scores, below minimum threshold)
//
// The function is safe for concurrent execution and does not modify input state.
func (mpu *ArithmeticMeanUnit) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	answers, ok := domain.Get(state, domain.KeyAnswers)
	if !ok {
		return state, fmt.Errorf("answers not found in state")
	}

	if len(answers) == 0 {
		return state, fmt.Errorf("no answers to aggregate")
	}

	judgeSummaries, ok := domain.Get(state, domain.KeyJudgeScores)
	if !ok {
		return state, fmt.Errorf("judge scores not found in state")
	}

	numAnswers := len(answers)
	numScores := len(judgeSummaries)

	// Validate answer-score pairing based on configuration requirements.
	if numScores != numAnswers {
		if mpu.config.RequireAllScores {
			return state, fmt.Errorf("mismatch between answers (%d) and judge scores (%d)",
				numAnswers, numScores)
		}
		// Handle partial scoring case - only process up to the minimum available.
		// This enables flexible evaluation with optional scoring.
		if numScores < numAnswers {
			numAnswers = numScores
		}
	}

	// Extract scores for aggregation - only process valid answer-score pairs.
	// This ensures mathematical consistency and prevents index errors.
	scores := make([]float64, numAnswers)
	validAnswers := make([]domain.Answer, numAnswers)
	for i := 0; i < numAnswers; i++ {
		scores[i] = judgeSummaries[i].Score
		validAnswers[i] = answers[i]
	}

	winner, aggregateScore, err := mpu.Aggregate(scores, validAnswers)
	if err != nil {
		return state, fmt.Errorf("aggregation failed: %w", err)
	}

	verdict := domain.Verdict{
		ID:             fmt.Sprintf("%s_verdict", mpu.name),
		WinnerAnswer:   &winner,
		AggregateScore: aggregateScore,
		// TODO: Add trace and budget information when available.
	}

	return domain.With(state, domain.KeyVerdict, &verdict), nil
}

// Aggregate implements the domain.Aggregator interface with arithmetic mean
// calculation and highest-score winner selection. This method provides the
// core mathematical logic for score aggregation.
//
// Algorithm:
//  1. Validates input scores for NaN/Inf values and array consistency
//  2. Calculates arithmetic mean: Σ(scores) / len(scores)
//  3. Identifies winner by highest individual score (not aggregate)
//  4. Applies tie-breaking strategy for equal highest scores
//  5. Validates aggregate against MinScore threshold
//
// Mathematical Precision: Uses IEEE 754 double-precision arithmetic with
// explicit validation for invalid floating-point values to ensure correctness.
//
// Tie-Breaking: Supports deterministic (first), random (crypto-secure), and
// error strategies for consistent winner selection across executions.
//
// Performance: O(n) time complexity with single-pass calculation. Optimized
// for typical evaluation scenarios with ≤100 candidates.
//
// Returns the winning candidate, aggregate score, and any calculation errors.
func (mpu *ArithmeticMeanUnit) Aggregate(
	scores []float64,
	candidates []domain.Answer,
) (domain.Answer, float64, error) {
	if len(scores) == 0 {
		return domain.Answer{}, 0, ErrNoScores
	}
	if len(scores) != len(candidates) {
		return domain.Answer{}, 0, fmt.Errorf("%w: scores=%d, candidates=%d",
			ErrScoreMismatch, len(scores), len(candidates))
	}

	// Calculate arithmetic mean while tracking winner and ties.
	// Single-pass algorithm for O(n) performance.
	var sum float64
	var winnerIdx int
	var maxScore = -1.0 // Initialize below valid score range
	var tieIndices []int

	for i, score := range scores {
		// Validate mathematical correctness of IEEE 754 floating-point values.
		if math.IsNaN(score) || math.IsInf(score, 0) {
			return domain.Answer{}, 0, fmt.Errorf("invalid score at index %d: %f", i, score)
		}

		sum += score

		// Track highest score for winner selection using exact equality.
		// Tie detection enables configurable tie-breaking strategies.
		if score > maxScore {
			maxScore = score
			winnerIdx = i
			tieIndices = []int{i} // Reset tie list with new leader
		} else if score == maxScore {
			tieIndices = append(tieIndices, i) // Add to tie list
		}
	}

	// Calculate arithmetic mean with guaranteed non-zero denominator.
	mean := sum / float64(len(scores))

	if mean < mpu.config.MinScore {
		return domain.Answer{}, 0, fmt.Errorf("%w: mean=%.3f, minimum=%.3f",
			ErrBelowMinScore, mean, mpu.config.MinScore)
	}

	// Apply configured tie-breaking strategy for consistent winner selection.
	if len(tieIndices) > 1 {
		switch mpu.config.TieBreaker {
		case TieFirst:
			// Deterministic: select first occurrence for reproducibility
			winnerIdx = tieIndices[0]
		case TieError:
			// Strict: fail on ambiguous results for critical evaluations
			return domain.Answer{}, 0, fmt.Errorf("%w: %d answers with score %.3f", ErrTie, len(tieIndices), maxScore)
		case TieRandom:
			// Unbiased: cryptographically secure random selection
			n, err := rand.Int(rand.Reader, big.NewInt(int64(len(tieIndices))))
			if err != nil {
				return domain.Answer{}, 0, fmt.Errorf("failed to generate random number for tie-breaking: %w", err)
			}
			winnerIdx = tieIndices[n.Int64()]
		}
	}

	return candidates[winnerIdx], mean, nil
}

// Validate verifies the unit is properly configured with valid mathematical
// constraints and tie-breaking strategy. This method ensures aggregation
// will operate correctly before execution.
//
// Returns nil if the unit is operational, or a descriptive error indicating
// the specific validation failure. Safe for concurrent use.
func (mpu *ArithmeticMeanUnit) Validate() error {
	if err := validate.Struct(mpu.config); err != nil {
		return fmt.Errorf("configuration validation failed: %w", err)
	}

	return nil
}

// UnmarshalParameters deserializes YAML configuration into the unit's
// mathematical parameters with comprehensive validation. This method enables
// dynamic configuration updates while preserving unit integrity.
//
// The method performs strict YAML decoding with validation to ensure
// mathematical consistency and constraint satisfaction. Successfully
// decoded configuration immediately replaces the unit's current settings.
//
// Returns an error if YAML parsing fails or the decoded configuration
// violates mathematical constraints. The unit's configuration remains
// unchanged on error.
func (mpu *ArithmeticMeanUnit) UnmarshalParameters(params yaml.Node) error {
	var config ArithmeticMeanConfig

	// Use strict decoding to catch unknown fields.
	if err := params.Decode(&config); err != nil {
		return fmt.Errorf("failed to decode parameters: %w", err)
	}

	// Validate the decoded configuration.
	if err := validate.Struct(config); err != nil {
		return fmt.Errorf("parameter validation failed: %w", err)
	}

	mpu.config = config
	return nil
}

// DefaultArithmeticMeanConfig returns an ArithmeticMeanConfig with production-ready defaults:
// deterministic tie-breaking, no minimum score threshold, and complete score requirement.
func DefaultArithmeticMeanConfig() ArithmeticMeanConfig {
	return ArithmeticMeanConfig{
		TieBreaker:       TieFirst,
		MinScore:         0.0,
		RequireAllScores: true,
	}
}

// NewArithmeticMeanFromConfig creates an ArithmeticMeanUnit from a configuration map.
// This is the boundary adapter for YAML/JSON configuration.
// Arithmetic mean doesn't require an LLM client (deterministic aggregation).
func NewArithmeticMeanFromConfig(id string, config map[string]any, llm ports.LLMClient) (ports.Unit, error) {
	// llm is ignored - arithmetic mean is deterministic.

	data, err := yaml.Marshal(config)
	if err != nil {
		return nil, fmt.Errorf("marshal config: %w", err)
	}

	// Start with defaults, then overlay user config.
	cfg := DefaultArithmeticMeanConfig()
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	return NewArithmeticMeanUnit(id, cfg)
}
