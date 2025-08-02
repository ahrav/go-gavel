package units

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"math"
	"math/big"

	"github.com/go-playground/validator/v10"
	"gopkg.in/yaml.v3"

	"github.com/ahrav/go-gavel/internal/domain"
	"github.com/ahrav/go-gavel/internal/ports"
)

var _ ports.Unit = (*MaxPoolUnit)(nil)

// TieBreaker represents the strategy for handling equal scores.
type TieBreaker string

// Supported tie-breaking strategies.
const (
	// TieFirst selects the first answer with the highest score.
	TieFirst TieBreaker = "first"
	// TieRandom randomly selects among answers with the highest score.
	TieRandom TieBreaker = "random"
	// TieError returns an error when multiple answers have the highest score.
	TieError TieBreaker = "error"
)

// Common errors returned by MaxPoolUnit.
var (
	// ErrTie is returned when multiple answers have the highest score and TieError is configured.
	ErrTie = errors.New("multiple answers tied with highest score")
	// ErrBelowMinScore is returned when the highest score is below the minimum threshold.
	ErrBelowMinScore = errors.New("highest score below minimum threshold")
	// ErrNoScores is returned when no scores are provided for aggregation.
	ErrNoScores = errors.New("no scores provided for aggregation")
	// ErrEmptyUnitName is returned when unit name is empty.
	ErrEmptyUnitName = errors.New("unit name cannot be empty")
	// ErrScoreMismatch is returned when scores and candidates lengths don't match.
	ErrScoreMismatch = errors.New("scores and candidates length mismatch")
)

// Package-level validator instance for configuration validation.
var validate = validator.New()

// MaxPoolUnit implements the Aggregator interface using maximum selection
// to determine the winning answer and aggregate score.
// It processes judge scores to select the highest-scoring answer.
// The unit is stateless and thread-safe for concurrent execution.
type MaxPoolUnit struct {
	// name is the unique identifier for this unit instance.
	name string
	// config contains the validated configuration parameters.
	config MaxPoolConfig
}

// MaxPoolConfig defines the configuration parameters for the MaxPoolUnit.
// All fields are validated during unit creation and parameter unmarshaling.
type MaxPoolConfig struct {
	// TieBreaker defines how to handle equal scores.
	// Options: "first" (select first), "random" (random selection), "error" (fail on ties).
	TieBreaker TieBreaker `yaml:"tie_breaker" json:"tie_breaker" validate:"required,oneof=first random error"`

	// MinScore sets the minimum acceptable aggregate score.
	// Answers below this threshold may be rejected.
	MinScore float64 `yaml:"min_score" json:"min_score" validate:"min=0.0,max=1.0"`

	// RequireAllScores determines if all answers must have scores.
	// When true, missing scores cause an error. When false, only scored answers are considered.
	RequireAllScores bool `yaml:"require_all_scores" json:"require_all_scores"`
}

// NewMaxPoolUnit creates a new MaxPoolUnit with the specified configuration.
// The unit validates its configuration to ensure proper aggregation behavior.
// Returns an error if configuration validation fails.
func NewMaxPoolUnit(name string, config MaxPoolConfig) (*MaxPoolUnit, error) {
	if name == "" {
		return nil, ErrEmptyUnitName
	}

	if err := validate.Struct(config); err != nil {
		return nil, fmt.Errorf("configuration validation failed: %w", err)
	}

	return &MaxPoolUnit{
		name:   name,
		config: config,
	}, nil
}

// Name returns the unique identifier for this unit instance.
// The name is used for logging, debugging, and graph node referencing.
func (mpu *MaxPoolUnit) Name() string { return mpu.name }

// Execute aggregates judge scores using maximum selection to determine
// the winning answer and calculate aggregate scores.
// It retrieves answers and judge scores from state, selects the highest score,
// and produces a Verdict with the winning answer.
// Returns updated state with the verdict or an error if aggregation fails.
func (mpu *MaxPoolUnit) Execute(ctx context.Context, state domain.State) (domain.State, error) {
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

	if numScores != numAnswers {
		if mpu.config.RequireAllScores {
			return state, fmt.Errorf("mismatch between answers (%d) and judge scores (%d)",
				numAnswers, numScores)
		}
		// Handle partial scoring case - only process up to the minimum available
		if numScores < numAnswers {
			numAnswers = numScores
		}
	}

	// Extract scores for aggregation - only process valid pairs
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

// Aggregate implements the domain.Aggregator interface using maximum selection
// to determine the winning answer and aggregate score.
// It selects the answer with the highest score from all candidates.
// Returns the winning answer, its aggregate score, and any error encountered.
func (mpu *MaxPoolUnit) Aggregate(
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

	// Find the highest score and corresponding answer.
	var winnerIdx int
	var maxScore = math.Inf(-1) // Start with negative infinity.
	var tieCount int

	for i, score := range scores {
		// Validate score is not NaN or infinite.
		if math.IsNaN(score) || math.IsInf(score, 0) {
			return domain.Answer{}, 0, fmt.Errorf("invalid score at index %d: %f", i, score)
		}

		if score > maxScore {
			maxScore = score
			winnerIdx = i
			tieCount = 1
		} else if score == maxScore {
			tieCount++
		}
	}

	// Check minimum score requirement.
	if maxScore < mpu.config.MinScore {
		return domain.Answer{}, 0, fmt.Errorf("%w: highest=%.3f, minimum=%.3f",
			ErrBelowMinScore, maxScore, mpu.config.MinScore)
	}

	// Handle tie-breaking.
	if tieCount > 1 {
		switch mpu.config.TieBreaker {
		case TieFirst:
			// Keep the first occurrence (winnerIdx is already correct).
		case TieError:
			return domain.Answer{}, 0, fmt.Errorf("%w: %d answers with score %.3f", ErrTie, tieCount, maxScore)
		case TieRandom:
			// Count how many candidates have the max score
			tiedCandidates := make([]int, 0, tieCount)
			for i, score := range scores {
				if score == maxScore {
					tiedCandidates = append(tiedCandidates, i)
				}
			}
			// Use crypto/rand for unbiased selection
			n, err := rand.Int(rand.Reader, big.NewInt(int64(len(tiedCandidates))))
			if err != nil {
				return domain.Answer{}, 0, fmt.Errorf("failed to generate random number: %w", err)
			}
			winnerIdx = tiedCandidates[n.Int64()]
		default:
			return domain.Answer{}, 0, fmt.Errorf("unknown tie breaker: %s", mpu.config.TieBreaker)
		}
	}

	return candidates[winnerIdx], maxScore, nil
}

// Validate checks if the unit is properly configured and ready for execution.
// It validates the configuration parameters to ensure proper aggregation behavior.
// Returns nil if validation passes, or an error describing what is invalid.
func (mpu *MaxPoolUnit) Validate() error {
	if err := validate.Struct(mpu.config); err != nil {
		return fmt.Errorf("configuration validation failed: %w", err)
	}

	return nil
}

// UnmarshalParameters deserializes YAML configuration parameters into the
// unit's configuration struct with strict validation.
// This method enables YAML-based configuration with unknown field detection
// to prevent configuration typos from being silently ignored.
// Returns an error if YAML parsing fails or unknown fields are detected.
func (mpu *MaxPoolUnit) UnmarshalParameters(params yaml.Node) error {
	var config MaxPoolConfig

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

// DefaultMaxPoolConfig returns a MaxPoolConfig with sensible defaults.
func DefaultMaxPoolConfig() MaxPoolConfig {
	return MaxPoolConfig{
		TieBreaker:       TieFirst,
		MinScore:         0.0,
		RequireAllScores: true,
	}
}

// CreateMaxPoolUnit is a factory function that creates a MaxPoolUnit
// from a configuration map, following the UnitFactory pattern.
// This function is used by the UnitRegistry for dynamic unit creation.
func CreateMaxPoolUnit(id string, config map[string]any) (*MaxPoolUnit, error) {
	// Start with default configuration.
	poolConfig := DefaultMaxPoolConfig()

	// Override with provided values.
	if tieBreaker, ok := config["tie_breaker"].(string); ok {
		poolConfig.TieBreaker = TieBreaker(tieBreaker)
	}

	if minScore, ok := config["min_score"]; ok {
		if val, ok := minScore.(float64); ok {
			poolConfig.MinScore = val
		}
	}

	if requireAllScores, ok := config["require_all_scores"].(bool); ok {
		poolConfig.RequireAllScores = requireAllScores
	}

	return NewMaxPoolUnit(id, poolConfig)
}
