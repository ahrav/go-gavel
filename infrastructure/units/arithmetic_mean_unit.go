// Package units provides domain-specific evaluation units that implement
// the ports.Unit interface for the go-gavel evaluation engine.
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

// ArithmeticMeanUnit implements an Aggregator that selects the candidate with the
// highest individual score and returns that candidate's score as the aggregate score.
//
// This unit implements the "best individual wins" strategy where:
// - The candidate with the highest score is selected as the winner
// - The aggregate score is the winner's actual score (not a calculated mean)
//
// This fulfills Story 1.3 MVP requirements - a basic aggregator implementation.
// Story 1.6 defines a true arithmetic mean aggregator for multi-judge scenarios,
// which would be a separate implementation that calculates the mean of all scores.
//
// The unit is stateless and thread-safe for concurrent execution.
type ArithmeticMeanUnit struct {
	// name is the unique identifier for this unit instance.
	name string
	// config contains the validated configuration parameters.
	config ArithmeticMeanConfig
}

// ArithmeticMeanConfig defines the configuration parameters for the ArithmeticMeanUnit.
// All fields are validated during unit creation and parameter unmarshaling.
type ArithmeticMeanConfig struct {
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

// NewArithmeticMeanUnit creates a new ArithmeticMeanUnit with the specified configuration.
// The unit validates its configuration to ensure proper aggregation behavior.
// Returns an error if configuration validation fails.
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
// The name is used for logging, debugging, and graph node referencing.
func (mpu *ArithmeticMeanUnit) Name() string { return mpu.name }

// Execute aggregates judge scores using arithmetic mean calculation. It retrieves answers
// and scores from the state, selects the highest-scoring candidate as the
// winner, and produces a Verdict with the mean of all scores.
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

// Aggregate implements the domain.Aggregator interface. It selects the
// candidate with the highest individual score and returns that candidate's
// score as the aggregate score.
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

	var winnerIdx int
	var maxScore = -1.0
	var tieIndices []int

	for i, score := range scores {
		if math.IsNaN(score) || math.IsInf(score, 0) {
			return domain.Answer{}, 0, fmt.Errorf("invalid score at index %d: %f", i, score)
		}
		if score > maxScore {
			maxScore = score
			winnerIdx = i
			tieIndices = []int{i}
		} else if score == maxScore {
			tieIndices = append(tieIndices, i)
		}
	}

	// Check if the winner's score meets the minimum requirement
	if maxScore < mpu.config.MinScore {
		return domain.Answer{}, 0, fmt.Errorf("%w: highest=%.3f, minimum=%.3f",
			ErrBelowMinScore, maxScore, mpu.config.MinScore)
	}

	if len(tieIndices) > 1 {
		switch mpu.config.TieBreaker {
		case TieFirst:
			winnerIdx = tieIndices[0]
		case TieError:
			return domain.Answer{}, 0, fmt.Errorf("%w: %d answers with score %.3f", ErrTie, len(tieIndices), maxScore)
		case TieRandom:
			n, err := rand.Int(rand.Reader, big.NewInt(int64(len(tieIndices))))
			if err != nil {
				return domain.Answer{}, 0, fmt.Errorf("failed to generate random number for tie-breaking: %w", err)
			}
			winnerIdx = tieIndices[n.Int64()]
		}
	}

	return candidates[winnerIdx], maxScore, nil
}

// Validate checks if the unit is properly configured and ready for execution.
// It validates the configuration parameters to ensure proper aggregation behavior.
// Returns nil if validation passes, or an error describing what is invalid.
func (mpu *ArithmeticMeanUnit) Validate() error {
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

// DefaultArithmeticMeanConfig returns a ArithmeticMeanConfig with sensible defaults.
func DefaultArithmeticMeanConfig() ArithmeticMeanConfig {
	return ArithmeticMeanConfig{
		TieBreaker:       TieFirst,
		MinScore:         0.0,
		RequireAllScores: true,
	}
}

// CreateArithmeticMeanUnit is a factory function that creates a ArithmeticMeanUnit
// from a configuration map, following the UnitFactory pattern.
// This function is used by the UnitRegistry for dynamic unit creation.
func CreateArithmeticMeanUnit(id string, config map[string]any) (*ArithmeticMeanUnit, error) {
	// Start with default configuration.
	poolConfig := DefaultArithmeticMeanConfig()

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

	return NewArithmeticMeanUnit(id, poolConfig)
}