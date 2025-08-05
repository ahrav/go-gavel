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

var _ ports.Unit = (*MaxPoolUnit)(nil)

// MaxPoolUnit implements the Aggregator interface using maximum selection
// to determine the winning answer and aggregate score.
// It processes judge scores to select the highest-scoring answer.
// The unit is stateless and thread-safe for concurrent execution.
type MaxPoolUnit struct {
	name   string
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
// It returns an error if the configuration is invalid.
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
		if numScores < numAnswers {
			numAnswers = numScores
		}
	}

	scores := make([]float64, numAnswers)
	for i := 0; i < numAnswers; i++ {
		scores[i] = judgeSummaries[i].Score
	}

	winner, aggregateScore, err := mpu.Aggregate(scores, answers[:numAnswers])
	if err != nil {
		return state, fmt.Errorf("aggregation failed: %w", err)
	}

	verdict := domain.Verdict{
		ID:             fmt.Sprintf("%s_verdict", mpu.name),
		WinnerAnswer:   &winner,
		AggregateScore: aggregateScore,
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
		// Validate score is not NaN or infinite to prevent corrupted aggregation.
		// NaN and infinite values can break comparison logic and produce invalid results.
		if math.IsNaN(score) || math.IsInf(score, 0) {
			return domain.Answer{}, 0, fmt.Errorf("invalid score at index %d: %f", i, score)
		}

		if score > maxScore {
			maxScore = score
			winnerIdx = i
			tieCount = 1 // Reset tie count when new max found
		} else if score == maxScore {
			tieCount++ // Track ties for tie-breaking logic
		}
	}

	// Check minimum score requirement.
	if maxScore < mpu.config.MinScore {
		return domain.Answer{}, 0, fmt.Errorf("%w: highest=%.3f, minimum=%.3f",
			ErrBelowMinScore, maxScore, mpu.config.MinScore)
	}

	// Handle tie-breaking when multiple candidates have the same highest score.
	// The strategy chosen affects determinism and fairness of selection.
	if tieCount > 1 {
		switch mpu.config.TieBreaker {
		case TieFirst:
			// Keep the first occurrence (winnerIdx is already correct).
			// This provides deterministic, reproducible results.
		case TieError:
			// Fail explicitly when ties occur, forcing caller to handle ambiguity.
			return domain.Answer{}, 0, fmt.Errorf("%w: %d answers with score %.3f", ErrTie, tieCount, maxScore)
		case TieRandom:
			// Randomly select among tied candidates for fairness.
			// This prevents systematic bias toward first/last positions.
			tiedCandidates := make([]int, 0, tieCount)
			for i, score := range scores {
				if score == maxScore {
					tiedCandidates = append(tiedCandidates, i)
				}
			}
			// Use crypto/rand for cryptographically secure, unbiased selection.
			// This ensures no predictable patterns in tie-breaking decisions.
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

// Validate checks if the unit is properly configured.
func (mpu *MaxPoolUnit) Validate() error {
	if err := validate.Struct(mpu.config); err != nil {
		return fmt.Errorf("configuration validation failed: %w", err)
	}
	return nil
}

// UnmarshalParameters deserializes YAML parameters into the unit's config.
func (mpu *MaxPoolUnit) UnmarshalParameters(params yaml.Node) error {
	var config MaxPoolConfig
	if err := params.Decode(&config); err != nil {
		return fmt.Errorf("failed to decode parameters: %w", err)
	}
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

// CreateMaxPoolUnit is a factory function that creates a MaxPoolUnit from a
// configuration map, for use with the UnitRegistry.
func CreateMaxPoolUnit(id string, config map[string]any) (*MaxPoolUnit, error) {
	poolConfig := DefaultMaxPoolConfig()
	if val, ok := config["tie_breaker"].(string); ok {
		poolConfig.TieBreaker = TieBreaker(val)
	}
	if val, ok := config["min_score"].(float64); ok {
		poolConfig.MinScore = val
	}
	if val, ok := config["require_all_scores"].(bool); ok {
		poolConfig.RequireAllScores = val
	}
	return NewMaxPoolUnit(id, poolConfig)
}
