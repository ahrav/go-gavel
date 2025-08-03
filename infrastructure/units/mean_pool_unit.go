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

var _ ports.Unit = (*MeanPoolUnit)(nil)

// MeanPoolUnit implements an Aggregator that calculates the arithmetic mean of
// all scores for the aggregate score, but selects the candidate with the
// highest individual score as the winner.
//
// This unit is useful when you want to:
// - Select the best individual answer (highest score)
// - But also consider the overall quality of all answers (mean score)
//
// Example use case: In a multi-judge evaluation where you want the winner
// to be the best-rated answer, but the overall score reflects the average
// quality across all submitted answers.
//
// The unit is stateless and thread-safe.
type MeanPoolUnit struct {
	name   string
	config MeanPoolConfig
}

// MeanPoolConfig defines the configuration parameters for the MeanPoolUnit.
// All fields are validated during unit creation and parameter unmarshaling.
type MeanPoolConfig struct {
	// TieBreaker defines how to handle equal scores when selecting a winner.
	// Options are "first", "random", or "error".
	TieBreaker TieBreaker `yaml:"tie_breaker" json:"tie_breaker" validate:"required,oneof=first random error"`

	// MinScore sets the minimum acceptable aggregate (mean) score.
	MinScore float64 `yaml:"min_score" json:"min_score" validate:"min=0.0,max=1.0"`

	// RequireAllScores determines if all answers must have scores. If true,
	// missing scores will cause an error.
	RequireAllScores bool `yaml:"require_all_scores" json:"require_all_scores"`
}

// NewMeanPoolUnit creates a new MeanPoolUnit with the specified configuration.
// It returns an error if the configuration is invalid.
func NewMeanPoolUnit(name string, config MeanPoolConfig) (*MeanPoolUnit, error) {
	if name == "" {
		return nil, ErrEmptyUnitName
	}
	if err := validate.Struct(config); err != nil {
		return nil, fmt.Errorf("configuration validation failed: %w", err)
	}
	return &MeanPoolUnit{
		name:   name,
		config: config,
	}, nil
}

// Name returns the unique identifier for this unit instance.
func (mpu *MeanPoolUnit) Name() string { return mpu.name }

// Execute aggregates judge scores using a mean calculation. It retrieves answers
// and scores from the state, selects the highest-scoring candidate as the
// winner, and produces a Verdict with the mean of all scores.
func (mpu *MeanPoolUnit) Execute(ctx context.Context, state domain.State) (domain.State, error) {
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

// Aggregate implements the domain.Aggregator interface. It selects the
// candidate with the highest individual score but returns the arithmetic mean
// of all scores as the aggregate score.
func (mpu *MeanPoolUnit) Aggregate(
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

	var sum float64
	var winnerIdx int
	var maxScore = -1.0
	var tieIndices []int

	for i, score := range scores {
		if math.IsNaN(score) || math.IsInf(score, 0) {
			return domain.Answer{}, 0, fmt.Errorf("invalid score at index %d: %f", i, score)
		}
		sum += score
		if score > maxScore {
			maxScore = score
			winnerIdx = i
			tieIndices = []int{i}
		} else if score == maxScore {
			tieIndices = append(tieIndices, i)
		}
	}

	meanScore := sum / float64(len(scores))
	if meanScore < mpu.config.MinScore {
		return domain.Answer{}, 0, fmt.Errorf("%w: mean=%.3f, minimum=%.3f",
			ErrBelowMinScore, meanScore, mpu.config.MinScore)
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

	return candidates[winnerIdx], meanScore, nil
}

// Validate checks if the unit is properly configured.
func (mpu *MeanPoolUnit) Validate() error {
	if err := validate.Struct(mpu.config); err != nil {
		return fmt.Errorf("configuration validation failed: %w", err)
	}
	return nil
}

// UnmarshalParameters deserializes YAML parameters into the unit's config.
func (mpu *MeanPoolUnit) UnmarshalParameters(params yaml.Node) error {
	var config MeanPoolConfig
	if err := params.Decode(&config); err != nil {
		return fmt.Errorf("failed to decode parameters: %w", err)
	}
	if err := validate.Struct(config); err != nil {
		return fmt.Errorf("parameter validation failed: %w", err)
	}
	mpu.config = config
	return nil
}

// DefaultMeanPoolConfig returns a MeanPoolConfig with sensible defaults.
func DefaultMeanPoolConfig() MeanPoolConfig {
	return MeanPoolConfig{
		TieBreaker:       TieFirst,
		MinScore:         0.0,
		RequireAllScores: true,
	}
}

// CreateMeanPoolUnit is a factory function that creates a MeanPoolUnit from a
// configuration map, for use with the UnitRegistry.
func CreateMeanPoolUnit(id string, config map[string]any) (*MeanPoolUnit, error) {
	poolConfig := DefaultMeanPoolConfig()
	if val, ok := config["tie_breaker"].(string); ok {
		poolConfig.TieBreaker = TieBreaker(val)
	}
	if val, ok := config["min_score"].(float64); ok {
		poolConfig.MinScore = val
	}
	if val, ok := config["require_all_scores"].(bool); ok {
		poolConfig.RequireAllScores = val
	}
	return NewMeanPoolUnit(id, poolConfig)
}
