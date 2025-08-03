// Package units provides domain-specific evaluation units that implement
// the ports.Unit interface for the go-gavel evaluation engine.
package units

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"

	"gopkg.in/yaml.v3"

	"github.com/ahrav/go-gavel/internal/domain"
	"github.com/ahrav/go-gavel/internal/ports"
)

var _ ports.Unit = (*MedianPoolUnit)(nil)

// MedianPoolUnit implements an Aggregator that uses the median score to
// determine the aggregate score. The candidate whose score is closest to the
// median is selected as the winner. It is stateless and thread-safe.
type MedianPoolUnit struct {
	name   string
	config MedianPoolConfig
}

// MedianPoolConfig defines the configuration parameters for the MedianPoolUnit.
// All fields are validated during unit creation and parameter unmarshaling.
type MedianPoolConfig struct {
	// TieBreaker defines how to handle cases where multiple candidates are
	// equidistant from the median score. Options: "first", "random", "error".
	TieBreaker TieBreaker `yaml:"tie_breaker" json:"tie_breaker" validate:"required,oneof=first random error"`

	// MinScore sets the minimum acceptable aggregate (median) score.
	MinScore float64 `yaml:"min_score" json:"min_score" validate:"min=0.0,max=1.0"`

	// RequireAllScores determines if all answers must have scores. If true,
	// missing scores will cause an error.
	RequireAllScores bool `yaml:"require_all_scores" json:"require_all_scores"`
}

// NewMedianPoolUnit creates a new MedianPoolUnit with the specified config.
// It returns an error if the configuration is invalid.
func NewMedianPoolUnit(name string, config MedianPoolConfig) (*MedianPoolUnit, error) {
	if name == "" {
		return nil, ErrEmptyUnitName
	}
	if err := validate.Struct(config); err != nil {
		return nil, fmt.Errorf("configuration validation failed: %w", err)
	}
	return &MedianPoolUnit{
		name:   name,
		config: config,
	}, nil
}

// Name returns the unique identifier for this unit instance.
func (mpu *MedianPoolUnit) Name() string { return mpu.name }

// Execute aggregates judge scores using a median calculation. It finds the
// candidate whose score is closest to the median and produces a Verdict with
// that winner and the median score as the aggregate.
func (mpu *MedianPoolUnit) Execute(ctx context.Context, state domain.State) (domain.State, error) {
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
			return state, fmt.Errorf("mismatch between answers (%d) and judge scores (%d)", numAnswers, numScores)
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

// calculateMedian computes the median from a slice of scores using standard
// mathematical definition. For an odd number of scores, returns the middle value.
// For an even number of scores, returns the average of the two middle values.
// The input slice is sorted in-place.
func (mpu *MedianPoolUnit) calculateMedian(scores []float64) float64 {
	if len(scores) == 0 {
		return 0
	}
	sort.Float64s(scores)
	n := len(scores)
	if n%2 == 1 {
		return scores[n/2]
	}
	// For even length, return average of two middle values
	return (scores[n/2-1] + scores[n/2]) / 2
}

// Aggregate implements the domain.Aggregator interface. It selects the
// candidate whose score is closest to the median of all scores. The aggregate
// score is the median itself.
func (mpu *MedianPoolUnit) Aggregate(
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

	for i, score := range scores {
		if math.IsNaN(score) || math.IsInf(score, 0) {
			return domain.Answer{}, 0, fmt.Errorf("invalid score at index %d: %f", i, score)
		}
	}

	scoresCopy := make([]float64, len(scores))
	copy(scoresCopy, scores)
	medianScore := mpu.calculateMedian(scoresCopy)

	if medianScore < mpu.config.MinScore {
		return domain.Answer{}, 0, fmt.Errorf("%w: median=%.3f, minimum=%.3f",
			ErrBelowMinScore, medianScore, mpu.config.MinScore)
	}

	var winnerIdx int
	var bestDistance = math.Inf(1)
	var tieIndices []int

	for i, score := range scores {
		distance := math.Abs(score - medianScore)
		if distance < bestDistance {
			bestDistance = distance
			winnerIdx = i
			tieIndices = []int{i}
		} else if distance == bestDistance {
			tieIndices = append(tieIndices, i)
		}
	}

	if len(tieIndices) > 1 {
		switch mpu.config.TieBreaker {
		case TieFirst:
			winnerIdx = tieIndices[0]
		case TieError:
			return domain.Answer{}, 0, fmt.Errorf("%w: %d answers with distance %.3f from median %.3f (tied candidates: %v)",
				ErrTie, len(tieIndices), bestDistance, medianScore, tieIndices)
		case TieRandom:
			// Use math/rand for better performance - cryptographic security not needed for tie-breaking
			winnerIdx = tieIndices[rand.Intn(len(tieIndices))] // #nosec G404
		}
	}

	return candidates[winnerIdx], medianScore, nil
}

// Validate checks if the unit is properly configured.
func (mpu *MedianPoolUnit) Validate() error {
	if err := validate.Struct(mpu.config); err != nil {
		return fmt.Errorf("configuration validation failed: %w", err)
	}
	return nil
}

// UnmarshalParameters deserializes YAML parameters into the unit's config.
func (mpu *MedianPoolUnit) UnmarshalParameters(params yaml.Node) error {
	var config MedianPoolConfig
	if err := params.Decode(&config); err != nil {
		return fmt.Errorf("failed to decode parameters: %w", err)
	}
	if err := validate.Struct(config); err != nil {
		return fmt.Errorf("parameter validation failed: %w", err)
	}
	mpu.config = config
	return nil
}

// DefaultMedianPoolConfig returns a MedianPoolConfig with sensible defaults.
func DefaultMedianPoolConfig() MedianPoolConfig {
	return MedianPoolConfig{
		TieBreaker:       TieFirst,
		MinScore:         0.0,
		RequireAllScores: true,
	}
}

// CreateMedianPoolUnit is a factory function that creates a MedianPoolUnit from a
// configuration map, for use with the UnitRegistry.
func CreateMedianPoolUnit(id string, config map[string]any) (*MedianPoolUnit, error) {
	poolConfig := DefaultMedianPoolConfig()

	if val, ok := config["tie_breaker"]; ok {
		strVal, isString := val.(string)
		if !isString {
			return nil, fmt.Errorf("tie_breaker must be a string, got %T", val)
		}
		// Validate the string value against allowed TieBreaker values
		switch strVal {
		case "first", "random", "error":
			poolConfig.TieBreaker = TieBreaker(strVal)
		default:
			return nil, fmt.Errorf("invalid tie_breaker value: %s (must be one of: first, random, error)", strVal)
		}
	}

	if val, ok := config["min_score"]; ok {
		floatVal, isFloat := val.(float64)
		if !isFloat {
			return nil, fmt.Errorf("min_score must be a float64, got %T", val)
		}
		poolConfig.MinScore = floatVal
	}

	if val, ok := config["require_all_scores"]; ok {
		boolVal, isBool := val.(bool)
		if !isBool {
			return nil, fmt.Errorf("require_all_scores must be a bool, got %T", val)
		}
		poolConfig.RequireAllScores = boolVal
	}

	return NewMedianPoolUnit(id, poolConfig)
}
