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
// median is selected as the winner.
//
// The unit applies statistical median calculation to judge scores and selects
// the candidate with minimum distance from the median. This approach reduces
// the impact of outlier scores and provides robust aggregation for evaluation scenarios
// with potential judge bias or inconsistency.
//
// Concurrency: The unit is stateless and thread-safe for concurrent execution.
// Multiple goroutines may safely call Execute simultaneously.
//
// Error Conditions:
//   - Returns ErrNoScores when no judge scores are available
//   - Returns ErrScoreMismatch when scores and candidates count differs
//   - Returns ErrBelowMinScore when median falls below configured threshold
//   - Returns ErrTie when multiple candidates are equidistant and TieError is configured
//
// Example:
//
//	config := MedianPoolConfig{
//	    TieBreaker: TieFirst,
//	    MinScore: 0.6,
//	    RequireAllScores: true,
//	}
//	unit, err := NewMedianPoolUnit("median_agg", config)
type MedianPoolUnit struct {
	// name is the unique identifier for this unit instance.
	// Used for logging, debugging, and verdict ID generation.
	name string
	// config contains validated configuration parameters.
	// Immutable after unit creation to ensure thread safety.
	config MedianPoolConfig
}

// MedianPoolConfig defines the configuration parameters for the MedianPoolUnit.
// All fields are validated during unit creation and parameter unmarshaling.
// Configuration is immutable after validation to ensure thread safety.
type MedianPoolConfig struct {
	// TieBreaker defines the strategy for handling cases where multiple candidates
	// are equidistant from the median score.
	//
	// Supported values:
	//   - "first": Select the first candidate (deterministic)
	//   - "random": Randomly select among tied candidates (fair but non-deterministic)
	//   - "error": Return an error requiring explicit handling
	//
	// Default: "first" for deterministic behavior in evaluation pipelines.
	TieBreaker TieBreaker `yaml:"tie_breaker" json:"tie_breaker" validate:"required,oneof=first random error"`

	// MinScore sets the minimum acceptable aggregate (median) score.
	// If the calculated median falls below this threshold, aggregation fails
	// with ErrBelowMinScore.
	//
	// Range: 0.0 to 1.0 (inclusive)
	// Default: 0.0 (no minimum threshold)
	MinScore float64 `yaml:"min_score" json:"min_score" validate:"min=0.0,max=1.0"`

	// RequireAllScores determines if all answers must have corresponding judge scores.
	// When true, a mismatch between answer count and score count triggers an error.
	// When false, the unit processes only answers with available scores.
	//
	// Set to true for strict evaluation scenarios requiring complete scoring.
	// Set to false when partial scoring is acceptable (e.g., optional judges).
	RequireAllScores bool `yaml:"require_all_scores" json:"require_all_scores"`
}

// NewMedianPoolUnit creates a new MedianPoolUnit with the specified configuration.
// The unit validates all configuration parameters and ensures the name is non-empty.
//
// Parameters:
//   - name: Unique identifier for this unit (used in logs and verdict IDs)
//   - config: Configuration parameters (validated using struct tags)
//
// Returns a configured MedianPoolUnit ready for execution, or an error if:
//   - name is empty (returns ErrEmptyUnitName)
//   - config validation fails (returns wrapped validation error)
//
// The returned unit is immutable and thread-safe for concurrent use.
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
// The name is immutable after unit creation and used for:
//   - Logging and debugging output
//   - Verdict ID generation ("<name>_verdict")
//   - Configuration management and unit registry lookups
func (mpu *MedianPoolUnit) Name() string { return mpu.name }

// Execute aggregates judge scores using median-based candidate selection.
// It calculates the median of all judge scores and selects the candidate
// whose score has the minimum distance from that median.
//
// State Requirements:
//   - domain.KeyAnswers: []domain.Answer - candidate answers to evaluate
//   - domain.KeyJudgeScores: []domain.JudgeSummary - scores from judge units
//
// State Updates:
//   - domain.KeyVerdict: *domain.Verdict - winner and aggregate score
//
// Algorithm:
//  1. Extract answers and judge scores from state
//  2. Validate counts match (if RequireAllScores is true)
//  3. Calculate statistical median of all scores
//  4. Find candidate with minimum distance from median
//  5. Apply tie-breaking strategy if multiple candidates are equidistant
//  6. Verify median meets minimum score threshold
//  7. Create verdict with winner and median as aggregate score
//
// Error Conditions:
//   - Missing required state keys
//   - Score/candidate count mismatch (when RequireAllScores=true)
//   - Median below MinScore threshold
//   - Tie resolution failure (when TieBreaker=TieError)
//   - Invalid scores (NaN, Inf)
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

// calculateMedian computes the statistical median from a slice of scores.
// The method implements the standard mathematical definition:
//   - Odd count: returns the middle value after sorting
//   - Even count: returns the arithmetic mean of the two middle values
//
// Side Effects: The input slice is sorted in-place for performance.
// Callers should pass a copy if original order must be preserved.
//
// Edge Cases:
//   - Empty slice returns 0.0 (caller should validate before calling)
//   - Single element returns that element
//   - All equal elements return that value
//
// Time Complexity: O(n log n) due to sorting
// Space Complexity: O(1) as sorting is in-place
func (mpu *MedianPoolUnit) calculateMedian(scores []float64) float64 {
	if len(scores) == 0 {
		return 0
	}
	sort.Float64s(scores)
	n := len(scores)
	if n%2 == 1 {
		// Odd count: middle element is at index n/2 after sorting
		return scores[n/2]
	}
	// Even count: median is arithmetic mean of two middle elements
	// This ensures the median represents the central tendency even
	// when no single score represents the exact middle.
	return (scores[n/2-1] + scores[n/2]) / 2
}

// Aggregate implements the domain.Aggregator interface by selecting the
// candidate whose score has minimum distance from the median of all scores.
//
// Parameters:
//   - scores: Judge scores for each candidate (must not contain NaN/Inf)
//   - candidates: Corresponding candidate answers (must match scores length)
//
// Returns:
//   - domain.Answer: The winning candidate closest to median
//   - float64: The median score as the aggregate value
//   - error: Validation or processing error
//
// Algorithm Details:
//  1. Validates input arrays have equal length and contain valid scores
//  2. Calculates median using standard statistical definition
//  3. Finds candidate(s) with minimum absolute distance from median
//  4. Applies configured tie-breaking strategy for equidistant candidates
//  5. Validates median meets minimum score threshold
//
// Error Conditions:
//   - ErrNoScores: empty scores slice
//   - ErrScoreMismatch: length mismatch between scores and candidates
//   - Invalid score error: NaN or Inf values detected
//   - ErrBelowMinScore: median below configured threshold
//   - ErrTie: multiple equidistant candidates with TieError strategy
//
// Thread Safety: Safe for concurrent use (no shared state modified)
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

	// Validate all scores are finite numbers before processing.
	// NaN and Inf values would corrupt median calculation and distance comparisons.
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

	// Handle ties: multiple candidates with identical distance from median
	if len(tieIndices) > 1 {
		switch mpu.config.TieBreaker {
		case TieFirst:
			// Deterministic selection: choose first tied candidate
			// Provides reproducible results for testing and evaluation consistency
			winnerIdx = tieIndices[0]
		case TieError:
			// Explicit handling required: force caller to address ambiguity
			// Useful when tie-breaking has business logic implications
			return domain.Answer{}, 0, fmt.Errorf("%w: %d answers with distance %.3f from median %.3f (tied candidates: %v)",
				ErrTie, len(tieIndices), bestDistance, medianScore, tieIndices)
		case TieRandom:
			// Fair random selection among tied candidates
			// Use math/rand for better performance - cryptographic security not needed for tie-breaking
			winnerIdx = tieIndices[rand.Intn(len(tieIndices))] // #nosec G404
		}
	}

	return candidates[winnerIdx], medianScore, nil
}

// Validate checks if the unit is properly configured and ready for execution.
// This method should be called after unit creation and before adding to
// evaluation pipelines to ensure configuration integrity.
//
// Validation includes:
//   - Configuration struct validation using validator tags
//   - TieBreaker enum value verification
//   - MinScore range validation (0.0-1.0)
//
// Returns nil if validation passes, or a descriptive error indicating
// the specific configuration issue that must be resolved.
func (mpu *MedianPoolUnit) Validate() error {
	if err := validate.Struct(mpu.config); err != nil {
		return fmt.Errorf("configuration validation failed: %w", err)
	}
	return nil
}

// UnmarshalParameters deserializes YAML configuration parameters and updates
// the unit's configuration. This method enables dynamic reconfiguration
// of existing units within evaluation pipelines.
//
// Parameters:
//   - params: YAML node containing configuration fields
//
// Supported YAML fields:
//   - tie_breaker: "first"|"random"|"error"
//   - min_score: float64 (0.0-1.0)
//   - require_all_scores: boolean
//
// Example YAML:
//
//	tie_breaker: "first"
//	min_score: 0.7
//	require_all_scores: true
//
// Error Conditions:
//   - YAML parsing errors for malformed input
//   - Validation errors for invalid configuration values
//   - Type conversion errors for incorrect field types
//
// Thread Safety: This method modifies unit state and is NOT thread-safe.
// Callers must ensure exclusive access during reconfiguration.
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

// DefaultMedianPoolConfig returns a MedianPoolConfig with production-ready defaults.
// These defaults provide deterministic, inclusive behavior suitable for most
// evaluation scenarios.
//
// Default Configuration:
//   - TieBreaker: TieFirst (deterministic selection)
//   - MinScore: 0.0 (no minimum threshold)
//   - RequireAllScores: true (strict scoring validation)
//
// Use this as a starting point and override specific fields as needed:
//
//	config := DefaultMedianPoolConfig()
//	config.MinScore = 0.6  // Add quality threshold
//	config.TieBreaker = TieRandom  // Enable fair tie-breaking
func DefaultMedianPoolConfig() MedianPoolConfig {
	return MedianPoolConfig{
		TieBreaker:       TieFirst,
		MinScore:         0.0,
		RequireAllScores: true,
	}
}

// NewMedianPoolFromConfig creates a MedianPoolUnit from a configuration map.
// This is the boundary adapter for YAML/JSON configuration.
// Median pool doesn't require an LLM client (deterministic aggregation).
func NewMedianPoolFromConfig(id string, config map[string]any, llm ports.LLMClient) (ports.Unit, error) {
	// llm is ignored - median pool is deterministic.

	// Use yaml marshaling for clean conversion.
	data, err := yaml.Marshal(config)
	if err != nil {
		return nil, fmt.Errorf("marshal config: %w", err)
	}

	// Start with defaults, then overlay user config.
	cfg := DefaultMedianPoolConfig()
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	return NewMedianPoolUnit(id, cfg)
}
