package units

import (
	"bytes"
	"context"
	"fmt"
	"time"
	"unicode/utf8"

	"github.com/agnivade/levenshtein"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"golang.org/x/text/cases"
	"gopkg.in/yaml.v3"

	"github.com/ahrav/go-gavel/internal/domain"
	"github.com/ahrav/go-gavel/internal/ports"
)

var (
	_ ports.Unit = (*FuzzyMatchUnit)(nil)

	// foldCaser is a package-level Unicode case folder for performance.
	// This avoids creating a new caser for each string preparation.
	foldCaser = cases.Fold()
)

// FuzzyMatchUnit implements a deterministic Unit that performs fuzzy string matching
// between candidate answers and a reference answer using the Levenshtein distance
// algorithm. It evaluates each answer based on string similarity, producing scores
// between 0.0 and 1.0 based on the edit distance.
//
// This unit provides deterministic evaluation without requiring an LLM, making it
// ideal for scenarios where approximate string matching is acceptable. It implements
// the ports.Unit interface and emits OpenTelemetry spans for observability.
//
// The unit is stateless and thread-safe for concurrent execution.
type FuzzyMatchUnit struct {
	// name is the unique identifier for this unit instance.
	name string
	// config contains the validated configuration parameters.
	config FuzzyMatchConfig
	// tracer is the OpenTelemetry tracer for observability.
	tracer trace.Tracer
}

// FuzzyMatchConfig defines the configuration parameters for the FuzzyMatchUnit.
// All fields are validated during unit creation and parameter unmarshaling.
type FuzzyMatchConfig struct {
	// Algorithm specifies the fuzzy matching algorithm to use.
	// Currently only "levenshtein" is supported.
	Algorithm string `yaml:"algorithm" json:"algorithm" validate:"required,oneof=levenshtein"`

	// Threshold defines the minimum similarity score (0.0-1.0) for a match.
	// Scores below this threshold are treated as no match (0.0).
	Threshold float64 `yaml:"threshold" json:"threshold" validate:"min=0.0,max=1.0"`

	// CaseSensitive determines whether string comparison is case-sensitive.
	// When false, both strings are converted to lowercase before comparison.
	CaseSensitive bool `yaml:"case_sensitive" json:"case_sensitive"`
}

// NewFuzzyMatchUnit creates a new FuzzyMatchUnit with the specified configuration.
// The unit validates its configuration to ensure proper matching behavior.
// Returns an error if configuration validation fails.
func NewFuzzyMatchUnit(name string, config FuzzyMatchConfig) (*FuzzyMatchUnit, error) {
	if name == "" {
		return nil, ErrEmptyUnitName
	}

	if err := validate.Struct(config); err != nil {
		return nil, fmt.Errorf("configuration validation failed: %w", err)
	}

	return &FuzzyMatchUnit{
		name:   name,
		config: config,
		tracer: otel.Tracer("fuzzy-match-unit"),
	}, nil
}

// Name returns the unique identifier for this unit instance.
// The name is used for logging, debugging, and graph node referencing.
func (fmu *FuzzyMatchUnit) Name() string { return fmu.name }

// Execute performs fuzzy string matching between candidate answers and a reference answer.
// It retrieves answers and the reference answer from the state, computes similarity
// scores using the Levenshtein distance algorithm, and returns judge scores in the state.
func (fmu *FuzzyMatchUnit) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	_, span := fmu.tracer.Start(ctx, "FuzzyMatchUnit.Execute",
		trace.WithAttributes(
			attribute.String("unit.type", "fuzzy_match"),
			attribute.String("unit.id", fmu.name),
			attribute.String("config.algorithm", fmu.config.Algorithm),
			attribute.Float64("config.threshold", fmu.config.Threshold),
			attribute.Bool("config.case_sensitive", fmu.config.CaseSensitive),
		),
	)
	defer span.End()

	start := time.Now()

	// Extract candidate answers from state.
	answers, ok := domain.Get(state, domain.KeyAnswers)
	if !ok {
		err := fmt.Errorf("answers not found in state")
		span.RecordError(err)
		return state, err
	}

	if len(answers) == 0 {
		err := fmt.Errorf("no answers provided for fuzzy match evaluation")
		span.RecordError(err)
		return state, err
	}

	// Validate answer count to prevent DoS.
	if len(answers) > MaxAnswers {
		err := fmt.Errorf("too many answers: %d exceeds limit of %d", len(answers), MaxAnswers)
		span.RecordError(err)
		return state, err
	}

	// Extract reference answer from state.
	referenceAnswer, ok := domain.Get(state, domain.KeyReferenceAnswer)
	if !ok {
		err := fmt.Errorf("reference_answer required for deterministic evaluation")
		span.RecordError(err)
		return state, err
	}

	// Validate reference answer length.
	if len(referenceAnswer) > MaxStringLength {
		err := fmt.Errorf("reference answer too long: %d bytes exceeds limit of %d", len(referenceAnswer), MaxStringLength)
		span.RecordError(err)
		return state, err
	}

	// Prepare the reference answer according to configuration.
	preparedReference := fmu.prepareString(referenceAnswer)

	// Compute fuzzy match scores for each answer.
	judgeSummaries := make([]domain.JudgeSummary, len(answers))
	totalScore := 0.0

	for i, answer := range answers {
		// Validate answer length.
		if len(answer.Content) > MaxStringLength {
			err := fmt.Errorf("answer %d too long: %d bytes exceeds limit of %d", i, len(answer.Content), MaxStringLength)
			span.RecordError(err)
			return state, err
		}

		preparedAnswer := fmu.prepareString(answer.Content)
		rawSimilarity := fmu.calculateSimilarity(preparedAnswer, preparedReference)

		// Apply threshold to determine final score.
		// Raw similarity below threshold is treated as no match (0.0) to filter weak matches.
		score := rawSimilarity
		if rawSimilarity < fmu.config.Threshold {
			score = 0.0
		}

		reasoning := fmt.Sprintf("Fuzzy match similarity: %.2f%%", score*100)
		if score == 0.0 {
			reasoning = fmt.Sprintf("No match (similarity %.2f%% below threshold %.2f%%)",
				rawSimilarity*100,
				fmu.config.Threshold*100)
		}

		judgeSummaries[i] = domain.JudgeSummary{
			Score:      score,
			Reasoning:  reasoning,
			Confidence: 1.0, // Deterministic matching has perfect confidence
		}

		totalScore += score
	}

	// Calculate metrics for observability.
	latency := time.Since(start)
	avgScore := totalScore / float64(len(answers))

	// Emit OpenTelemetry attributes as required by AC#7.
	span.SetAttributes(
		attribute.Float64("eval.score", avgScore),
		attribute.Int64("eval.latency_ms", latency.Milliseconds()),
		attribute.Int("eval.answers_count", len(answers)),
		// no_llm_cost helps filter deterministic units in observability tools
		attribute.Bool("no_llm_cost", true), // Deterministic units have no LLM cost
	)

	return domain.With(state, domain.KeyJudgeScores, judgeSummaries), nil
}

// prepareString normalizes a string according to the unit's configuration.
// It applies case conversion as specified.
func (fmu *FuzzyMatchUnit) prepareString(s string) string {
	result := s

	if !fmu.config.CaseSensitive {
		// Use package-level Unicode case folder for better performance
		result = foldCaser.String(result)
	}

	return result
}

// calculateSimilarity computes the similarity score between two strings
// using the Levenshtein distance algorithm. Returns a value between 0.0 and 1.0
// where 1.0 indicates identical strings and 0.0 indicates maximum dissimilarity.
func (fmu *FuzzyMatchUnit) calculateSimilarity(s1, s2 string) float64 {
	if s1 == s2 {
		return 1.0
	}

	// Calculate Levenshtein distance (operates on runes for Unicode correctness).
	// The levenshtein library correctly handles multi-byte UTF-8 characters.
	distance := levenshtein.ComputeDistance(s1, s2)

	// Calculate maximum possible distance using rune count for correct Unicode handling.
	// The Levenshtein distance operates on runes, so we must use rune count for consistency.
	// For example: "café" has 4 runes but 5 bytes due to the é character.
	maxLen := utf8.RuneCountInString(s1)
	if n := utf8.RuneCountInString(s2); n > maxLen {
		maxLen = n
	}

	// Handle edge case where both strings are empty.
	// Two empty strings are considered identical (similarity = 1.0).
	if maxLen == 0 {
		return 1.0
	}

	// Calculate similarity as 1 - (distance / maxLength).
	// This normalizes the edit distance to a similarity score between 0 and 1.
	// Example: distance=2, maxLen=10 → similarity = 1 - (2/10) = 0.8
	similarity := 1.0 - float64(distance)/float64(maxLen)

	// Ensure similarity is within [0, 1] range due to floating-point precision.
	// This guards against edge cases in the Levenshtein implementation.
	if similarity < 0 {
		similarity = 0
	}

	return similarity
}

// Validate checks if the unit is properly configured and ready for execution.
// It validates the configuration parameters to ensure proper matching behavior.
// Returns nil if validation passes, or an error describing what is invalid.
func (fmu *FuzzyMatchUnit) Validate() error {
	if err := validate.Struct(fmu.config); err != nil {
		return fmt.Errorf("configuration validation failed: %w", err)
	}

	return nil
}

// UnmarshalParameters deserializes YAML configuration parameters and returns
// a new FuzzyMatchUnit instance to maintain thread-safety.
// This method enables YAML-based configuration with strict field validation
// to prevent configuration typos from being silently ignored.
// Returns a new unit instance or an error if YAML parsing fails or validation fails.
func (fmu *FuzzyMatchUnit) UnmarshalParameters(params yaml.Node) (*FuzzyMatchUnit, error) {
	var config FuzzyMatchConfig

	// Marshal the yaml.Node to bytes first.
	var buf bytes.Buffer
	encoder := yaml.NewEncoder(&buf)
	if err := encoder.Encode(&params); err != nil {
		return nil, fmt.Errorf("failed to encode YAML node: %w", err)
	}
	if err := encoder.Close(); err != nil {
		return nil, fmt.Errorf("failed to close YAML encoder: %w", err)
	}

	// Use strict decoder with KnownFields to catch typos.
	decoder := yaml.NewDecoder(&buf)
	decoder.KnownFields(true)

	if err := decoder.Decode(&config); err != nil {
		return nil, fmt.Errorf("failed to decode parameters (check for typos): %w", err)
	}

	// Validate the decoded configuration.
	if err := validate.Struct(config); err != nil {
		return nil, fmt.Errorf("parameter validation failed: %w", err)
	}

	// Return a new unit instance with the updated configuration.
	return &FuzzyMatchUnit{
		name:   fmu.name,
		config: config,
		tracer: fmu.tracer,
	}, nil
}

// DefaultFuzzyMatchConfig returns a FuzzyMatchConfig with sensible defaults.
func DefaultFuzzyMatchConfig() FuzzyMatchConfig {
	return FuzzyMatchConfig{
		Algorithm:     "levenshtein",
		Threshold:     0.8,
		CaseSensitive: false,
	}
}

// CreateFuzzyMatchUnit is a factory function that creates a FuzzyMatchUnit
// from a configuration map, following the UnitFactory pattern.
// This function is used by the UnitRegistry for dynamic unit creation.
func CreateFuzzyMatchUnit(id string, config map[string]any) (*FuzzyMatchUnit, error) {
	// Start with default configuration.
	matchConfig := DefaultFuzzyMatchConfig()

	// Override with provided values.
	if algorithm, ok := config["algorithm"].(string); ok {
		matchConfig.Algorithm = algorithm
	}

	if threshold, ok := config["threshold"]; ok {
		if val, ok := threshold.(float64); ok {
			matchConfig.Threshold = val
		}
	}

	if caseSensitive, ok := config["case_sensitive"].(bool); ok {
		matchConfig.CaseSensitive = caseSensitive
	}

	return NewFuzzyMatchUnit(id, matchConfig)
}
