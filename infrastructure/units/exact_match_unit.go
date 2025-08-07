package units

import (
	"context"
	"fmt"
	"strings"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"golang.org/x/text/cases"
	"gopkg.in/yaml.v3"

	"github.com/ahrav/go-gavel/internal/domain"
	"github.com/ahrav/go-gavel/internal/ports"
)

var _ ports.Unit = (*ExactMatchUnit)(nil)

// Input validation constants to prevent DoS attacks.
const (
	// MaxAnswers is the maximum number of answers allowed for evaluation.
	MaxAnswers = 10000
	// MaxStringLength is the maximum allowed length for any string (10MB).
	MaxStringLength = 10 * 1024 * 1024 // 10MB
)

// ExactMatchUnit performs deterministic exact string matching between candidate
// answers and a reference answer. Each answer receives a binary score: 1.0 for
// exact matches or 0.0 for non-matches, with configurable case sensitivity and
// whitespace handling.
//
// This unit provides deterministic evaluation without LLM costs, making it ideal
// for scenarios with known ground truth answers, automated testing, or high-volume
// batch processing where consistency and speed are critical.
//
// Concurrency: ExactMatchUnit is stateless and safe for concurrent execution.
// Multiple goroutines can call Execute simultaneously without synchronization.
//
// Observability: Emits OpenTelemetry spans with evaluation scores, latency metrics,
// and deterministic processing indicators for monitoring and analysis.
type ExactMatchUnit struct {
	// name is the unique identifier for this unit instance.
	name string
	// config contains the validated configuration parameters.
	config ExactMatchConfig
	// tracer is the OpenTelemetry tracer for observability.
	tracer trace.Tracer
}

// ExactMatchConfig controls string normalization behavior during exact matching.
// The zero value provides case-insensitive matching without whitespace trimming.
//
// Configuration is immutable after unit creation and thread-safe for concurrent
// access. Changes require creating a new unit instance.
type ExactMatchConfig struct {
	// CaseSensitive controls case sensitivity during string comparison.
	// When false, uses Unicode-aware case folding for proper internationalization.
	// Default: false (case-insensitive matching).
	CaseSensitive bool `yaml:"case_sensitive" json:"case_sensitive"`

	// TrimWhitespace controls leading/trailing whitespace normalization.
	// When true, applies strings.TrimSpace before comparison.
	// Default: true (whitespace is trimmed).
	TrimWhitespace bool `yaml:"trim_whitespace" json:"trim_whitespace"`
}

// NewExactMatchUnit creates a new ExactMatchUnit with validated configuration.
// The unit is immediately ready for concurrent execution after successful creation.
//
// The name parameter serves as a unique identifier for logging, debugging, and
// observability spans. It must be non-empty.
//
// Returns ErrEmptyUnitName if name is empty, or a configuration validation error
// if the config struct fails validation constraints.
func NewExactMatchUnit(name string, config ExactMatchConfig) (*ExactMatchUnit, error) {
	if name == "" {
		return nil, ErrEmptyUnitName
	}

	if err := validate.Struct(config); err != nil {
		return nil, fmt.Errorf("configuration validation failed: %w", err)
	}

	return &ExactMatchUnit{
		name:   name,
		config: config,
		tracer: otel.Tracer("exact-match-unit"),
	}, nil
}

// Name returns the unique identifier for this unit instance.
// The returned value is immutable and safe for concurrent access.
func (emu *ExactMatchUnit) Name() string { return emu.name }

// Execute performs exact string matching evaluation on the provided state.
// It extracts candidate answers and a reference answer from the state, applies
// configured normalization (case folding, whitespace trimming), and computes
// binary match scores.
//
// State requirements:
//   - domain.KeyAnswers: []domain.Answer with candidate responses
//   - domain.KeyReferenceAnswer: string containing the ground truth
//
// Returns a new state containing domain.KeyJudgeScores with match results.
// Each JudgeSummary contains a score of 1.0 (exact match) or 0.0 (no match),
// deterministic reasoning text, and confidence of 1.0.
//
// Errors:
//   - Missing or empty answers in state
//   - Missing reference answer in state
//   - Answer count exceeds MaxAnswers limit
//   - Answer or reference content exceeds MaxStringLength limit
//   - Context cancellation during processing
//
// The function is safe for concurrent execution and does not modify the input state.
func (emu *ExactMatchUnit) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	_, span := emu.tracer.Start(ctx, "ExactMatchUnit.Execute",
		trace.WithAttributes(
			attribute.String("unit.type", "exact_match"),
			attribute.String("unit.id", emu.name),
			attribute.Bool("config.case_sensitive", emu.config.CaseSensitive),
			attribute.Bool("config.trim_whitespace", emu.config.TrimWhitespace),
		),
	)
	defer span.End()

	start := time.Now()

	// Extract candidate answers from state.
	// This is a required input for deterministic evaluation.
	answers, ok := domain.Get(state, domain.KeyAnswers)
	if !ok {
		err := fmt.Errorf("answers not found in state")
		span.RecordError(err)
		return state, err
	}

	if len(answers) == 0 {
		err := fmt.Errorf("no answers provided for exact match evaluation")
		span.RecordError(err)
		return state, err
	}

	// Validate answer count to prevent DoS attacks.
	// Large answer sets could consume excessive memory and processing time.
	if len(answers) > MaxAnswers {
		err := fmt.Errorf("too many answers: %d exceeds limit of %d", len(answers), MaxAnswers)
		span.RecordError(err)
		return state, err
	}

	// Extract reference answer from state.
	// Ground truth answer is mandatory for exact matching evaluation.
	referenceAnswer, ok := domain.Get(state, domain.KeyReferenceAnswer)
	if !ok {
		err := fmt.Errorf("reference_answer required for deterministic evaluation")
		span.RecordError(err)
		return state, err
	}

	// Validate reference answer length to prevent resource exhaustion.
	// Extremely long strings could cause memory pressure during processing.
	if len(referenceAnswer) > MaxStringLength {
		err := fmt.Errorf("reference answer too long: %d bytes exceeds limit of %d", len(referenceAnswer), MaxStringLength)
		span.RecordError(err)
		return state, err
	}

	// Prepare the reference answer according to configuration.
	// Apply case folding and whitespace normalization once for efficiency.
	preparedReference := emu.prepareString(referenceAnswer)

	judgeSummaries := make([]domain.JudgeSummary, len(answers))
	totalScore := 0.0

	for i, answer := range answers {
		if len(answer.Content) > MaxStringLength {
			err := fmt.Errorf("answer %d too long: %d bytes exceeds limit of %d", i, len(answer.Content), MaxStringLength)
			span.RecordError(err)
			return state, err
		}

		preparedAnswer := emu.prepareString(answer.Content)
		score := 0.0
		reasoning := "No exact match"

		if preparedAnswer == preparedReference {
			score = 1.0
			reasoning = "Exact match found"
		}

		judgeSummaries[i] = domain.JudgeSummary{
			Score:      score,
			Reasoning:  reasoning,
			Confidence: 1.0, // Deterministic matching has perfect confidence
		}

		totalScore += score
	}

	latency := time.Since(start)
	avgScore := totalScore / float64(len(answers))

	span.SetAttributes(
		attribute.Float64("eval.score", avgScore),
		attribute.Int64("eval.latency_ms", latency.Milliseconds()),
		attribute.Int("eval.answers_count", len(answers)),
		// no_llm_cost helps filter deterministic units in observability tools.
		// This attribute enables cost analysis and performance monitoring.
		attribute.Bool("no_llm_cost", true), // Deterministic units have no LLM cost
	)

	return domain.With(state, domain.KeyJudgeScores, judgeSummaries), nil
}

// prepareString normalizes a string according to the unit's configuration.
// Applies transformations in order: whitespace trimming, then case folding.
// Uses Unicode-aware case folding for proper internationalization support.
func (emu *ExactMatchUnit) prepareString(s string) string {
	result := s

	if emu.config.TrimWhitespace {
		result = strings.TrimSpace(result)
	}

	if !emu.config.CaseSensitive {
		// Use Unicode-aware case folding for proper internationalization.
		// This handles complex Unicode characters correctly, unlike strings.ToLower.
		caser := cases.Fold()
		result = caser.String(result)
	}

	return result
}

// Validate verifies the unit is properly configured and ready for execution.
// This method can be called at any time to check unit health and is safe
// for concurrent use.
//
// Returns nil if the unit is valid, or a descriptive error if configuration
// constraints are violated. Currently validates the embedded config struct
// using the validator package.
func (emu *ExactMatchUnit) Validate() error {
	if err := validate.Struct(emu.config); err != nil {
		return fmt.Errorf("configuration validation failed: %w", err)
	}

	return nil
}

// UnmarshalParameters deserializes YAML configuration into the unit's config.
// This method enables dynamic configuration from YAML sources with validation
// to ensure parameter correctness.
//
// The method performs strict decoding to catch configuration errors early,
// followed by struct validation using the validator package. Successfully
// decoded configuration immediately replaces the unit's current config.
//
// Returns an error if YAML parsing fails or the decoded configuration fails
// validation constraints. The unit's configuration remains unchanged on error.
func (emu *ExactMatchUnit) UnmarshalParameters(params yaml.Node) error {
	var config ExactMatchConfig

	if err := params.Decode(&config); err != nil {
		return fmt.Errorf("failed to decode parameters: %w", err)
	}

	if err := validate.Struct(config); err != nil {
		return fmt.Errorf("parameter validation failed: %w", err)
	}

	emu.config = config
	return nil
}

// DefaultExactMatchConfig returns an ExactMatchConfig with production-ready defaults:
// case-insensitive matching with whitespace trimming enabled for robust text comparison.
func DefaultExactMatchConfig() ExactMatchConfig {
	return ExactMatchConfig{
		CaseSensitive:  false,
		TrimWhitespace: true,
	}
}

// CreateExactMatchUnit creates an ExactMatchUnit from a configuration map.
// This factory function follows the UnitFactory pattern for dynamic unit
// instantiation by the unit registry system.
//
// The config map supports the following optional keys:
//   - "case_sensitive" (bool): controls case sensitivity
//   - "trim_whitespace" (bool): controls whitespace trimming
//
// Missing keys default to DefaultExactMatchConfig values. Invalid values
// are silently ignored, preserving defaults for robustness.
//
// Returns an error only if the id parameter is empty or unit creation fails
// during validation.

// NewExactMatchFromConfig creates an ExactMatchUnit from a configuration map.
// This is the boundary adapter for YAML/JSON configuration.
// Exact match doesn't require an LLM client.
func NewExactMatchFromConfig(id string, config map[string]any, llm ports.LLMClient) (ports.Unit, error) {
	// llm is ignored - exact match is deterministic.

	data, err := yaml.Marshal(config)
	if err != nil {
		return nil, fmt.Errorf("marshal config: %w", err)
	}

	// Start with defaults, then overlay user config.
	cfg := DefaultExactMatchConfig()
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	return NewExactMatchUnit(id, cfg)
}
