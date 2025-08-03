// Package middleware provides cross-cutting concerns for the evaluation engine.
package middleware

import (
	"context"
	"fmt"
	"slices"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/ahrav/go-gavel/internal/domain"
	"github.com/ahrav/go-gavel/internal/ports"
)

var _ ports.Unit = (*PositionSwapMiddleware)(nil)

// PositionSwapMiddleware mitigates positional bias by executing a judge Unit
// twice with reversed candidate answer order, then combining the scores using
// an arithmetic mean. This stateless middleware follows the decorator pattern
// and integrates with OpenTelemetry for observability.
type PositionSwapMiddleware struct {
	next ports.Unit
	name string
}

// NewPositionSwapMiddleware creates a new PositionSwapMiddleware instance that
// wraps the specified judge unit. The middleware is stateless and thread-safe.
func NewPositionSwapMiddleware(next ports.Unit, name string) *PositionSwapMiddleware {
	if next == nil {
		panic("position swap middleware: next unit is required")
	}
	if name == "" {
		panic("position swap middleware: name is required")
	}
	return &PositionSwapMiddleware{next: next, name: name}
}

// Name returns the unique identifier for this middleware instance.
func (psm *PositionSwapMiddleware) Name() string { return psm.name }

// startSpan creates a new OpenTelemetry span with common attributes.
func (psm *PositionSwapMiddleware) startSpan(ctx context.Context, name string, attrs ...attribute.KeyValue) (context.Context, trace.Span) {
	tracer := otel.Tracer("position-swap-middleware")
	ctx, span := tracer.Start(ctx, name)

	span.SetAttributes(
		attribute.String("middleware.name", psm.name),
		attribute.String("middleware.type", "position_swap"),
	)
	span.SetAttributes(attrs...)

	return ctx, span
}

// Execute performs bias mitigation by executing the wrapped judge twice with
// reversed answer order and combining the scores. It is thread-safe due to
// its stateless design and immutable State operations.
func (psm *PositionSwapMiddleware) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	ctx, span := psm.startSpan(ctx, "PositionSwapMiddleware.Execute",
		attribute.String("wrapped_unit.name", psm.next.Name()))
	defer span.End()

	answers, ok := domain.Get(state, domain.KeyAnswers)
	if !ok {
		err := fmt.Errorf("answers not found in state")
		span.SetStatus(codes.Error, err.Error())
		return state, err
	}
	if len(answers) == 0 {
		err := fmt.Errorf("answers cannot be empty")
		span.SetStatus(codes.Error, err.Error())
		return state, err
	}

	// If there's only one answer, positional bias is not a concern, so we
	// just pass through to the wrapped unit.
	if len(answers) == 1 {
		return psm.next.Execute(ctx, state)
	}

	span.AddEvent("dual_execution_started", trace.WithAttributes(
		attribute.Int("answer_count", len(answers)),
	))

	firstResult, err := psm.executeWrappedUnit(ctx, state, answers, 0)
	if err != nil {
		return state, fmt.Errorf("first execution failed: %w", err)
	}

	reversedAnswers := make([]domain.Answer, len(answers))
	copy(reversedAnswers, answers)
	slices.Reverse(reversedAnswers)
	stateWithReversedAnswers := domain.With(firstResult, domain.KeyAnswers, reversedAnswers)

	secondResult, err := psm.executeWrappedUnit(ctx, stateWithReversedAnswers, reversedAnswers, 1)
	if err != nil {
		return state, fmt.Errorf("second execution failed: %w", err)
	}

	combinedResult, err := psm.combineScores(firstResult, secondResult, answers)
	if err != nil {
		return state, fmt.Errorf("score combination failed: %w", err)
	}

	span.AddEvent("bias_mitigation_completed", trace.WithAttributes(
		attribute.String("combination_method", "arithmetic_mean"),
	))
	span.SetStatus(codes.Ok, "Position swap bias mitigation completed successfully")
	return combinedResult, nil
}

// executeWrappedUnit executes the wrapped judge unit with OpenTelemetry tracing.
func (psm *PositionSwapMiddleware) executeWrappedUnit(
	ctx context.Context,
	state domain.State,
	answers []domain.Answer,
	runIndex int,
) (domain.State, error) {
	ctx, span := psm.startSpan(ctx, fmt.Sprintf("PositionSwapMiddleware.Run%d", runIndex),
		attribute.Int("run_index", runIndex),
		attribute.String("unit.name", psm.next.Name()))
	defer span.End()

	answerIDs := make([]string, len(answers))
	for i, answer := range answers {
		answerIDs[i] = answer.ID
	}
	span.SetAttributes(attribute.StringSlice("answer_order", answerIDs))

	result, err := psm.next.Execute(ctx, state)
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		return state, err
	}

	span.SetStatus(codes.Ok, "Unit execution completed successfully")
	return result, nil
}

// combineScores combines judge scores from two execution runs using an
// arithmetic mean. It preserves the original answer order in the final result.
func (psm *PositionSwapMiddleware) combineScores(
	firstResult, secondResult domain.State,
	originalAnswers []domain.Answer,
) (domain.State, error) {
	firstScores, ok1 := domain.Get(firstResult, domain.KeyJudgeScores)
	secondScores, ok2 := domain.Get(secondResult, domain.KeyJudgeScores)
	if !ok1 || !ok2 {
		return firstResult, fmt.Errorf("judge scores not found in execution results")
	}
	if len(firstScores) != len(originalAnswers) || len(secondScores) != len(originalAnswers) {
		return firstResult, fmt.Errorf("score count mismatch: expected %d, got first=%d, second=%d",
			len(originalAnswers), len(firstScores), len(secondScores))
	}

	// Since second execution had reversed answers, reverse the scores back
	reversedSecondScores := make([]domain.JudgeSummary, len(secondScores))
	for i := range secondScores {
		reversedSecondScores[i] = secondScores[len(secondScores)-1-i]
	}

	combinedScores := make([]domain.JudgeSummary, len(originalAnswers))
	for i := range originalAnswers {
		firstScore := firstScores[i].Score
		secondScore := reversedSecondScores[i].Score
		meanScore := (firstScore + secondScore) / 2.0

		combinedScores[i] = domain.JudgeSummary{
			Reasoning: fmt.Sprintf("Position swap: (%.3f + %.3f) / 2 = %.3f",
				firstScore, secondScore, meanScore),
			Confidence: (firstScores[i].Confidence + reversedSecondScores[i].Confidence) / 2.0,
			Score:      meanScore,
		}
	}

	result := domain.With(firstResult, domain.KeyAnswers, originalAnswers)
	return domain.With(result, domain.KeyJudgeScores, combinedScores), nil
}

// Validate checks if the PositionSwapMiddleware is properly configured by
// delegating validation to the wrapped unit.
func (psm *PositionSwapMiddleware) Validate() error {
	if psm.next == nil {
		return fmt.Errorf("next unit is required")
	}
	if psm.name == "" {
		return fmt.Errorf("name is required")
	}
	if err := psm.next.Validate(); err != nil {
		return fmt.Errorf("wrapped unit validation failed: %w", err)
	}
	return nil
}
