package domain

import (
	"time"
)

// Answer represents a candidate response being evaluated.
// Each answer has a unique identifier and its content.
type Answer struct {
	// ID uniquely identifies this answer within an evaluation.
	ID string `json:"id"`

	// Content contains the actual answer text or data.
	Content string `json:"content"`
}

// TraceMeta captures detailed execution metadata for a single judge's
// evaluation. This information is crucial for debugging, performance
// analysis, and cost tracking.
type TraceMeta struct {
	// JudgeID identifies which judge unit produced this evaluation.
	JudgeID string `json:"judge_id"`

	// Score is the numerical score assigned by this judge.
	Score float64 `json:"score"`

	// LatencyMs measures the execution time in milliseconds.
	LatencyMs int64 `json:"latency_ms"`

	// TokensUsed tracks the number of tokens consumed by this judge.
	TokensUsed int `json:"tokens_used"`

	// Summary provides detailed reasoning and confidence information.
	Summary *JudgeSummary `json:"summary,omitempty"`
}

// JudgeSummary contains qualitative information about a judge's decision.
// It provides transparency into the evaluation process.
type JudgeSummary struct {
	// Reasoning explains why the judge assigned the given score.
	Reasoning string `json:"reasoning"`

	// Confidence indicates how certain the judge is about its evaluation
	// (0.0 to 1.0).
	Confidence float64 `json:"confidence"`

	// Score is the numerical score assigned by this judge.
	// This field tracks individual judge scores for aggregation patterns.
	Score float64 `json:"score"`
}

// BudgetReport tracks resource consumption across the entire evaluation.
// It helps monitor costs and enforce resource limits.
type BudgetReport struct {
	// TotalSpent represents the total cost in dollars.
	TotalSpent float64 `json:"total_spent"`

	// TokensUsed is the cumulative token count across all operations.
	TokensUsed int `json:"tokens_used"`

	// CallsMade counts the total number of LLM API calls.
	CallsMade int `json:"calls_made"`
}

// Verdict represents the final outcome of an evaluation process.
// It contains the winning answer, aggregate scores, and detailed
// execution traces.
type Verdict struct {
	// ID uniquely identifies this verdict (typically a UUID).
	ID string `json:"id"`

	// WinnerAnswer is the answer that scored highest in the evaluation.
	// It may be nil if no clear winner could be determined.
	WinnerAnswer *Answer `json:"winner_answer,omitempty"`

	// AggregateScore is the final computed score for the winning answer.
	AggregateScore float64 `json:"aggregate_score"`

	// Trace contains detailed execution metadata for each judge.
	// It is omitted from JSON when empty to reduce payload size.
	Trace []TraceMeta `json:"trace,omitempty"`

	// Budget reports the total resources consumed during evaluation.
	// It is omitted from JSON when nil to reduce payload size.
	Budget *BudgetReport `json:"budget,omitempty"`

	// Timestamp records when this verdict was created.
	Timestamp time.Time `json:"timestamp"`
}
