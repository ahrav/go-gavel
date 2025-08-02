package domain

// Aggregator defines the interface for combining multiple judge scores
// into a single aggregate score and determining the winning answer.
// Implementations should provide different aggregation strategies
// such as arithmetic mean, weighted average, or median calculation.
type Aggregator interface {
	// Aggregate combines multiple judge scores to determine the winning
	// answer and calculate an aggregate score.
	// The scores slice contains individual judge scores for each candidate.
	// The candidates slice contains the corresponding answer objects.
	// Both slices must have the same length and be ordered consistently.
	//
	// Returns:
	//   - Answer: The winning answer based on aggregation logic
	//   - float64: The aggregate score for the winning answer
	//   - error: Any error encountered during aggregation
	//
	// The method should handle edge cases such as:
	//   - Empty score lists (return error)
	//   - NaN or infinite values (return error or filter)
	//   - Equal scores (implementation-specific tie-breaking)
	//
	// Example:
	//
	//	scores := []float64{0.8, 0.9, 0.7}
	//	candidates := []Answer{{ID: "a1", Content: "Answer 1"}, ...}
	//	winner, aggregateScore, err := aggregator.Aggregate(scores, candidates)
	Aggregate(scores []float64, candidates []Answer) (Answer, float64, error)
}
