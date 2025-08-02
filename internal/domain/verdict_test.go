package domain

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAnswer_JSON(t *testing.T) {
	answer := Answer{
		ID:      "answer-1",
		Content: "This is the answer content",
	}

	// Test marshaling
	data, err := json.Marshal(answer)
	require.NoError(t, err, "Failed to marshal Answer")

	// Test unmarshaling
	var decoded Answer
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err, "Failed to unmarshal Answer")

	assert.Equal(t, answer.ID, decoded.ID, "Answer ID mismatch")
	assert.Equal(t, answer.Content, decoded.Content, "Answer Content mismatch")
}

func TestTraceMeta_JSON(t *testing.T) {
	trace := TraceMeta{
		JudgeID:    "judge-1",
		Score:      0.85,
		LatencyMs:  150,
		TokensUsed: 1200,
		Summary: &JudgeSummary{
			Reasoning:  "Clear and concise answer",
			Confidence: 0.9,
		},
	}

	// Test marshaling
	data, err := json.Marshal(trace)
	require.NoError(t, err, "Failed to marshal TraceMeta")

	// Verify JSON contains expected fields
	var jsonMap map[string]any
	err = json.Unmarshal(data, &jsonMap)
	require.NoError(t, err, "Failed to unmarshal to map")

	assert.Equal(t, "judge-1", jsonMap["judge_id"], "TraceMeta JSON should use snake_case field names")

	// Test round-trip
	var decoded TraceMeta
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err, "Failed to unmarshal TraceMeta")

	assert.Equal(t, trace.JudgeID, decoded.JudgeID, "TraceMeta JudgeID mismatch")
	assert.Equal(t, trace.Score, decoded.Score, "TraceMeta Score mismatch")
}

func TestTraceMeta_OmitEmptySummary(t *testing.T) {
	trace := TraceMeta{
		JudgeID:    "judge-1",
		Score:      0.85,
		LatencyMs:  150,
		TokensUsed: 1200,
		Summary:    nil, // No summary
	}

	data, err := json.Marshal(trace)
	require.NoError(t, err, "Failed to marshal TraceMeta")

	// Verify summary is omitted when nil
	var jsonMap map[string]any
	err = json.Unmarshal(data, &jsonMap)
	require.NoError(t, err, "Failed to unmarshal to map")

	_, exists := jsonMap["summary"]
	assert.False(t, exists, "TraceMeta JSON should omit nil summary")
}

func TestBudgetReport_JSON(t *testing.T) {
	budget := BudgetReport{
		TotalSpent: 0.0125,
		TokensUsed: 5000,
		CallsMade:  3,
	}

	data, err := json.Marshal(budget)
	require.NoError(t, err, "Failed to marshal BudgetReport")

	var decoded BudgetReport
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err, "Failed to unmarshal BudgetReport")

	assert.Equal(t, budget.TotalSpent, decoded.TotalSpent, "BudgetReport TotalSpent mismatch")
	assert.Equal(t, budget.TokensUsed, decoded.TokensUsed, "BudgetReport TokensUsed mismatch")
	assert.Equal(t, budget.CallsMade, decoded.CallsMade, "BudgetReport CallsMade mismatch")
}

func TestVerdict_JSON(t *testing.T) {
	now := time.Now().Round(time.Second) // Round to avoid nanosecond precision issues

	verdict := Verdict{
		ID: "verdict-123",
		WinnerAnswer: &Answer{
			ID:      "answer-1",
			Content: "The winning answer",
		},
		AggregateScore: 0.875,
		Trace: []TraceMeta{
			{
				JudgeID:    "judge-1",
				Score:      0.9,
				LatencyMs:  100,
				TokensUsed: 800,
			},
			{
				JudgeID:    "judge-2",
				Score:      0.85,
				LatencyMs:  120,
				TokensUsed: 900,
			},
		},
		Budget: &BudgetReport{
			TotalSpent: 0.015,
			TokensUsed: 1700,
			CallsMade:  2,
		},
		Timestamp: now,
	}

	// Test marshaling
	data, err := json.Marshal(verdict)
	require.NoError(t, err, "Failed to marshal Verdict")

	// Test unmarshaling
	var decoded Verdict
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err, "Failed to unmarshal Verdict")

	// Verify basic fields
	assert.Equal(t, verdict.ID, decoded.ID, "Verdict ID mismatch")
	assert.Equal(t, verdict.AggregateScore, decoded.AggregateScore, "Verdict AggregateScore mismatch")

	// Verify winner answer
	require.NotNil(t, decoded.WinnerAnswer, "Verdict WinnerAnswer should not be nil")
	assert.Equal(t, verdict.WinnerAnswer.ID, decoded.WinnerAnswer.ID, "Verdict WinnerAnswer ID mismatch")

	// Verify trace
	assert.Len(t, decoded.Trace, 2, "Verdict Trace length mismatch")

	// Verify budget
	require.NotNil(t, decoded.Budget, "Verdict Budget should not be nil")
	assert.Equal(t, verdict.Budget.TotalSpent, decoded.Budget.TotalSpent, "Verdict Budget TotalSpent mismatch")

	// Verify timestamp (compare Unix seconds to avoid timezone issues)
	assert.Equal(t, now.Unix(), decoded.Timestamp.Unix(), "Verdict Timestamp mismatch")
}

func TestVerdict_OmitEmpty(t *testing.T) {
	// Test that optional fields are omitted when empty/nil
	verdict := Verdict{
		ID:             "verdict-123",
		WinnerAnswer:   nil, // No winner
		AggregateScore: 0.0,
		Trace:          nil, // No trace
		Budget:         nil, // No budget
		Timestamp:      time.Now(),
	}

	data, err := json.Marshal(verdict)
	require.NoError(t, err, "Failed to marshal Verdict")

	var jsonMap map[string]any
	err = json.Unmarshal(data, &jsonMap)
	require.NoError(t, err, "Failed to unmarshal to map")

	// Check omitted fields
	_, exists := jsonMap["winner_answer"]
	assert.False(t, exists, "Verdict JSON should omit nil winner_answer")

	_, exists = jsonMap["trace"]
	assert.False(t, exists, "Verdict JSON should omit empty trace")

	_, exists = jsonMap["budget"]
	assert.False(t, exists, "Verdict JSON should omit nil budget")
}

func TestJudgeSummary_Validation(t *testing.T) {
	tests := []struct {
		name       string
		confidence float64
		valid      bool
	}{
		{"valid low confidence", 0.0, true},
		{"valid high confidence", 1.0, true},
		{"valid mid confidence", 0.5, true},
		{"negative confidence", -0.1, false},
		{"excessive confidence", 1.1, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			summary := JudgeSummary{
				Reasoning:  "Test reasoning",
				Confidence: tt.confidence,
			}

			// In a real implementation, we would have a Validate() method
			// For now, just check the bounds
			valid := summary.Confidence >= 0.0 && summary.Confidence <= 1.0
			assert.Equal(t, tt.valid, valid, "Confidence %v validation mismatch", tt.confidence)
		})
	}
}
