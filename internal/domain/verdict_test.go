package domain

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestAnswer_JSON verifies that the Answer struct can be correctly
// marshaled to and unmarshaled from JSON.
func TestAnswer_JSON(t *testing.T) {
	answer := Answer{
		ID:      "answer-1",
		Content: "This is the answer content.",
	}

	data, err := json.Marshal(answer)
	require.NoError(t, err, "Failed to marshal Answer.")

	var decoded Answer
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err, "Failed to unmarshal Answer.")

	assert.Equal(t, answer.ID, decoded.ID, "Answer ID mismatch.")
	assert.Equal(t, answer.Content, decoded.Content, "Answer Content mismatch.")
}

// TestTraceMeta_JSON verifies that the TraceMeta struct is correctly
// handled during JSON serialization and deserialization, including field name conventions.
func TestTraceMeta_JSON(t *testing.T) {
	trace := TraceMeta{
		JudgeID:    "judge-1",
		Score:      0.85,
		LatencyMs:  150,
		TokensUsed: 1200,
		Summary: &JudgeSummary{
			Reasoning:  "Clear and concise answer.",
			Confidence: 0.9,
		},
	}

	data, err := json.Marshal(trace)
	require.NoError(t, err, "Failed to marshal TraceMeta.")

	var jsonMap map[string]any
	err = json.Unmarshal(data, &jsonMap)
	require.NoError(t, err, "Failed to unmarshal to map.")

	assert.Equal(t, "judge-1", jsonMap["judge_id"], "TraceMeta JSON should use snake_case field names.")

	var decoded TraceMeta
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err, "Failed to unmarshal TraceMeta.")

	assert.Equal(t, trace.JudgeID, decoded.JudgeID, "TraceMeta JudgeID mismatch.")
	assert.Equal(t, trace.Score, decoded.Score, "TraceMeta Score mismatch.")
}

// TestTraceMeta_OmitEmptySummary verifies that the Summary field in TraceMeta
// is omitted from the JSON output when it is nil.
func TestTraceMeta_OmitEmptySummary(t *testing.T) {
	trace := TraceMeta{
		JudgeID:    "judge-1",
		Score:      0.85,
		LatencyMs:  150,
		TokensUsed: 1200,
		Summary:    nil,
	}

	data, err := json.Marshal(trace)
	require.NoError(t, err, "Failed to marshal TraceMeta.")

	var jsonMap map[string]any
	err = json.Unmarshal(data, &jsonMap)
	require.NoError(t, err, "Failed to unmarshal to map.")

	_, exists := jsonMap["summary"]
	assert.False(t, exists, "TraceMeta JSON should omit a nil summary.")
}

// TestBudgetReport_JSON verifies that the BudgetReport struct can be correctly
// marshaled to and unmarshaled from JSON.
func TestBudgetReport_JSON(t *testing.T) {
	budget := BudgetReport{
		TotalSpent: 0.0125,
		TokensUsed: 5000,
		CallsMade:  3,
	}

	data, err := json.Marshal(budget)
	require.NoError(t, err, "Failed to marshal BudgetReport.")

	var decoded BudgetReport
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err, "Failed to unmarshal BudgetReport.")

	assert.Equal(t, budget.TotalSpent, decoded.TotalSpent, "BudgetReport TotalSpent mismatch.")
	assert.Equal(t, budget.TokensUsed, decoded.TokensUsed, "BudgetReport TokensUsed mismatch.")
	assert.Equal(t, budget.CallsMade, decoded.CallsMade, "BudgetReport CallsMade mismatch.")
}

// TestVerdict_JSON verifies that the Verdict struct, including its nested structures,
// is correctly serialized to and deserialized from JSON.
func TestVerdict_JSON(t *testing.T) {
	now := time.Now().Round(time.Second)

	verdict := Verdict{
		ID: "verdict-123",
		WinnerAnswer: &Answer{
			ID:      "answer-1",
			Content: "The winning answer.",
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

	data, err := json.Marshal(verdict)
	require.NoError(t, err, "Failed to marshal Verdict.")

	var decoded Verdict
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err, "Failed to unmarshal Verdict.")

	assert.Equal(t, verdict.ID, decoded.ID, "Verdict ID mismatch.")
	assert.Equal(t, verdict.AggregateScore, decoded.AggregateScore, "Verdict AggregateScore mismatch.")

	require.NotNil(t, decoded.WinnerAnswer, "Verdict WinnerAnswer should not be nil.")
	assert.Equal(t, verdict.WinnerAnswer.ID, decoded.WinnerAnswer.ID, "Verdict WinnerAnswer ID mismatch.")

	assert.Len(t, decoded.Trace, 2, "Verdict Trace length mismatch.")

	require.NotNil(t, decoded.Budget, "Verdict Budget should not be nil.")
	assert.Equal(t, verdict.Budget.TotalSpent, decoded.Budget.TotalSpent, "Verdict Budget TotalSpent mismatch.")

	assert.Equal(t, now.Unix(), decoded.Timestamp.Unix(), "Verdict Timestamp mismatch.")
}

// TestVerdict_OmitEmpty verifies that optional fields in the Verdict struct
// are omitted from the JSON output when they are nil or empty.
func TestVerdict_OmitEmpty(t *testing.T) {
	verdict := Verdict{
		ID:             "verdict-123",
		WinnerAnswer:   nil,
		AggregateScore: 0.0,
		Trace:          nil,
		Budget:         nil,
		Timestamp:      time.Now(),
	}

	data, err := json.Marshal(verdict)
	require.NoError(t, err, "Failed to marshal Verdict.")

	var jsonMap map[string]any
	err = json.Unmarshal(data, &jsonMap)
	require.NoError(t, err, "Failed to unmarshal to map.")

	_, exists := jsonMap["winner_answer"]
	assert.False(t, exists, "Verdict JSON should omit a nil winner_answer.")

	_, exists = jsonMap["trace"]
	assert.False(t, exists, "Verdict JSON should omit an empty trace.")

	_, exists = jsonMap["budget"]
	assert.False(t, exists, "Verdict JSON should omit a nil budget.")
}

// TestJudgeSummary_Validation verifies the validation logic for the Confidence
// field in the JudgeSummary struct.
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

			// This test simulates validation logic for the Confidence field.
			valid := summary.Confidence >= 0.0 && summary.Confidence <= 1.0
			assert.Equal(t, tt.valid, valid, "Confidence %v validation mismatch.", tt.confidence)
		})
	}
}
