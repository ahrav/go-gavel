package domain

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestNewState verifies that a new State instance is initialized correctly.
func TestNewState(t *testing.T) {
	state := NewState()

	assert.NotNil(t, state.data, "NewState() should initialize the data map.")
	assert.Empty(t, state.data, "NewState() should create an empty state.")
}

// TestState_Get tests the retrieval of values from a State instance.
// It covers various data types and ensures that existing keys return the correct
// values and non-existent keys are handled properly.
func TestState_Get(t *testing.T) {
	tests := []struct {
		name   string
		setup  func() State
		assert func(t *testing.T, state State)
	}{
		{
			name: "get existing string value",
			setup: func() State {
				return With(NewState(), KeyQuestion, "test question")
			},
			assert: func(t *testing.T, state State) {
				got, ok := Get(state, KeyQuestion)
				assert.True(t, ok, "Get() should find an existing key.")
				assert.Equal(t, "test question", got, "Get() returned an incorrect value.")
			},
		},
		{
			name: "get non-existent key",
			setup: func() State {
				return NewState()
			},
			assert: func(t *testing.T, state State) {
				_, ok := Get(state, KeyQuestion)
				assert.False(t, ok, "Get() should not find a non-existent key.")
			},
		},
		{
			name: "get answers array",
			setup: func() State {
				answers := []Answer{{ID: "1", Content: "A"}, {ID: "2", Content: "B"}}
				return With(NewState(), KeyAnswers, answers)
			},
			assert: func(t *testing.T, state State) {
				got, ok := Get(state, KeyAnswers)
				assert.True(t, ok, "Get() should find the answers.")
				assert.Len(t, got, 2, "Should have 2 answers.")
				assert.Equal(t, "A", got[0].Content, "First answer content mismatch.")
			},
		},
		{
			name: "get int64 budget tokens",
			setup: func() State {
				return With(NewState(), KeyBudgetTokensUsed, int64(1000))
			},
			assert: func(t *testing.T, state State) {
				got, ok := Get(state, KeyBudgetTokensUsed)
				assert.True(t, ok, "Get() should find the tokens.")
				assert.Equal(t, int64(1000), got, "Token value mismatch.")
			},
		},
		{
			name: "get verdict pointer",
			setup: func() State {
				verdict := &Verdict{ID: "v1", AggregateScore: 0.9}
				return With(NewState(), KeyVerdict, verdict)
			},
			assert: func(t *testing.T, state State) {
				got, ok := Get(state, KeyVerdict)
				assert.True(t, ok, "Get() should find the verdict.")
				assert.NotNil(t, got, "Verdict should not be nil.")
				assert.Equal(t, "v1", got.ID, "Verdict ID mismatch.")
				assert.Equal(t, 0.9, got.AggregateScore, "Verdict score mismatch.")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			state := tt.setup()
			tt.assert(t, state)
		})
	}
}

// TestState_With tests the addition of values to a State instance.
// It verifies that the operation is immutable and that new values are correctly added or updated.
func TestState_With(t *testing.T) {
	original := NewState()
	value := "test question"

	updated := With(original, KeyQuestion, value)

	_, ok := Get(original, KeyQuestion)
	assert.False(t, ok, "With() should not modify the original state.")

	got, ok := Get(updated, KeyQuestion)
	require.True(t, ok, "With() should add a new value to the state.")
	assert.Equal(t, value, got, "With() returned an incorrect value.")

	newValue := "updated question"
	updated2 := With(updated, KeyQuestion, newValue)

	v, _ := Get(updated, KeyQuestion)
	assert.Equal(t, value, v, "With() should not modify the previous state when updating.")

	v2, _ := Get(updated2, KeyQuestion)
	assert.Equal(t, newValue, v2, "With() returned an incorrect updated value.")
}

// TestState_WithMultiple tests the batch update functionality of a State instance.
// It ensures that multiple key-value pairs are added immutably and correctly.
func TestState_WithMultiple(t *testing.T) {
	original := NewState()
	answers := []Answer{{ID: "1", Content: "Option A"}, {ID: "2", Content: "Option B"}}
	judgeSummaries := []JudgeSummary{{Score: 0.8, Reasoning: "Good"}, {Score: 0.6, Reasoning: "OK"}}

	updates := map[string]any{
		KeyQuestion.name:    "Which is better?",
		KeyAnswers.name:     answers,
		KeyJudgeScores.name: judgeSummaries,
	}

	updated := original.WithMultiple(updates)

	assert.Empty(t, original.Keys(), "WithMultiple() should not modify the original state.")

	question, ok := Get(updated, KeyQuestion)
	require.True(t, ok, "WithMultiple() should apply the question update.")
	assert.Equal(t, "Which is better?", question, "Question mismatch.")

	gotAnswers, ok := Get(updated, KeyAnswers)
	require.True(t, ok, "WithMultiple() should apply the answers update.")
	assert.Len(t, gotAnswers, 2, "Answers length mismatch.")

	gotScores, ok := Get(updated, KeyJudgeScores)
	require.True(t, ok, "WithMultiple() should apply the scores update.")
	assert.Len(t, gotScores, 2, "Scores length mismatch.")
}

// TestState_Keys tests the retrieval of all keys from a State instance.
// It verifies that the correct number of keys is returned and that all expected keys are present.
func TestState_Keys(t *testing.T) {
	answers := []Answer{{ID: "1", Content: "a"}}
	judgeSummaries := []JudgeSummary{{Score: 0.5, Reasoning: "OK"}}

	state := With(With(With(NewState(),
		KeyQuestion, "q"),
		KeyAnswers, answers),
		KeyJudgeScores, judgeSummaries)

	keys := state.Keys()
	assert.Len(t, keys, 3, "Keys() should return 3 keys.")

	keyMap := make(map[string]bool)
	for _, k := range keys {
		keyMap[k] = true
	}

	assert.True(t, keyMap[KeyQuestion.name], "Keys() should include KeyQuestion.")
	assert.True(t, keyMap[KeyAnswers.name], "Keys() should include KeyAnswers.")
	assert.True(t, keyMap[KeyJudgeScores.name], "Keys() should include KeyJudgeScores.")
}

// TestState_Immutability verifies that modifications to retrieved slices or maps
// do not affect the original State instance, ensuring immutability.
func TestState_Immutability(t *testing.T) {
	answers := []Answer{{ID: "1", Content: "A"}, {ID: "2", Content: "B"}, {ID: "3", Content: "C"}}
	state := With(NewState(), KeyAnswers, answers)

	answers[0].Content = "Modified"

	retrieved, ok := Get(state, KeyAnswers)
	require.True(t, ok, "Should retrieve answers.")
	assert.NotEqual(t, "Modified", retrieved[0].Content, "State should not be affected by external slice modifications.")
}

// TestState_DeepCopy ensures that complex data types are deeply copied when added to
// or retrieved from a State instance, preventing unintended side effects.
func TestState_DeepCopy(t *testing.T) {
	var (
		keyStrings = Key[[]string]{"strings"}
		keyMap     = Key[map[string]string]{"map"}
		keyNested  = Key[[][]int]{"nested"}
		keyComplex = Key[map[string][]int]{"complex"}
		keyStruct  = Key[struct {
			Name  string
			Count int
		}]{"struct"}
		keyPtr = Key[*TraceMeta]{"ptr"}
	)

	tests := []struct {
		name     string
		setup    func() State
		modifier func(any)
		verifier func(*testing.T, any, any)
	}{
		{
			name: "slice of strings",
			setup: func() State {
				return With(NewState(), keyStrings, []string{"a", "b", "c"})
			},
			modifier: func(v any) {
				s, ok := v.([]string)
				if !ok {
					t.Fatalf("Expected []string, got %T", v)
				}
				s[0] = "modified"
			},
			verifier: func(t *testing.T, original, retrieved any) {
				orig, ok := original.([]string)
				if !ok {
					t.Fatalf("Expected original to be []string, got %T", original)
				}
				ret, ok := retrieved.([]string)
				if !ok {
					t.Fatalf("Expected retrieved to be []string, got %T", retrieved)
				}
				assert.Equal(t, "modified", orig[0], "Original should be modified.")
				assert.Equal(t, "a", ret[0], "State copy should be unchanged.")
			},
		},
		{
			name: "map of strings",
			setup: func() State {
				return With(NewState(), keyMap, map[string]string{"key1": "value1", "key2": "value2"})
			},
			modifier: func(v any) {
				m, ok := v.(map[string]string)
				if !ok {
					t.Fatalf("Expected map[string]string, got %T", v)
				}
				m["key1"] = "modified"
				m["key3"] = "new"
			},
			verifier: func(t *testing.T, original, retrieved any) {
				orig, ok := original.(map[string]string)
				if !ok {
					t.Fatalf("Expected original to be map[string]string, got %T", original)
				}
				ret, ok := retrieved.(map[string]string)
				if !ok {
					t.Fatalf("Expected retrieved to be map[string]string, got %T", retrieved)
				}
				assert.Equal(t, "modified", orig["key1"], "Original map should be modified.")
				assert.Equal(t, "value1", ret["key1"], "State map should be unchanged.")
				assert.Contains(t, orig, "key3", "Original should have a new key.")
				assert.NotContains(t, ret, "key3", "State should not have a new key.")
			},
		},
		{
			name: "nested slices",
			setup: func() State {
				return With(NewState(), keyNested, [][]int{{1, 2}, {3, 4}})
			},
			modifier: func(v any) {
				s, ok := v.([][]int)
				if !ok {
					t.Fatalf("Expected [][]int, got %T", v)
				}
				s[0][0] = 99
			},
			verifier: func(t *testing.T, original, retrieved any) {
				orig, ok := original.([][]int)
				if !ok {
					t.Fatalf("Expected original to be [][]int, got %T", original)
				}
				ret, ok := retrieved.([][]int)
				if !ok {
					t.Fatalf("Expected retrieved to be [][]int, got %T", retrieved)
				}
				assert.Equal(t, 99, orig[0][0], "Original nested slice should be modified.")
				assert.Equal(t, 1, ret[0][0], "State nested slice should be unchanged.")
			},
		},
		{
			name: "map with slice values",
			setup: func() State {
				return With(NewState(), keyComplex, map[string][]int{"a": {1, 2, 3}, "b": {4, 5, 6}})
			},
			modifier: func(v any) {
				m, ok := v.(map[string][]int)
				if !ok {
					t.Fatalf("Expected map[string][]int, got %T", v)
				}
				m["a"][0] = 99
			},
			verifier: func(t *testing.T, original, retrieved any) {
				orig, ok := original.(map[string][]int)
				if !ok {
					t.Fatalf("Expected original to be map[string][]int, got %T", original)
				}
				ret, ok := retrieved.(map[string][]int)
				if !ok {
					t.Fatalf("Expected retrieved to be map[string][]int, got %T", retrieved)
				}
				assert.Equal(t, 99, orig["a"][0], "Original map slice should be modified.")
				assert.Equal(t, 1, ret["a"][0], "State map slice should be unchanged.")
			},
		},
		{
			name: "struct value",
			setup: func() State {
				return With(NewState(), keyStruct, struct {
					Name  string
					Count int
				}{"test", 42})
			},
			modifier: func(v any) {
				// Structs are copied by value, so this won't affect the original.
			},
			verifier: func(t *testing.T, original, retrieved any) {
				assert.Equal(t, original, retrieved, "Struct values should be equal.")
			},
		},
		{
			name: "pointer to struct",
			setup: func() State {
				return With(NewState(), keyPtr, &TraceMeta{JudgeID: "judge1", Score: 0.9})
			},
			modifier: func(v any) {
				ptr, ok := v.(*TraceMeta)
				if !ok {
					t.Fatalf("Expected *TraceMeta, got %T", v)
				}
				ptr.Score = 0.5
			},
			verifier: func(t *testing.T, original, retrieved any) {
				orig, ok := original.(*TraceMeta)
				if !ok {
					t.Fatalf("Expected original to be *TraceMeta, got %T", original)
				}
				ret, ok := retrieved.(*TraceMeta)
				if !ok {
					t.Fatalf("Expected retrieved to be *TraceMeta, got %T", retrieved)
				}
				assert.Equal(t, 0.5, orig.Score, "Original pointer struct should be modified.")
				assert.Equal(t, 0.9, ret.Score, "State pointer struct should be unchanged.")
				assert.NotSame(t, orig, ret, "Pointers should be different.")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			state := tt.setup()

			var retrieved any
			var ok bool

			switch tt.name {
			case "slice of strings":
				retrieved, ok = Get(state, keyStrings)
			case "map of strings":
				retrieved, ok = Get(state, keyMap)
			case "nested slices":
				retrieved, ok = Get(state, keyNested)
			case "map with slice values":
				retrieved, ok = Get(state, keyComplex)
			case "struct value":
				retrieved, ok = Get(state, keyStruct)
			case "pointer to struct":
				retrieved, ok = Get(state, keyPtr)
			}

			require.True(t, ok, "Get() should find the value.")

			switch tt.name {
			case "slice of strings":
				original := []string{"a", "b", "c"}
				tt.modifier(original)
				tt.verifier(t, original, retrieved)
			case "map of strings":
				original := map[string]string{"key1": "value1", "key2": "value2"}
				tt.modifier(original)
				tt.verifier(t, original, retrieved)
			case "nested slices":
				original := [][]int{{1, 2}, {3, 4}}
				tt.modifier(original)
				tt.verifier(t, original, retrieved)
			case "map with slice values":
				original := map[string][]int{"a": {1, 2, 3}, "b": {4, 5, 6}}
				tt.modifier(original)
				tt.verifier(t, original, retrieved)
			case "struct value":
				original := struct {
					Name  string
					Count int
				}{"test", 42}
				tt.modifier(original)
				tt.verifier(t, original, retrieved)
			case "pointer to struct":
				original := &TraceMeta{JudgeID: "judge1", Score: 0.9}
				tt.modifier(original)
				tt.verifier(t, original, retrieved)
			}
		})
	}
}

// TestState_String tests the string representation of a State instance.
func TestState_String(t *testing.T) {
	state := With(NewState(), KeyQuestion, "test")
	str := state.String()

	assert.NotEmpty(t, str, "String() should return a non-empty representation.")
	assert.Contains(t, str, "State", "String() should contain 'State'.")
}

// TestState_ConcurrentAccess verifies the thread safety of the State instance.
// It runs concurrent reads and writes to ensure that the immutable design
// prevents race conditions.
func TestState_ConcurrentAccess(t *testing.T) {
	t.Run("concurrent reads", func(t *testing.T) {
		answers := []Answer{{ID: "1", Content: "A"}, {ID: "2", Content: "B"}, {ID: "3", Content: "C"}}
		customKey := Key[map[string]int]{"data"}

		state := With(With(With(NewState(),
			KeyQuestion, "test question"),
			KeyAnswers, answers),
			customKey, map[string]int{"count": 100})

		const numReaders = 100
		done := make(chan bool, numReaders)

		for i := 0; i < numReaders; i++ {
			go func(id int) {
				defer func() { done <- true }()

				for j := 0; j < 100; j++ {
					q, ok := Get(state, KeyQuestion)
					assert.True(t, ok, "Reader %d: Should get question.", id)
					assert.Equal(t, "test question", q, "Reader %d: Question mismatch.", id)

					a, ok := Get(state, KeyAnswers)
					assert.True(t, ok, "Reader %d: Should get answers.", id)
					assert.Len(t, a, 3, "Reader %d: Answers length mismatch.", id)

					d, ok := Get(state, customKey)
					assert.True(t, ok, "Reader %d: Should get data.", id)
					assert.NotNil(t, d, "Reader %d: Data should not be nil.", id)

					keys := state.Keys()
					assert.Len(t, keys, 3, "Reader %d: Keys length mismatch.", id)
				}
			}(i)
		}

		for i := 0; i < numReaders; i++ {
			<-done
		}
	})

	t.Run("concurrent writes create independent states", func(t *testing.T) {
		baseState := With(NewState(), KeyQuestion, "initial")

		const numWriters = 50
		states := make([]State, numWriters)
		done := make(chan bool, numWriters)

		for i := 0; i < numWriters; i++ {
			go func(id int) {
				defer func() { done <- true }()

				writerKey := Key[int]{fmt.Sprintf("writer_%d", id)}
				newState := With(baseState, writerKey, id)
				states[id] = newState

				val, ok := Get(newState, writerKey)
				assert.True(t, ok, "Writer %d: Should have its own key.", id)
				assert.Equal(t, id, val, "Writer %d: Value mismatch.", id)

				q, ok := Get(baseState, KeyQuestion)
				assert.True(t, ok, "Writer %d: Base should have the question.", id)
				assert.Equal(t, "initial", q, "Writer %d: Base question should be unchanged.", id)
			}(i)
		}

		for i := 0; i < numWriters; i++ {
			<-done
		}

		for i := 0; i < numWriters; i++ {
			keys := states[i].Keys()
			assert.Len(t, keys, 2, "State %d should have 2 keys.", i)

			for j := 0; j < numWriters; j++ {
				if i != j {
					otherKey := Key[int]{fmt.Sprintf("writer_%d", j)}
					_, ok := Get(states[i], otherKey)
					assert.False(t, ok, "State %d should not have a key from writer %d.", i, j)
				}
			}
		}
	})
}

// TestState_ExecutionContext verifies the handling of execution context within a State instance.
// It ensures that context fields are correctly set, retrieved, and initialized.
func TestState_ExecutionContext(t *testing.T) {
	ctx := ExecutionContext{
		GraphID:        "graph-123",
		EvaluationType: "comparison",
		ExecutionID:    "exec-456",
	}

	state := NewState().WithExecutionContext(ctx)

	graphID, ok := Get(state, KeyGraphID)
	require.True(t, ok, "Should have a graph ID.")
	assert.Equal(t, "graph-123", graphID, "Graph ID mismatch.")

	evalType, ok := Get(state, KeyEvaluationType)
	require.True(t, ok, "Should have an evaluation type.")
	assert.Equal(t, "comparison", evalType, "Evaluation type mismatch.")

	execID, ok := Get(state, KeyExecutionID)
	require.True(t, ok, "Should have an execution ID.")
	assert.Equal(t, "exec-456", execID, "Execution ID mismatch.")

	tokens, ok := Get(state, KeyBudgetTokensUsed)
	require.True(t, ok, "Should have a tokens counter.")
	assert.Equal(t, int64(0), tokens, "Tokens should be initialized to 0.")

	calls, ok := Get(state, KeyBudgetCallsMade)
	require.True(t, ok, "Should have a calls counter.")
	assert.Equal(t, int64(0), calls, "Calls should be initialized to 0.")

	retrievedCtx, ok := state.GetExecutionContext()
	require.True(t, ok, "Should retrieve the execution context.")
	assert.Equal(t, ctx, retrievedCtx, "Retrieved context should match.")

	emptyState := NewState()
	_, ok = emptyState.GetExecutionContext()
	assert.False(t, ok, "Should not retrieve a context from an empty state.")
}

// TestState_BudgetUsage verifies the tracking of budget usage within a State instance.
// It ensures that token and call counts are correctly updated and accumulated.
func TestState_BudgetUsage(t *testing.T) {
	ctx := ExecutionContext{
		GraphID:        "graph-123",
		EvaluationType: "comparison",
		ExecutionID:    "exec-456",
	}
	state := NewState().WithExecutionContext(ctx)

	state = state.UpdateBudgetUsage(100, 1)

	usage := state.GetBudgetUsage()
	assert.Equal(t, int64(100), usage.Tokens, "Tokens mismatch.")
	assert.Equal(t, int64(1), usage.Calls, "Calls mismatch.")

	state = state.UpdateBudgetUsage(50, 2)

	usage = state.GetBudgetUsage()
	assert.Equal(t, int64(150), usage.Tokens, "Accumulated tokens mismatch.")
	assert.Equal(t, int64(3), usage.Calls, "Accumulated calls mismatch.")

	emptyState := NewState()
	usage = emptyState.GetBudgetUsage()
	assert.Equal(t, int64(0), usage.Tokens, "Empty state tokens should be 0.")
	assert.Equal(t, int64(0), usage.Calls, "Empty state calls should be 0.")
}

// TestState_TypedKeys verifies the compile-time type safety provided by generic Keys.
// It also confirms that keys with the same name but different types will overwrite each other.
func TestState_TypedKeys(t *testing.T) {
	intKey := Key[int]{"count"}
	stringKey := Key[string]{"message"}

	state := With(With(NewState(),
		intKey, 42),
		stringKey, "forty-two")

	intVal, ok := Get(state, intKey)
	require.True(t, ok, "Should get the int value.")
	assert.Equal(t, 42, intVal, "Int value mismatch.")

	strVal, ok := Get(state, stringKey)
	require.True(t, ok, "Should get the string value.")
	assert.Equal(t, "forty-two", strVal, "String value mismatch.")

	keys := state.Keys()
	assert.Len(t, keys, 2, "Should have two distinct keys.")

	// This test verifies that a key name shared by different types results in overwriting.
	overwriteKey1 := Key[int]{"shared"}
	overwriteKey2 := Key[string]{"shared"}

	state2 := With(NewState(), overwriteKey1, 100)
	state2 = With(state2, overwriteKey2, "hundred")

	_, ok = Get(state2, overwriteKey1)
	assert.False(t, ok, "The int value should be overwritten.")

	strVal2, ok := Get(state2, overwriteKey2)
	require.True(t, ok, "The string value should exist.")
	assert.Equal(t, "hundred", strVal2, "The string value should be stored.")
}

// TestState_ComplexTypes tests the ability of a State instance to handle various
// complex data types, including time.Time, slices of maps, and pointers to structs.
func TestState_ComplexTypes(t *testing.T) {
	now := time.Now()

	timeKey := Key[time.Time]{"timestamp"}
	sliceMapKey := Key[[]map[string]any]{"complex"}

	complexData := []map[string]any{
		{"name": "test", "value": 123},
		{"name": "another", "value": 456},
	}

	state := With(With(With(NewState(),
		timeKey, now),
		sliceMapKey, complexData),
		KeyBudget, &BudgetReport{
			TotalSpent: 10.5,
			TokensUsed: 1000,
			CallsMade:  5,
		})

	gotTime, ok := Get(state, timeKey)
	require.True(t, ok, "Should get the time.")
	assert.Equal(t, now, gotTime, "Time mismatch.")

	gotComplex, ok := Get(state, sliceMapKey)
	require.True(t, ok, "Should get the complex data.")
	assert.Len(t, gotComplex, 2, "Complex data length mismatch.")

	budget, ok := Get(state, KeyBudget)
	require.True(t, ok, "Should get the budget.")
	assert.Equal(t, 10.5, budget.TotalSpent, "Budget total spent mismatch.")
	assert.Equal(t, 1000, budget.TokensUsed, "Budget tokens mismatch.")
	assert.Equal(t, 5, budget.CallsMade, "Budget calls mismatch.")
}
