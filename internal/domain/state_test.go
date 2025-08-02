package domain

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewState(t *testing.T) {
	state := NewState()

	assert.NotNil(t, state.data, "NewState() should initialize data map")
	assert.Empty(t, state.data, "NewState() should create empty state")
}

func TestState_Get(t *testing.T) {
	tests := []struct {
		name      string
		setup     func() State
		key       StateKey
		wantValue any
		wantOK    bool
	}{
		{
			name: "get existing string value",
			setup: func() State {
				return NewState().With(KeyQuestion, "test question")
			},
			key:       KeyQuestion,
			wantValue: "test question",
			wantOK:    true,
		},
		{
			name: "get non-existent key",
			setup: func() State {
				return NewState()
			},
			key:       KeyQuestion,
			wantValue: "",
			wantOK:    false,
		},
		{
			name: "get with wrong type",
			setup: func() State {
				return NewState().With(KeyQuestion, 123)
			},
			key:       KeyQuestion,
			wantValue: "",
			wantOK:    false,
		},
		{
			name: "get slice value",
			setup: func() State {
				return NewState().With(KeyAnswers, []string{"A", "B", "C"})
			},
			key:       KeyAnswers,
			wantValue: []string{"A", "B", "C"},
			wantOK:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			state := tt.setup()

			switch v := tt.wantValue.(type) {
			case string:
				got, ok := state.GetString(tt.key)
				assert.Equal(t, tt.wantOK, ok, "GetString() ok mismatch")
				assert.Equal(t, v, got, "GetString() value mismatch")
			case []string:
				got, ok := state.GetStrings(tt.key)
				assert.Equal(t, tt.wantOK, ok, "GetStrings() ok mismatch")
				if ok {
					assert.Equal(t, v, got, "GetStrings() value mismatch")
				}
			}
		})
	}
}

func TestState_With(t *testing.T) {
	original := NewState()
	key := KeyQuestion
	value := "test question"

	// Test adding new value
	updated := original.With(key, value)

	// Verify original is unchanged
	_, ok := original.GetString(key)
	assert.False(t, ok, "With() should not modify original state")

	// Verify new state has the value
	got, ok := updated.GetString(key)
	require.True(t, ok, "With() should add new value to state")
	assert.Equal(t, value, got, "With() value mismatch")

	// Test updating existing value
	newValue := "updated question"
	updated2 := updated.With(key, newValue)

	// Verify previous state unchanged
	v, _ := updated.GetString(key)
	assert.Equal(t, value, v, "With() should not modify previous state when updating")

	// Verify new state has updated value
	v2, _ := updated2.GetString(key)
	assert.Equal(t, newValue, v2, "With() updated value mismatch")
}

func TestState_WithMultiple(t *testing.T) {
	original := NewState()
	updates := map[StateKey]any{
		KeyQuestion:    "Which is better?",
		KeyAnswers:     []string{"Option A", "Option B"},
		KeyJudgeScores: map[string]float64{"judge1": 0.8, "judge2": 0.6},
	}

	updated := original.WithMultiple(updates)

	// Verify original unchanged
	assert.Empty(t, original.Keys(), "WithMultiple() should not modify original state")

	// Verify all updates applied
	question, ok := updated.GetString(KeyQuestion)
	require.True(t, ok, "WithMultiple() should apply question update")
	assert.Equal(t, "Which is better?", question, "Question mismatch")

	answers, ok := updated.GetStrings(KeyAnswers)
	require.True(t, ok, "WithMultiple() should apply answers update")
	assert.Len(t, answers, 2, "Answers length mismatch")

	scoresRaw, ok := updated.Get(KeyJudgeScores)
	require.True(t, ok, "WithMultiple() should apply scores update")
	scores, ok := scoresRaw.(map[string]float64)
	require.True(t, ok, "Scores should be map[string]float64")
	assert.Len(t, scores, 2, "Scores length mismatch")
}

func TestState_Keys(t *testing.T) {
	state := NewState().
		With(KeyQuestion, "q").
		With(KeyAnswers, []string{"a"}).
		With(KeyJudgeScores, map[string]float64{})

	keys := state.Keys()
	assert.Len(t, keys, 3, "Keys() should return 3 keys")

	// Verify all expected keys present
	keyMap := make(map[StateKey]bool)
	for _, k := range keys {
		keyMap[k] = true
	}

	assert.True(t, keyMap[KeyQuestion], "Keys() should include KeyQuestion")
	assert.True(t, keyMap[KeyAnswers], "Keys() should include KeyAnswers")
	assert.True(t, keyMap[KeyJudgeScores], "Keys() should include KeyJudgeScores")
}

func TestState_Immutability(t *testing.T) {
	// Test that modifying retrieved slices doesn't affect state
	answers := []string{"A", "B", "C"}
	state := NewState().With(KeyAnswers, answers)

	// Modify original slice
	answers[0] = "Modified"

	// Verify state is unchanged
	retrieved, ok := state.GetStrings(KeyAnswers)
	require.True(t, ok, "Should retrieve answers")
	assert.NotEqual(t, "Modified", retrieved[0], "State should not be affected by external slice modifications")
}

func TestState_DeepCopy(t *testing.T) {
	tests := []struct {
		name     string
		key      StateKey
		value    any
		modifier func(any)
		verifier func(*testing.T, any, any)
	}{
		{
			name:  "slice of strings",
			key:   StateKey("strings"),
			value: []string{"a", "b", "c"},
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
				assert.Equal(t, "modified", orig[0], "Original should be modified")
				assert.Equal(t, "a", ret[0], "State copy should be unchanged")
			},
		},
		{
			name:  "map of strings",
			key:   StateKey("map"),
			value: map[string]string{"key1": "value1", "key2": "value2"},
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
				assert.Equal(t, "modified", orig["key1"], "Original map should be modified")
				assert.Equal(t, "value1", ret["key1"], "State map should be unchanged")
				assert.Contains(t, orig, "key3", "Original should have new key")
				assert.NotContains(t, ret, "key3", "State should not have new key")
			},
		},
		{
			name:  "nested slices",
			key:   StateKey("nested"),
			value: [][]int{{1, 2}, {3, 4}},
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
				assert.Equal(t, 99, orig[0][0], "Original nested slice should be modified")
				assert.Equal(t, 1, ret[0][0], "State nested slice should be unchanged")
			},
		},
		{
			name:  "map with slice values",
			key:   StateKey("complex"),
			value: map[string][]int{"a": {1, 2, 3}, "b": {4, 5, 6}},
			modifier: func(v any) {
				m, ok := v.(map[string][]int)
				if !ok {
					t.Fatalf("Expected map[string][]int, got %T", v)
				}
				m["a"][0] = 99
				m["c"] = []int{7, 8, 9}
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
				assert.Equal(t, 99, orig["a"][0], "Original map slice should be modified")
				assert.Equal(t, 1, ret["a"][0], "State map slice should be unchanged")
				assert.Contains(t, orig, "c", "Original should have new key")
				assert.NotContains(t, ret, "c", "State should not have new key")
			},
		},
		{
			name: "struct value",
			key:  StateKey("struct"),
			value: struct {
				Name  string
				Count int
			}{"test", 42},
			modifier: func(v any) {
				// Structs are copied by value, so this won't affect original
			},
			verifier: func(t *testing.T, original, retrieved any) {
				// Just verify they're equal since structs are value types
				assert.Equal(t, original, retrieved, "Struct values should be equal")
			},
		},
		{
			name:  "pointer to struct",
			key:   StateKey("ptr"),
			value: &TraceMeta{JudgeID: "judge1", Score: 0.9},
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
				assert.Equal(t, 0.5, orig.Score, "Original pointer struct should be modified")
				assert.Equal(t, 0.9, ret.Score, "State pointer struct should be unchanged")
				assert.NotSame(t, orig, ret, "Pointers should be different")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create state with value
			state := NewState().With(tt.key, tt.value)

			// Get value from state
			retrieved, ok := state.Get(tt.key)
			require.True(t, ok, "Should retrieve value")

			// Modify the original value
			tt.modifier(tt.value)

			// Verify state is unaffected
			tt.verifier(t, tt.value, retrieved)
		})
	}
}

func TestState_String(t *testing.T) {
	state := NewState().With(KeyQuestion, "test")
	str := state.String()

	assert.NotEmpty(t, str, "String() should return non-empty representation")
	assert.Contains(t, str, "State", "String() should contain 'State'")
}

func TestState_ConcurrentAccess(t *testing.T) {
	t.Run("concurrent reads", func(t *testing.T) {
		// Create a state with some initial data
		state := NewState().
			With(KeyQuestion, "test question").
			With(KeyAnswers, []string{"A", "B", "C"}).
			With(StateKey("data"), map[string]int{"count": 100})

		// Run multiple goroutines reading concurrently
		const numReaders = 100
		done := make(chan bool, numReaders)

		for i := 0; i < numReaders; i++ {
			go func(id int) {
				defer func() { done <- true }()

				// Perform multiple reads
				for j := 0; j < 100; j++ {
					// Read different keys
					q, ok := state.GetString(KeyQuestion)
					assert.True(t, ok, "Reader %d: Should get question", id)
					assert.Equal(t, "test question", q, "Reader %d: Question mismatch", id)

					a, ok := state.GetStrings(KeyAnswers)
					assert.True(t, ok, "Reader %d: Should get answers", id)
					assert.Len(t, a, 3, "Reader %d: Answers length mismatch", id)

					d, ok := state.Get(StateKey("data"))
					assert.True(t, ok, "Reader %d: Should get data", id)
					assert.NotNil(t, d, "Reader %d: Data should not be nil", id)

					// Verify keys
					keys := state.Keys()
					assert.Len(t, keys, 3, "Reader %d: Keys length mismatch", id)
				}
			}(i)
		}

		// Wait for all readers to complete
		for i := 0; i < numReaders; i++ {
			<-done
		}
	})

	t.Run("concurrent writes create independent states", func(t *testing.T) {
		baseState := NewState().With(KeyQuestion, "initial")

		// Run multiple goroutines creating new states concurrently
		const numWriters = 50
		states := make([]State, numWriters)
		done := make(chan bool, numWriters)

		for i := 0; i < numWriters; i++ {
			go func(id int) {
				defer func() { done <- true }()

				// Each writer creates its own state based on the base
				newState := baseState.With(StateKey(fmt.Sprintf("writer_%d", id)), id)
				states[id] = newState

				// Verify the writer's own key exists
				val, ok := newState.Get(StateKey(fmt.Sprintf("writer_%d", id)))
				assert.True(t, ok, "Writer %d: Should have own key", id)
				assert.Equal(t, id, val, "Writer %d: Value mismatch", id)

				// Verify base state is unchanged
				q, ok := baseState.GetString(KeyQuestion)
				assert.True(t, ok, "Writer %d: Base should have question", id)
				assert.Equal(t, "initial", q, "Writer %d: Base question unchanged", id)
			}(i)
		}

		// Wait for all writers to complete
		for i := 0; i < numWriters; i++ {
			<-done
		}

		// Verify each state is independent
		for i := 0; i < numWriters; i++ {
			// Check that each state has only its own writer key plus the base key
			keys := states[i].Keys()
			assert.Len(t, keys, 2, "State %d should have 2 keys", i)

			// Verify no cross-contamination between states
			for j := 0; j < numWriters; j++ {
				if i != j {
					_, ok := states[i].Get(StateKey(fmt.Sprintf("writer_%d", j)))
					assert.False(t, ok, "State %d should not have writer_%d key", i, j)
				}
			}
		}
	})

	t.Run("concurrent mixed operations", func(t *testing.T) {
		// Create a base state
		state := NewState().
			With(KeyQuestion, "base question").
			With(StateKey("counter"), 0)

		const numOps = 100
		done := make(chan State, numOps)

		// Mix of readers and writers
		for i := 0; i < numOps; i++ {
			go func(id int) {
				if id%2 == 0 {
					// Reader operation
					q, _ := state.GetString(KeyQuestion)
					assert.Equal(t, "base question", q, "Reader %d: Question should be unchanged", id)
					done <- state
				} else {
					// Writer operation - creates new state
					newState := state.With(StateKey(fmt.Sprintf("op_%d", id)), id)
					done <- newState
				}
			}(i)
		}

		// Collect all resulting states
		resultStates := make([]State, numOps)
		for i := 0; i < numOps; i++ {
			resultStates[i] = <-done
		}

		// Verify original state is unchanged
		keys := state.Keys()
		assert.Len(t, keys, 2, "Original state should have only 2 keys")
		q, _ := state.GetString(KeyQuestion)
		assert.Equal(t, "base question", q, "Original question should be unchanged")
	})

	t.Run("concurrent deep copy verification", func(t *testing.T) {
		// Create a state with mutable reference types
		originalSlice := []string{"a", "b", "c"}
		originalMap := map[string]int{"x": 1, "y": 2}
		state := NewState().
			With(StateKey("slice"), originalSlice).
			With(StateKey("map"), originalMap)

		const numGoroutines = 20
		done := make(chan bool, numGoroutines)

		// Each goroutine creates its own copy and modifies it
		for i := 0; i < numGoroutines; i++ {
			go func(id int) {
				defer func() { done <- true }()

				// Get value from state and modify a local copy
				if id%2 == 0 {
					// Test slice immutability
					s, ok := state.Get(StateKey("slice"))
					assert.True(t, ok, "Worker %d: Should get slice", id)

					// Try to modify what we got (should not affect state)
					localSlice, ok := s.([]string)
					if !ok {
						t.Errorf("Worker %d: Expected []string, got %T", id, s)
						return
					}
					if len(localSlice) > 0 {
						// This modifies the local reference, not the state
						localSlice[0] = fmt.Sprintf("modified_%d", id)
					}

					// Verify state is unchanged
					s2, _ := state.Get(StateKey("slice"))
					stateSlice, ok := s2.([]string)
					if !ok {
						t.Errorf("Worker %d: Expected []string for state slice, got %T", id, s2)
						return
					}
					assert.Equal(t, []string{"a", "b", "c"}, stateSlice,
						"Worker %d: State slice should be unchanged", id)
				} else {
					// Test map immutability
					m, ok := state.Get(StateKey("map"))
					assert.True(t, ok, "Worker %d: Should get map", id)

					// Try to modify what we got (should not affect state)
					localMap, ok := m.(map[string]int)
					if !ok {
						t.Errorf("Worker %d: Expected map[string]int, got %T", id, m)
						return
					}
					localMap[fmt.Sprintf("key_%d", id)] = id

					// Verify state is unchanged
					m2, _ := state.Get(StateKey("map"))
					stateMap, ok := m2.(map[string]int)
					if !ok {
						t.Errorf("Worker %d: Expected map[string]int for state map, got %T", id, m2)
						return
					}
					assert.Equal(t, 2, len(stateMap),
						"Worker %d: State map should have only 2 keys", id)
					assert.Equal(t, 1, stateMap["x"],
						"Worker %d: Map value x should be unchanged", id)
					assert.Equal(t, 2, stateMap["y"],
						"Worker %d: Map value y should be unchanged", id)
				}
			}(i)
		}

		// Wait for all operations to complete
		for i := 0; i < numGoroutines; i++ {
			<-done
		}

		// Final verification - state should still be unchanged
		finalSlice, _ := state.Get(StateKey("slice"))
		finalSliceTyped, ok := finalSlice.([]string)
		require.True(t, ok, "Expected finalSlice to be []string")
		assert.Equal(t, []string{"a", "b", "c"}, finalSliceTyped,
			"Final state slice should be unchanged")

		finalMap, _ := state.Get(StateKey("map"))
		finalMapTyped, ok := finalMap.(map[string]int)
		require.True(t, ok, "Expected finalMap to be map[string]int")
		assert.Equal(t, map[string]int{"x": 1, "y": 2}, finalMapTyped,
			"Final state map should be unchanged")
	})

	t.Run("race detection on State methods", func(t *testing.T) {
		// This test specifically targets potential race conditions in State methods
		state := NewState()
		const numOps = 200
		done := make(chan bool, numOps)

		// Concurrent operations on the same state instance
		for i := 0; i < numOps; i++ {
			go func(id int) {
				defer func() { done <- true }()

				key := StateKey(fmt.Sprintf("key_%d", id%10))

				switch id % 5 {
				case 0:
					// With operation
					newState := state.With(key, id)
					assert.NotNil(t, newState, "Op %d: With should return non-nil state", id)
				case 1:
					// WithMultiple operation
					updates := map[StateKey]any{
						key:                                   id,
						StateKey(fmt.Sprintf("extra_%d", id)): "value",
					}
					newState := state.WithMultiple(updates)
					assert.NotNil(t, newState, "Op %d: WithMultiple should return non-nil state", id)
				case 2:
					// Get operation
					_, _ = state.Get(key)
				case 3:
					// Keys operation
					keys := state.Keys()
					assert.NotNil(t, keys, "Op %d: Keys should return non-nil slice", id)
				case 4:
					// String operation
					str := state.String()
					assert.NotEmpty(t, str, "Op %d: String should return non-empty", id)
				}
			}(i)
		}

		// Wait for all operations
		for i := 0; i < numOps; i++ {
			<-done
		}
	})
}

func TestState_TypeSafeAccessors(t *testing.T) {
	t.Run("GetInt", func(t *testing.T) {
		state := NewState().With(StateKey("count"), 42)

		// Valid int retrieval
		val, ok := state.GetInt(StateKey("count"))
		assert.True(t, ok, "GetInt should succeed")
		assert.Equal(t, 42, val, "GetInt value mismatch")

		// Non-existent key
		val, ok = state.GetInt(StateKey("missing"))
		assert.False(t, ok, "GetInt should fail for missing key")
		assert.Equal(t, 0, val, "GetInt should return zero value for missing key")

		// Wrong type
		state2 := state.With(StateKey("str"), "not an int")
		val, ok = state2.GetInt(StateKey("str"))
		assert.False(t, ok, "GetInt should fail for wrong type")
		assert.Equal(t, 0, val, "GetInt should return zero value for wrong type")
	})

	t.Run("GetInt64", func(t *testing.T) {
		var bigNum int64 = 9223372036854775807 // max int64
		state := NewState().With(StateKey("bignum"), bigNum)

		val, ok := state.GetInt64(StateKey("bignum"))
		assert.True(t, ok, "GetInt64 should succeed")
		assert.Equal(t, bigNum, val, "GetInt64 value mismatch")

		// Wrong type
		state2 := state.With(StateKey("int32"), int32(100))
		_, ok = state2.GetInt64(StateKey("int32"))
		assert.False(t, ok, "GetInt64 should fail for int32")
	})

	t.Run("GetFloat64", func(t *testing.T) {
		state := NewState().With(StateKey("pi"), 3.14159)

		val, ok := state.GetFloat64(StateKey("pi"))
		assert.True(t, ok, "GetFloat64 should succeed")
		assert.InDelta(t, 3.14159, val, 0.00001, "GetFloat64 value mismatch")

		// Wrong type
		state2 := state.With(StateKey("int"), 3)
		val, ok = state2.GetFloat64(StateKey("int"))
		assert.False(t, ok, "GetFloat64 should fail for int")
		assert.Equal(t, float64(0), val, "GetFloat64 should return zero value")
	})

	t.Run("GetBool", func(t *testing.T) {
		state := NewState().
			With(StateKey("enabled"), true).
			With(StateKey("disabled"), false)

		// True value
		val, ok := state.GetBool(StateKey("enabled"))
		assert.True(t, ok, "GetBool should succeed for true")
		assert.True(t, val, "GetBool should return true")

		// False value
		val, ok = state.GetBool(StateKey("disabled"))
		assert.True(t, ok, "GetBool should succeed for false")
		assert.False(t, val, "GetBool should return false")

		// Missing key
		val, ok = state.GetBool(StateKey("missing"))
		assert.False(t, ok, "GetBool should fail for missing key")
		assert.False(t, val, "GetBool should return false for missing key")
	})

	t.Run("GetTime", func(t *testing.T) {
		now := time.Now().Round(time.Second)
		state := NewState().With(StateKey("timestamp"), now)

		val, ok := state.GetTime(StateKey("timestamp"))
		assert.True(t, ok, "GetTime should succeed")
		assert.Equal(t, now, val, "GetTime value mismatch")

		// Zero time for missing key
		val, ok = state.GetTime(StateKey("missing"))
		assert.False(t, ok, "GetTime should fail for missing key")
		assert.True(t, val.IsZero(), "GetTime should return zero time")
	})

	t.Run("GetStringMap", func(t *testing.T) {
		original := map[string]string{
			"key1": "value1",
			"key2": "value2",
		}
		state := NewState().With(StateKey("config"), original)

		// Valid retrieval
		val, ok := state.GetStringMap(StateKey("config"))
		assert.True(t, ok, "GetStringMap should succeed")
		assert.Equal(t, original, val, "GetStringMap value mismatch")

		// Test immutability
		val["key3"] = "value3"
		original["key4"] = "value4"

		// Get again and verify original in state is unchanged
		val2, _ := state.GetStringMap(StateKey("config"))
		assert.Len(t, val2, 2, "State map should be unchanged")
		assert.NotContains(t, val2, "key3", "State should not have external modifications")
		assert.NotContains(t, val2, "key4", "State should not have original modifications")
	})

	t.Run("GetIntMap", func(t *testing.T) {
		original := map[string]int{
			"a": 1,
			"b": 2,
			"c": 3,
		}
		state := NewState().With(StateKey("scores"), original)

		val, ok := state.GetIntMap(StateKey("scores"))
		assert.True(t, ok, "GetIntMap should succeed")
		assert.Equal(t, original, val, "GetIntMap value mismatch")

		// Test immutability
		val["d"] = 4
		val2, _ := state.GetIntMap(StateKey("scores"))
		assert.Len(t, val2, 3, "State map should be unchanged")
	})

	t.Run("GetFloat64Map", func(t *testing.T) {
		original := map[string]float64{
			"weight": 0.8,
			"score":  0.95,
		}
		state := NewState().With(StateKey("metrics"), original)

		val, ok := state.GetFloat64Map(StateKey("metrics"))
		assert.True(t, ok, "GetFloat64Map should succeed")
		assert.Equal(t, original, val, "GetFloat64Map value mismatch")

		// Wrong type
		state2 := state.With(StateKey("intmap"), map[string]int{"a": 1})
		val, ok = state2.GetFloat64Map(StateKey("intmap"))
		assert.False(t, ok, "GetFloat64Map should fail for wrong map type")
		assert.Nil(t, val, "GetFloat64Map should return nil for wrong type")
	})

	t.Run("mixed types in same state", func(t *testing.T) {
		now := time.Now()
		state := NewState().
			With(StateKey("string"), "hello").
			With(StateKey("int"), 42).
			With(StateKey("float"), 3.14).
			With(StateKey("bool"), true).
			With(StateKey("time"), now).
			With(StateKey("strmap"), map[string]string{"k": "v"})

		// Verify all types can coexist
		s, ok := state.GetString(StateKey("string"))
		assert.True(t, ok)
		assert.Equal(t, "hello", s)

		i, ok := state.GetInt(StateKey("int"))
		assert.True(t, ok)
		assert.Equal(t, 42, i)

		f, ok := state.GetFloat64(StateKey("float"))
		assert.True(t, ok)
		assert.InDelta(t, 3.14, f, 0.01)

		b, ok := state.GetBool(StateKey("bool"))
		assert.True(t, ok)
		assert.True(t, b)

		tm, ok := state.GetTime(StateKey("time"))
		assert.True(t, ok)
		assert.Equal(t, now, tm)

		m, ok := state.GetStringMap(StateKey("strmap"))
		assert.True(t, ok)
		assert.Equal(t, "v", m["k"])
	})
}
