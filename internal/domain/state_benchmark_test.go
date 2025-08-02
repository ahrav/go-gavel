package domain

import (
	"fmt"
	"testing"
	"time"
)

// BenchmarkState_Get benchmarks retrieving values from State
func BenchmarkState_Get(b *testing.B) {
	// Create a state with various types of data
	state := NewState().
		With(KeyQuestion, "test question").
		With(KeyAnswers, []string{"A", "B", "C", "D", "E"}).
		With(StateKey("count"), 42).
		With(StateKey("score"), 0.95).
		With(StateKey("enabled"), true).
		With(StateKey("data"), map[string]int{
			"a": 1, "b": 2, "c": 3, "d": 4, "e": 5,
		})

	b.Run("GetString", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			_, _ = state.GetString(KeyQuestion)
		}
	})

	b.Run("GetStrings", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			_, _ = state.GetStrings(KeyAnswers)
		}
	})

	b.Run("GetInt", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			_, _ = state.GetInt(StateKey("count"))
		}
	})

	b.Run("GetFloat64", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			_, _ = state.GetFloat64(StateKey("score"))
		}
	})

	b.Run("GetBool", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			_, _ = state.GetBool(StateKey("enabled"))
		}
	})

	b.Run("GetIntMap", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			_, _ = state.GetIntMap(StateKey("data"))
		}
	})

	b.Run("Get_NonExistent", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			_, _ = state.Get(StateKey("nonexistent"))
		}
	})
}

// BenchmarkState_With benchmarks adding values to State
func BenchmarkState_With(b *testing.B) {
	baseState := NewState()

	b.Run("With_String", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			_ = baseState.With(KeyQuestion, "test question")
		}
	})

	b.Run("With_Slice", func(b *testing.B) {
		slice := []string{"A", "B", "C", "D", "E"}
		b.ResetTimer()
		for b.Loop() {
			_ = baseState.With(KeyAnswers, slice)
		}
	})

	b.Run("With_Map", func(b *testing.B) {
		m := map[string]int{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
		b.ResetTimer()
		for b.Loop() {
			_ = baseState.With(StateKey("data"), m)
		}
	})

	b.Run("With_LargeSlice", func(b *testing.B) {
		// Create a larger slice to test deep copy performance
		largeSlice := make([]string, 1000)
		for i := range largeSlice {
			largeSlice[i] = fmt.Sprintf("item_%d", i)
		}
		b.ResetTimer()
		for b.Loop() {
			_ = baseState.With(StateKey("large"), largeSlice)
		}
	})

	b.Run("With_LargeMap", func(b *testing.B) {
		// Create a larger map to test deep copy performance
		largeMap := make(map[string]int, 1000)
		for i := 0; i < 1000; i++ {
			largeMap[fmt.Sprintf("key_%d", i)] = i
		}
		b.ResetTimer()
		for b.Loop() {
			_ = baseState.With(StateKey("largemap"), largeMap)
		}
	})
}

// BenchmarkState_WithMultiple benchmarks batch updates
func BenchmarkState_WithMultiple(b *testing.B) {
	baseState := NewState()

	b.Run("WithMultiple_Small", func(b *testing.B) {
		updates := map[StateKey]any{
			KeyQuestion:       "test",
			KeyAnswers:        []string{"A", "B"},
			StateKey("count"): 42,
		}
		b.ResetTimer()
		for b.Loop() {
			_ = baseState.WithMultiple(updates)
		}
	})

	b.Run("WithMultiple_Large", func(b *testing.B) {
		// Create a large update map
		updates := make(map[StateKey]any)
		for i := 0; i < 50; i++ {
			updates[StateKey(fmt.Sprintf("key_%d", i))] = i
		}
		b.ResetTimer()
		for b.Loop() {
			_ = baseState.WithMultiple(updates)
		}
	})
}

// BenchmarkState_DeepCopy benchmarks the deep copy functionality
func BenchmarkState_DeepCopy(b *testing.B) {
	b.Run("DeepCopy_String", func(b *testing.B) {
		value := "test string"
		b.ResetTimer()
		for b.Loop() {
			_ = deepCopyValue(value)
		}
	})

	b.Run("DeepCopy_SmallSlice", func(b *testing.B) {
		value := []string{"A", "B", "C", "D", "E"}
		b.ResetTimer()
		for b.Loop() {
			_ = deepCopyValue(value)
		}
	})

	b.Run("DeepCopy_LargeSlice", func(b *testing.B) {
		value := make([]string, 1000)
		for i := range value {
			value[i] = fmt.Sprintf("item_%d", i)
		}
		b.ResetTimer()
		for b.Loop() {
			_ = deepCopyValue(value)
		}
	})

	b.Run("DeepCopy_SmallMap", func(b *testing.B) {
		value := map[string]int{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
		b.ResetTimer()
		for b.Loop() {
			_ = deepCopyValue(value)
		}
	})

	b.Run("DeepCopy_LargeMap", func(b *testing.B) {
		value := make(map[string]int, 1000)
		for i := 0; i < 1000; i++ {
			value[fmt.Sprintf("key_%d", i)] = i
		}
		b.ResetTimer()
		for b.Loop() {
			_ = deepCopyValue(value)
		}
	})

	b.Run("DeepCopy_NestedMap", func(b *testing.B) {
		value := map[string][]int{
			"a": {1, 2, 3},
			"b": {4, 5, 6},
			"c": {7, 8, 9},
		}
		b.ResetTimer()
		for b.Loop() {
			_ = deepCopyValue(value)
		}
	})

	b.Run("DeepCopy_Time", func(b *testing.B) {
		value := time.Now()
		b.ResetTimer()
		for b.Loop() {
			_ = deepCopyValue(value)
		}
	})
}

// BenchmarkState_Keys benchmarks the Keys operation
func BenchmarkState_Keys(b *testing.B) {
	b.Run("Keys_Small", func(b *testing.B) {
		state := NewState().
			With(KeyQuestion, "test").
			With(KeyAnswers, []string{"A", "B"}).
			With(StateKey("count"), 42)

		b.ResetTimer()
		for b.Loop() {
			_ = state.Keys()
		}
	})

	b.Run("Keys_Large", func(b *testing.B) {
		state := NewState()
		for i := range 100 {
			state = state.With(StateKey(fmt.Sprintf("key_%d", i)), i)
		}

		b.ResetTimer()
		for b.Loop() {
			_ = state.Keys()
		}
	})
}

// BenchmarkState_ConcurrentRead benchmarks concurrent read operations
func BenchmarkState_ConcurrentRead(b *testing.B) {
	state := NewState().
		With(KeyQuestion, "test question").
		With(KeyAnswers, []string{"A", "B", "C", "D", "E"}).
		With(StateKey("data"), map[string]int{
			"a": 1, "b": 2, "c": 3, "d": 4, "e": 5,
		})

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// Mix of different read operations
			_, _ = state.GetString(KeyQuestion)
			_, _ = state.GetStrings(KeyAnswers)
			_, _ = state.Get(StateKey("data"))
		}
	})
}

// BenchmarkState_ConcurrentWrite benchmarks concurrent write operations
func BenchmarkState_ConcurrentWrite(b *testing.B) {
	baseState := NewState()

	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			// Each goroutine creates its own state
			key := StateKey(fmt.Sprintf("key_%d", i))
			_ = baseState.With(key, i)
			i++
		}
	})
}
