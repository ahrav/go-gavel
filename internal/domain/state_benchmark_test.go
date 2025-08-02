package domain

import (
	"fmt"
	"testing"
	"time"
)

// BenchmarkState_Get benchmarks retrieving values from State
func BenchmarkState_Get(b *testing.B) {
	// Define custom keys for benchmarking
	var (
		countKey   = Key[int]{"count"}
		scoreKey   = Key[float64]{"score"}
		enabledKey = Key[bool]{"enabled"}
		dataKey    = Key[map[string]int]{"data"}
	)

	// Create a state with various types of data
	answers := []Answer{{ID: "1", Content: "A"}, {ID: "2", Content: "B"}, {ID: "3", Content: "C"}, {ID: "4", Content: "D"}, {ID: "5", Content: "E"}}
	state := With(
		With(
			With(
				With(
					With(
						With(NewState(), KeyQuestion, "test question"),
						KeyAnswers, answers),
					countKey, 42),
				scoreKey, 0.95),
			enabledKey, true),
		dataKey, map[string]int{
			"a": 1, "b": 2, "c": 3, "d": 4, "e": 5,
		})

	b.Run("Get_String", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			_, _ = Get(state, KeyQuestion)
		}
	})

	b.Run("Get_Slice", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			_, _ = Get(state, KeyAnswers)
		}
	})

	b.Run("Get_Int", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			_, _ = Get(state, countKey)
		}
	})

	b.Run("Get_Float64", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			_, _ = Get(state, scoreKey)
		}
	})

	b.Run("Get_Bool", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			_, _ = Get(state, enabledKey)
		}
	})

	b.Run("Get_Map", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			_, _ = Get(state, dataKey)
		}
	})

	b.Run("Get_NonExistent", func(b *testing.B) {
		nonExistentKey := Key[string]{"nonexistent"}
		b.ResetTimer()
		for b.Loop() {
			_, _ = Get(state, nonExistentKey)
		}
	})
}

// BenchmarkState_With benchmarks adding values to State
func BenchmarkState_With(b *testing.B) {
	baseState := NewState()

	b.Run("With_String", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			_ = With(baseState, KeyQuestion, "test question")
		}
	})

	b.Run("With_Slice", func(b *testing.B) {
		answers := []Answer{{ID: "1", Content: "A"}, {ID: "2", Content: "B"}, {ID: "3", Content: "C"}, {ID: "4", Content: "D"}, {ID: "5", Content: "E"}}
		b.ResetTimer()
		for b.Loop() {
			_ = With(baseState, KeyAnswers, answers)
		}
	})

	b.Run("With_Map", func(b *testing.B) {
		dataKey := Key[map[string]int]{"data"}
		m := map[string]int{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
		b.ResetTimer()
		for b.Loop() {
			_ = With(baseState, dataKey, m)
		}
	})

	b.Run("With_LargeSlice", func(b *testing.B) {
		// Create a larger slice to test deep copy performance
		largeKey := Key[[]string]{"large"}
		largeSlice := make([]string, 1000)
		for i := range largeSlice {
			largeSlice[i] = fmt.Sprintf("item_%d", i)
		}
		b.ResetTimer()
		for b.Loop() {
			_ = With(baseState, largeKey, largeSlice)
		}
	})

	b.Run("With_LargeMap", func(b *testing.B) {
		// Create a larger map to test deep copy performance
		largeMapKey := Key[map[string]int]{"largemap"}
		largeMap := make(map[string]int, 1000)
		for i := 0; i < 1000; i++ {
			largeMap[fmt.Sprintf("key_%d", i)] = i
		}
		b.ResetTimer()
		for b.Loop() {
			_ = With(baseState, largeMapKey, largeMap)
		}
	})
}

// BenchmarkState_WithMultiple benchmarks batch updates
func BenchmarkState_WithMultiple(b *testing.B) {
	baseState := NewState()

	b.Run("WithMultiple_Small", func(b *testing.B) {
		answers := []Answer{{ID: "1", Content: "A"}, {ID: "2", Content: "B"}}
		updates := map[string]any{
			KeyQuestion.name: "test",
			KeyAnswers.name:  answers,
			"count":          42,
		}
		b.ResetTimer()
		for b.Loop() {
			_ = baseState.WithMultiple(updates)
		}
	})

	b.Run("WithMultiple_Large", func(b *testing.B) {
		// Create a large update map
		updates := make(map[string]any)
		for i := 0; i < 50; i++ {
			updates[fmt.Sprintf("key_%d", i)] = i
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

	b.Run("DeepCopy_Answer", func(b *testing.B) {
		value := Answer{ID: "1", Content: "Test answer content"}
		b.ResetTimer()
		for b.Loop() {
			_ = deepCopyValue(value)
		}
	})

	b.Run("DeepCopy_JudgeSummary", func(b *testing.B) {
		value := JudgeSummary{
			Reasoning:  "This is a good answer because...",
			Confidence: 0.95,
			Score:      8.5,
		}
		b.ResetTimer()
		for b.Loop() {
			_ = deepCopyValue(value)
		}
	})
}

// BenchmarkState_Keys benchmarks the Keys operation
func BenchmarkState_Keys(b *testing.B) {
	b.Run("Keys_Small", func(b *testing.B) {
		answers := []Answer{{ID: "1", Content: "A"}, {ID: "2", Content: "B"}}
		state := With(
			With(
				With(NewState(), KeyQuestion, "test"),
				KeyAnswers, answers),
			KeyBudgetTokensUsed, int64(42))

		b.ResetTimer()
		for b.Loop() {
			_ = state.Keys()
		}
	})

	b.Run("Keys_Large", func(b *testing.B) {
		state := NewState()
		for i := range 100 {
			key := Key[int]{fmt.Sprintf("key_%d", i)}
			state = With(state, key, i)
		}

		b.ResetTimer()
		for b.Loop() {
			_ = state.Keys()
		}
	})
}

// BenchmarkState_ConcurrentRead benchmarks concurrent read operations
func BenchmarkState_ConcurrentRead(b *testing.B) {
	answers := []Answer{{ID: "1", Content: "A"}, {ID: "2", Content: "B"}, {ID: "3", Content: "C"}, {ID: "4", Content: "D"}, {ID: "5", Content: "E"}}
	dataKey := Key[map[string]int]{"data"}

	state := With(
		With(
			With(NewState(), KeyQuestion, "test question"),
			KeyAnswers, answers),
		dataKey, map[string]int{
			"a": 1, "b": 2, "c": 3, "d": 4, "e": 5,
		})

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			// Mix of different read operations
			_, _ = Get(state, KeyQuestion)
			_, _ = Get(state, KeyAnswers)
			_, _ = Get(state, dataKey)
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
			key := Key[int]{fmt.Sprintf("key_%d", i)}
			_ = With(baseState, key, i)
			i++
		}
	})
}

// BenchmarkState_GenericVsSpecific compares the performance of generic Get vs old specific getters
// This benchmark demonstrates that the generic approach has similar performance characteristics
func BenchmarkState_GenericVsSpecific(b *testing.B) {
	// Create state with different typed values
	verdict := &Verdict{ID: "v1", AggregateScore: 0.95}
	answers := []Answer{{ID: "1", Content: "A"}, {ID: "2", Content: "B"}}
	judgeSummaries := []JudgeSummary{{Score: 0.8, Reasoning: "Good", Confidence: 0.9}}

	state := With(
		With(
			With(
				With(
					With(
						With(NewState(), KeyQuestion, "test question"),
						KeyAnswers, answers),
					KeyJudgeScores, judgeSummaries),
				KeyVerdict, verdict),
			KeyBudgetTokensUsed, int64(1000)),
		KeyTraceLevel, "debug")

	// Benchmark the generic Get method
	b.Run("Generic_Get_String", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			v, ok := Get(state, KeyQuestion)
			if !ok {
				b.Fatal("key not found")
			}
			_ = v // Use to prevent optimization
		}
	})

	b.Run("Generic_Get_Slice", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			v, ok := Get(state, KeyAnswers)
			if !ok {
				b.Fatal("key not found")
			}
			_ = v
		}
	})

	b.Run("Generic_Get_Pointer", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			v, ok := Get(state, KeyVerdict)
			if !ok {
				b.Fatal("key not found")
			}
			_ = v
		}
	})

	b.Run("Generic_Get_Int64", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			v, ok := Get(state, KeyBudgetTokensUsed)
			if !ok {
				b.Fatal("key not found")
			}
			_ = v
		}
	})
}

// BenchmarkState_TypeSafety demonstrates that the generic approach provides compile-time type safety
// without runtime type assertions in user code
func BenchmarkState_TypeSafety(b *testing.B) {
	answers := []Answer{{ID: "1", Content: "A"}, {ID: "2", Content: "B"}}
	state := With(NewState(), KeyAnswers, answers)

	b.Run("TypeSafe_Access", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			// With generics, we get the correct type directly
			answers, ok := Get(state, KeyAnswers)
			if !ok {
				b.Fatal("key not found")
			}
			// No type assertion needed - answers is already []Answer
			if len(answers) != 2 {
				b.Fatal("unexpected length")
			}
		}
	})
}
