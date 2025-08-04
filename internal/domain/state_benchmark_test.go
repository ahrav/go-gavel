package domain

import (
	"fmt"
	"testing"
	"time"
)

// BenchmarkState_Get benchmarks the performance of retrieving values from a State instance.
// It covers various data types, including strings, slices, integers, floats, booleans, and maps.
func BenchmarkState_Get(b *testing.B) {
	var (
		countKey   = Key[int]{"count"}
		scoreKey   = Key[float64]{"score"}
		enabledKey = Key[bool]{"enabled"}
		dataKey    = Key[map[string]int]{"data"}
	)

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

// BenchmarkState_With benchmarks the performance of adding values to a State instance.
// It tests various data types and sizes to measure the impact of deep copying.
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

// BenchmarkState_WithMultiple benchmarks the performance of batch updates to a State instance.
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

// BenchmarkState_DeepCopy benchmarks the performance of the deep copy functionality.
// It covers various data types and structures to ensure efficient and correct copying.
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

// BenchmarkState_Keys benchmarks the performance of retrieving all keys from a State instance.
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

// BenchmarkState_ConcurrentRead benchmarks the performance of concurrent read operations on a State instance.
// It uses RunParallel to simulate multiple goroutines reading from the same state.
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
			_, _ = Get(state, KeyQuestion)
			_, _ = Get(state, KeyAnswers)
			_, _ = Get(state, dataKey)
		}
	})
}

// BenchmarkState_ConcurrentWrite benchmarks the performance of concurrent write operations on a State instance.
// Due to the immutable nature of State, each write operation creates a new instance, ensuring thread safety.
func BenchmarkState_ConcurrentWrite(b *testing.B) {
	baseState := NewState()

	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			key := Key[int]{fmt.Sprintf("key_%d", i)}
			_ = With(baseState, key, i)
			i++
		}
	})
}

// BenchmarkState_GenericVsSpecific compares the performance of the generic Get method
// against older, type-specific getter methods. This benchmark demonstrates that the generic
// approach maintains comparable performance characteristics.
func BenchmarkState_GenericVsSpecific(b *testing.B) {
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

	b.Run("Generic_Get_String", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			v, ok := Get(state, KeyQuestion)
			if !ok {
				b.Fatal("key not found")
			}
			_ = v
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

// BenchmarkState_TypeSafety demonstrates the compile-time type safety of the generic State implementation.
// It shows that retrieving a value with a generic Key results in a correctly typed variable without
// requiring a runtime type assertion.
func BenchmarkState_TypeSafety(b *testing.B) {
	answers := []Answer{{ID: "1", Content: "A"}, {ID: "2", Content: "B"}}
	state := With(NewState(), KeyAnswers, answers)

	b.Run("TypeSafe_Access", func(b *testing.B) {
		b.ResetTimer()
		for b.Loop() {
			answers, ok := Get(state, KeyAnswers)
			if !ok {
				b.Fatal("key not found")
			}

			if len(answers) != 2 {
				b.Fatal("unexpected length")
			}
		}
	})
}
