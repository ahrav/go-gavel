// Package domain contains pure, dependency-free domain models and types
// for the evaluation engine.
package domain

import (
	"fmt"
	"maps"
	"reflect"
	"time"
)

// Key represents a type-safe generic key for accessing values in State.
// The type parameter T ensures compile-time type safety when getting and
// setting values, eliminating the need for runtime type assertions.
type Key[T any] struct{ name string }

// NewKey creates a new Key with the specified name and type.
// This function is provided for creating keys outside of the domain package.
func NewKey[T any](name string) Key[T] {
	return Key[T]{name: name}
}

// Predefined state keys used throughout the evaluation process.
// Each key is strongly typed to ensure type safety at compile time.
var (
	// KeyQuestion stores the evaluation question or prompt.
	KeyQuestion = Key[string]{"question"}

	// KeyAnswers stores the candidate answers being evaluated.
	KeyAnswers = Key[[]Answer]{"answers"}

	// KeyJudgeScores stores individual judge scoring results.
	KeyJudgeScores = Key[[]JudgeSummary]{"judge_scores"}

	// KeyVerdict stores the final verdict from aggregation.
	KeyVerdict = Key[*Verdict]{"verdict"}

	// Execution context keys for tracking metadata across graph traversal.

	// KeyGraphID stores the unique identifier of the evaluation graph being
	// executed, used for tracking and observability.
	KeyGraphID = Key[string]{"execution.graph_id"}

	// KeyEvaluationType stores the type of evaluation being performed
	// (e.g., "comparison", "scoring", "classification").
	KeyEvaluationType = Key[string]{"execution.evaluation_type"}

	// KeyExecutionID stores a unique identifier for this specific execution
	// instance, useful for tracing and correlation.
	KeyExecutionID = Key[string]{"execution.execution_id"}

	// KeyBudgetTokensUsed tracks cumulative token consumption across the
	// entire graph execution for budget management.
	KeyBudgetTokensUsed = Key[int64]{"execution.budget.tokens_used"}

	// KeyBudgetCallsMade tracks cumulative API calls made across the
	// entire graph execution for budget management.
	KeyBudgetCallsMade = Key[int64]{"execution.budget.calls_made"}

	// KeyTraceLevel stores the current trace level (e.g., "debug", "info").
	// It determines what level of detail to include in execution traces.
	KeyTraceLevel = Key[string]{"execution.trace_level"}

	// KeyVerificationTrace stores the verification unit's output when the
	// trace level is set to debug.
	KeyVerificationTrace = Key[string]{"verification_trace"}

	// KeyBudget stores the complete budget report object for tracking
	// resource consumption.
	KeyBudget = Key[*BudgetReport]{"budget"}
)

// deepCopyValue creates a deep copy of a value to ensure true immutability.
// It handles slices, maps, and other reference types that would otherwise
// allow external modification of State data.
func deepCopyValue(value any) any {
	if value == nil {
		return nil
	}

	// time.Time is immutable and can be returned directly.
	if val, ok := value.(time.Time); ok {
		return val
	}

	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Slice:
		newSlice := reflect.MakeSlice(v.Type(), v.Len(), v.Cap())
		for i := 0; i < v.Len(); i++ {
			newSlice.Index(i).Set(reflect.ValueOf(deepCopyValue(v.Index(i).Interface())))
		}
		return newSlice.Interface()

	case reflect.Map:
		newMap := reflect.MakeMap(v.Type())
		for _, key := range v.MapKeys() {
			copiedKey := deepCopyValue(key.Interface())
			copiedValue := deepCopyValue(v.MapIndex(key).Interface())
			newMap.SetMapIndex(reflect.ValueOf(copiedKey), reflect.ValueOf(copiedValue))
		}
		return newMap.Interface()

	case reflect.Ptr:
		if v.IsNil() {
			return v.Interface()
		}
		newPtr := reflect.New(v.Elem().Type())
		newPtr.Elem().Set(reflect.ValueOf(deepCopyValue(v.Elem().Interface())))
		return newPtr.Interface()

	case reflect.Struct:
		// This performs a shallow copy for unexported fields but deep copies
		// exported fields.
		newStruct := reflect.New(v.Type()).Elem()
		for i := 0; i < v.NumField(); i++ {
			if newStruct.Field(i).CanSet() {
				newStruct.Field(i).Set(reflect.ValueOf(deepCopyValue(v.Field(i).Interface())))
			}
		}
		return newStruct.Interface()

	default:
		// Primitive types are returned as-is since they are copied by value.
		return value
	}
}

// State represents an immutable collection of evaluation data that flows
// through the pipeline. It uses copy-on-write semantics to ensure
// thread-safety and prevent unintended mutations. State is the primary
// data structure for passing information between Units.
type State struct {
	// data holds the key-value pairs that make up the state.
	// It is unexported to maintain immutability guarantees.
	data map[string]any
}

// NewState creates a new empty State.
// The returned State is ready to use and can be safely shared across
// goroutines.
func NewState() State {
	return State{
		data: make(map[string]any),
	}
}

// Get retrieves a value from the State with compile-time type safety.
// It returns the value and a boolean indicating whether the key exists
// and contains a value of the correct type. The returned value is a deep
// copy to maintain immutability.
//
// Example:
//
//	question, ok := Get(state, KeyQuestion)
//	if !ok {
//	    // handle missing value
//	}
//	// question is typed as string, no type assertion needed
func Get[T any](s State, key Key[T]) (T, bool) {
	var zero T
	value, exists := s.data[key.name]
	if !exists {
		return zero, false
	}

	copied := deepCopyValue(value)
	val, ok := copied.(T)
	return val, ok
}

// GetRaw is a method version of Get that uses a string key.
// For type safety, use the generic Get function instead.
func (s State) GetRaw(keyName string) (any, bool) {
	value, exists := s.data[keyName]
	if !exists {
		return nil, false
	}
	return deepCopyValue(value), true
}

// With creates a new State with the specified key-value pair added or
// updated. It implements copy-on-write semantics, returning a new State
// instance while leaving the original unchanged. This function is the
// primary way to add or update data in a State.
//
// Example:
//
//	newState := With(state, KeyQuestion, "Which answer is better?")
func With[T any](s State, key Key[T], value T) State {
	newData := maps.Clone(s.data)
	newData[key.name] = deepCopyValue(value)
	return State{data: newData}
}

// WithRaw is a method version of With that uses a string key and allows
// chaining. For type safety, use the generic With function instead.
func (s State) WithRaw(keyName string, value any) State {
	newData := maps.Clone(s.data)
	newData[keyName] = deepCopyValue(value)
	return State{data: newData}
}

// WithMultiple creates a new State with multiple key-value pairs added
// or updated. It is more efficient than chaining multiple With calls as
// it performs a single clone operation. The updates map uses string keys
// for flexibility when updating multiple values at once.
//
// Example:
//
//	updates := map[string]any{
//	    KeyQuestion.name: "Which is better?",
//	    KeyAnswers.name: []Answer{{ID: "1", Content: "A"}},
//	}
//	newState := state.WithMultiple(updates)
func (s State) WithMultiple(updates map[string]any) State {
	newData := maps.Clone(s.data)
	for k, v := range updates {
		newData[k] = deepCopyValue(v)
	}
	return State{data: newData}
}

// Keys returns all keys present in the State.
// The returned slice can be used to iterate over all stored values and
// is safe to modify without affecting the original State.
func (s State) Keys() []string {
	keys := make([]string, 0, len(s.data))
	for k := range s.data {
		keys = append(keys, k)
	}
	return keys
}

// String returns a string representation of the State for debugging purposes.
func (s State) String() string {
	return fmt.Sprintf("State%v", s.data)
}

// ExecutionContext contains metadata about the current evaluation execution
// that flows through the State during graph traversal. It provides consistent
// access to execution metadata for middleware and observability.
type ExecutionContext struct {
	// GraphID is the unique identifier of the evaluation graph being executed.
	GraphID string

	// EvaluationType describes the type of evaluation being performed
	// (e.g., "comparison", "scoring", "classification").
	EvaluationType string

	// ExecutionID is a unique identifier for this specific execution instance,
	// useful for tracing and correlation across distributed systems.
	ExecutionID string
}

// WithExecutionContext creates a new State with execution context metadata
// included, enabling proper tracking and observability. This method should
// be called at the beginning of graph execution.
func (s State) WithExecutionContext(ctx ExecutionContext) State {
	updates := map[string]any{
		KeyGraphID.name:          ctx.GraphID,
		KeyEvaluationType.name:   ctx.EvaluationType,
		KeyExecutionID.name:      ctx.ExecutionID,
		KeyBudgetTokensUsed.name: int64(0),
		KeyBudgetCallsMade.name:  int64(0),
	}
	return s.WithMultiple(updates)
}

// GetExecutionContext extracts execution context metadata from the State.
// It returns the execution context and a boolean indicating whether all
// required context fields are present and valid.
func (s State) GetExecutionContext() (ExecutionContext, bool) {
	graphID, ok1 := Get(s, KeyGraphID)
	evaluationType, ok2 := Get(s, KeyEvaluationType)
	executionID, ok3 := Get(s, KeyExecutionID)

	if !ok1 || !ok2 || !ok3 {
		return ExecutionContext{}, false
	}

	return ExecutionContext{
		GraphID:        graphID,
		EvaluationType: evaluationType,
		ExecutionID:    executionID,
	}, true
}

// Usage tracks current resource consumption during evaluation.
// It maintains counters for tokens used and API calls made.
type Usage struct {
	// Tokens represents the cumulative token consumption.
	Tokens int64

	// Calls represents the cumulative API call count.
	Calls int64
}

// UpdateBudgetUsage creates a new State with updated budget consumption values.
// This method provides atomic updates to budget tracking and should be used
// by middleware components that track resource consumption. It increments
// the existing values rather than replacing them.
func (s State) UpdateBudgetUsage(tokensUsed, callsMade int64) State {
	currentTokens, _ := Get(s, KeyBudgetTokensUsed)
	currentCalls, _ := Get(s, KeyBudgetCallsMade)

	updates := map[string]any{
		KeyBudgetTokensUsed.name: currentTokens + tokensUsed,
		KeyBudgetCallsMade.name:  currentCalls + callsMade,
	}
	return s.WithMultiple(updates)
}

// GetBudgetUsage retrieves the current budget consumption from the State.
// It returns a Usage struct containing cumulative resource consumption,
// enabling middleware and monitoring components to access current usage.
func (s State) GetBudgetUsage() Usage {
	tokens, _ := Get(s, KeyBudgetTokensUsed)
	calls, _ := Get(s, KeyBudgetCallsMade)

	return Usage{
		Tokens: tokens,
		Calls:  calls,
	}
}
