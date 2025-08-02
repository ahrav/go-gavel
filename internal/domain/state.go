// Package domain contains pure, dependency-free domain models and types
// for the evaluation engine.
package domain

import (
	"fmt"
	"maps"
	"reflect"
	"time"
)

// StateKey represents a type-safe key for accessing values in State.
// Using a distinct type prevents accidental key collisions and provides
// compile-time safety when working with State data.
type StateKey string

// Predefined state keys used throughout the evaluation process.
const (
	// KeyQuestion stores the evaluation question or prompt.
	KeyQuestion StateKey = "question"

	// KeyAnswers stores the candidate answers being evaluated.
	KeyAnswers StateKey = "answers"

	// KeyJudgeScores stores individual judge scoring results.
	KeyJudgeScores StateKey = "judge_scores"

	// KeyVerdict stores the final verdict from aggregation.
	KeyVerdict StateKey = "verdict"

	// Execution context keys for tracking metadata across graph traversal.

	// KeyGraphID stores the unique identifier of the evaluation graph
	// being executed, used for tracking and observability.
	KeyGraphID StateKey = "execution.graph_id"

	// KeyEvaluationType stores the type of evaluation being performed
	// (e.g., "comparison", "scoring", "classification").
	KeyEvaluationType StateKey = "execution.evaluation_type"

	// KeyExecutionID stores a unique identifier for this specific
	// execution instance, useful for tracing and correlation.
	KeyExecutionID StateKey = "execution.execution_id"

	// KeyBudgetTokensUsed tracks cumulative token consumption across
	// the entire graph execution for budget management.
	KeyBudgetTokensUsed StateKey = "execution.budget.tokens_used"

	// KeyBudgetCallsMade tracks cumulative API calls made across
	// the entire graph execution for budget management.
	KeyBudgetCallsMade StateKey = "execution.budget.calls_made"
)

// deepCopyValue creates a deep copy of a value to ensure true immutability.
// It handles slices, maps, and other reference types that would otherwise
// allow external modification of State data.
func deepCopyValue(value any) any {
	if value == nil {
		return nil
	}

	// Handle special cases first.
	switch val := value.(type) {
	case time.Time:
		// time.Time is immutable, so we can return it directly.
		return val
	}

	// Use reflection to handle different types.
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Slice:
		// Create a new slice with the same type and length.
		newSlice := reflect.MakeSlice(v.Type(), v.Len(), v.Cap())
		for i := 0; i < v.Len(); i++ {
			// Recursively deep copy each element.
			newSlice.Index(i).Set(reflect.ValueOf(deepCopyValue(v.Index(i).Interface())))
		}
		return newSlice.Interface()

	case reflect.Map:
		// Create a new map with the same type.
		newMap := reflect.MakeMap(v.Type())
		for _, key := range v.MapKeys() {
			// Deep copy both key and value.
			copiedKey := deepCopyValue(key.Interface())
			copiedValue := deepCopyValue(v.MapIndex(key).Interface())
			newMap.SetMapIndex(reflect.ValueOf(copiedKey), reflect.ValueOf(copiedValue))
		}
		return newMap.Interface()

	case reflect.Ptr:
		// Handle pointers by creating a new pointer to a deep copy.
		if v.IsNil() {
			return v.Interface()
		}
		newPtr := reflect.New(v.Elem().Type())
		newPtr.Elem().Set(reflect.ValueOf(deepCopyValue(v.Elem().Interface())))
		return newPtr.Interface()

	case reflect.Struct:
		// For structs, create a new instance and copy fields.
		// Note: This doesn't handle unexported fields.
		newStruct := reflect.New(v.Type()).Elem()
		for i := 0; i < v.NumField(); i++ {
			field := v.Field(i)
			newField := newStruct.Field(i)
			// Skip fields we can't set (unexported) or access (private interfaces).
			if newField.CanSet() && field.CanInterface() {
				newField.Set(reflect.ValueOf(deepCopyValue(field.Interface())))
			}
		}
		return newStruct.Interface()

	default:
		// For primitive types (int, string, bool, etc.), return as-is.
		// These are already copied by value.
		return value
	}
}

// State represents an immutable collection of evaluation data that flows
// through the pipeline.
// It uses copy-on-write semantics to ensure thread-safety and prevent
// unintended mutations.
// State is the primary data structure for passing information between
// Units in the evaluation process.
type State struct {
	// data holds the key-value pairs that make up the state.
	// It is never exposed directly to maintain immutability guarantees.
	data map[StateKey]any
}

// NewState creates a new empty State.
// The returned State is ready to use and can be safely shared across
// goroutines.
func NewState() State {
	return State{
		data: make(map[StateKey]any),
	}
}

// Get retrieves a value from the State and returns it as an interface{}.
// It returns the value and a boolean indicating whether the key exists.
// This method provides type-safe access to State values.
// The returned value is a deep copy to maintain immutability.
//
// Example:
//
//	var question string
//	question, ok := state.Get(KeyQuestion).(string)
//	if !ok {
//	    // handle missing value or wrong type
//	}
func (s State) Get(key StateKey) (any, bool) {
	value, exists := s.data[key]
	if !exists {
		return nil, false
	}
	// Return a deep copy to prevent external modifications.
	return deepCopyValue(value), true
}

// GetString retrieves a string value from the State.
// It returns the value and a boolean indicating whether the key exists
// and contains a string value.
func (s State) GetString(key StateKey) (string, bool) {
	value, exists := s.data[key]
	if !exists {
		return "", false
	}
	str, ok := value.(string)
	return str, ok
}

// GetStrings retrieves a string slice value from the State.
// It returns the value and a boolean indicating whether the key exists
// and contains a string slice value.
// The returned slice is a deep copy to maintain immutability.
func (s State) GetStrings(key StateKey) ([]string, bool) {
	value, exists := s.data[key]
	if !exists {
		return nil, false
	}
	strs, ok := value.([]string)
	if !ok {
		return nil, false
	}
	// Return a copy to prevent external modifications.
	copied := make([]string, len(strs))
	copy(copied, strs)
	return copied, true
}

// GetInt retrieves an int value from the State.
// It returns the value and a boolean indicating whether the key exists
// and contains an int value.
func (s State) GetInt(key StateKey) (int, bool) {
	value, exists := s.data[key]
	if !exists {
		return 0, false
	}
	i, ok := value.(int)
	return i, ok
}

// GetInt64 retrieves an int64 value from the State.
// It returns the value and a boolean indicating whether the key exists
// and contains an int64 value.
func (s State) GetInt64(key StateKey) (int64, bool) {
	value, exists := s.data[key]
	if !exists {
		return 0, false
	}
	i, ok := value.(int64)
	return i, ok
}

// GetFloat64 retrieves a float64 value from the State.
// It returns the value and a boolean indicating whether the key exists
// and contains a float64 value.
func (s State) GetFloat64(key StateKey) (float64, bool) {
	value, exists := s.data[key]
	if !exists {
		return 0, false
	}
	f, ok := value.(float64)
	return f, ok
}

// GetBool retrieves a bool value from the State.
// It returns the value and a boolean indicating whether the key exists
// and contains a bool value.
func (s State) GetBool(key StateKey) (bool, bool) {
	value, exists := s.data[key]
	if !exists {
		return false, false
	}
	b, ok := value.(bool)
	return b, ok
}

// GetTime retrieves a time.Time value from the State.
// It returns the value and a boolean indicating whether the key exists
// and contains a time.Time value.
func (s State) GetTime(key StateKey) (time.Time, bool) {
	value, exists := s.data[key]
	if !exists {
		return time.Time{}, false
	}
	t, ok := value.(time.Time)
	return t, ok
}

// GetStringMap retrieves a map[string]string value from the State.
// It returns the value and a boolean indicating whether the key exists
// and contains a map[string]string value.
// The returned map is a deep copy to maintain immutability.
func (s State) GetStringMap(key StateKey) (map[string]string, bool) {
	value, exists := s.data[key]
	if !exists {
		return nil, false
	}
	m, ok := value.(map[string]string)
	if !ok {
		return nil, false
	}
	// Return a copy to prevent external modifications.
	copied := make(map[string]string, len(m))
	maps.Copy(copied, m)
	return copied, true
}

// GetIntMap retrieves a map[string]int value from the State.
// It returns the value and a boolean indicating whether the key exists
// and contains a map[string]int value.
// The returned map is a deep copy to maintain immutability.
func (s State) GetIntMap(key StateKey) (map[string]int, bool) {
	value, exists := s.data[key]
	if !exists {
		return nil, false
	}
	m, ok := value.(map[string]int)
	if !ok {
		return nil, false
	}
	// Return a copy to prevent external modifications.
	copied := make(map[string]int, len(m))
	maps.Copy(copied, m)
	return copied, true
}

// GetFloat64Map retrieves a map[string]float64 value from the State.
// It returns the value and a boolean indicating whether the key exists
// and contains a map[string]float64 value.
// The returned map is a deep copy to maintain immutability.
func (s State) GetFloat64Map(key StateKey) (map[string]float64, bool) {
	value, exists := s.data[key]
	if !exists {
		return nil, false
	}
	m, ok := value.(map[string]float64)
	if !ok {
		return nil, false
	}
	// Return a copy to prevent external modifications.
	copied := make(map[string]float64, len(m))
	maps.Copy(copied, m)
	return copied, true
}

// With creates a new State with the specified key-value pair added or
// updated.
// It implements copy-on-write semantics, returning a new State instance
// while leaving the original unchanged.
// This method is the primary way to add or update data in a State.
//
// Example:
//
//	newState := state.With(KeyQuestion, "Which answer is better?")
func (s State) With(key StateKey, value any) State {
	newData := maps.Clone(s.data)
	newData[key] = deepCopyValue(value)
	return State{data: newData}
}

// WithMultiple creates a new State with multiple key-value pairs added
// or updated.
// It is more efficient than chaining multiple With calls as it performs
// a single clone operation.
// The updates map specifies all key-value pairs to be added or updated.
//
// Example:
//
//	updates := map[StateKey]any{
//	    KeyQuestion: "Which is better?",
//	    KeyAnswers: []string{"A", "B"},
//	}
//	newState := state.WithMultiple(updates)
func (s State) WithMultiple(updates map[StateKey]any) State {
	newData := maps.Clone(s.data)
	for k, v := range updates {
		newData[k] = deepCopyValue(v)
	}
	return State{data: newData}
}

// Keys returns all keys present in the State.
// The returned slice can be used to iterate over all stored values and
// is safe to modify without affecting the original State.
func (s State) Keys() []StateKey {
	keys := make([]StateKey, 0, len(s.data))
	for k := range s.data {
		keys = append(keys, k)
	}
	return keys
}

// String returns a string representation of the State for debugging purposes.
// It includes all key-value pairs in a readable format.
func (s State) String() string {
	return fmt.Sprintf("State%v", s.data)
}

// ExecutionContext contains metadata about the current evaluation execution
// that flows through the State during graph traversal.
// It provides consistent access to execution metadata for middleware,
// metrics collection, and observability.
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
// included, enabling proper tracking and observability throughout graph execution.
// This method should be called at the beginning of graph execution to establish
// the execution context for all subsequent middleware and units.
func (s State) WithExecutionContext(ctx ExecutionContext) State {
	updates := map[StateKey]any{
		KeyGraphID:        ctx.GraphID,
		KeyEvaluationType: ctx.EvaluationType,
		KeyExecutionID:    ctx.ExecutionID,
		// Initialize budget counters to zero for new execution
		KeyBudgetTokensUsed: 0,
		KeyBudgetCallsMade:  0,
	}
	return s.WithMultiple(updates)
}

// GetExecutionContext extracts execution context metadata from the State.
// It returns the execution context and a boolean indicating whether all
// required context fields are present and valid.
// This method is used by middleware and observability components to access
// execution metadata.
func (s State) GetExecutionContext() (ExecutionContext, bool) {
	graphID, ok1 := s.GetString(KeyGraphID)
	if !ok1 {
		return ExecutionContext{}, false
	}

	evaluationType, ok2 := s.GetString(KeyEvaluationType)
	if !ok2 {
		return ExecutionContext{}, false
	}

	executionID, ok3 := s.GetString(KeyExecutionID)
	if !ok3 {
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
// by middleware components that track resource consumption during execution.
// It increments the existing values rather than replacing them.
func (s State) UpdateBudgetUsage(tokensUsed, callsMade int64) State {
	// Get current values or default to 0
	currentTokens, _ := s.GetInt64(KeyBudgetTokensUsed)
	currentCalls, _ := s.GetInt64(KeyBudgetCallsMade)

	updates := map[StateKey]any{
		KeyBudgetTokensUsed: currentTokens + tokensUsed,
		KeyBudgetCallsMade:  currentCalls + callsMade,
	}
	return s.WithMultiple(updates)
}

// GetBudgetUsage retrieves the current budget consumption from the State.
// It returns a Usage struct containing cumulative resource consumption.
// This method enables middleware and monitoring components to access
// current resource consumption levels.
func (s State) GetBudgetUsage() Usage {
	tokens, _ := s.GetInt64(KeyBudgetTokensUsed)
	calls, _ := s.GetInt64(KeyBudgetCallsMade)

	return Usage{
		Tokens: tokens,
		Calls:  calls,
	}
}
