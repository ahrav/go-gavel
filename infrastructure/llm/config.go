package llm

// config.go provides configuration parsing and validation utilities
// for LLM provider options. This file contains functions for extracting
// and validating parameters from generic option maps used across providers.

// ExtractOptionalInt extracts an integer value from options map with validation.
// Returns defaultVal if key doesn't exist, value is not an int, or validator fails.
func ExtractOptionalInt(opts map[string]any, key string, defaultVal int, validator func(int) bool) int {
	if opts == nil {
		return defaultVal
	}

	val, ok := opts[key]
	if !ok {
		return defaultVal
	}

	intVal, ok := val.(int)
	if !ok {
		return defaultVal
	}

	if validator != nil && !validator(intVal) {
		return defaultVal
	}

	return intVal
}

// ExtractOptionalString extracts a string value from options map with validation.
// Returns defaultVal if key doesn't exist, value is not a string, or validator fails.
func ExtractOptionalString(opts map[string]any, key string, defaultVal string, validator func(string) bool) string {
	if opts == nil {
		return defaultVal
	}

	val, ok := opts[key]
	if !ok {
		return defaultVal
	}

	strVal, ok := val.(string)
	if !ok {
		return defaultVal
	}

	if validator != nil && !validator(strVal) {
		return defaultVal
	}

	return strVal
}

// ExtractOptionalFloat64 extracts a float64 value from options map with validation.
// Returns defaultVal if key doesn't exist, value is not a float64, or validator fails.
func ExtractOptionalFloat64(opts map[string]any, key string, defaultVal float64, validator func(float64) bool) float64 {
	if opts == nil {
		return defaultVal
	}

	val, ok := opts[key]
	if !ok {
		return defaultVal
	}

	floatVal, ok := val.(float64)
	if !ok {
		return defaultVal
	}

	if validator != nil && !validator(floatVal) {
		return defaultVal
	}

	return floatVal
}
