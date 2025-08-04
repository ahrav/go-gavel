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

// Configuration validation functions

// IsPositiveInt returns true if the integer is greater than 0.
func IsPositiveInt(val int) bool { return val > 0 }

// IsNonEmptyString returns true if the string is not empty.
func IsNonEmptyString(val string) bool { return val != "" }

// IsValidTemperature returns true if the float is a valid temperature (0.0 to 1.0).
func IsValidTemperature(val float64) bool { return val >= 0.0 && val <= 1.0 }

// IsValidTopP returns true if the float is a valid top_p value (0.0 to 1.0).
func IsValidTopP(val float64) bool { return val >= 0.0 && val <= 1.0 }
