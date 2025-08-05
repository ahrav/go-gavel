// Package llm provides a standard interface for interacting with Large Language Models from various providers.
// It includes functionality for validation, configuration, and middleware to enhance reliability and observability.
package llm

import (
	"fmt"
	"net/url"
	"time"
)

// These constants define the valid ranges for common LLM parameters.
// They are used for validation across different providers to ensure consistency.
const (
	// MinTemperature is the minimum allowed value for temperature.
	MinTemperature = 0.0
	// MaxTemperature is the maximum allowed value for temperature.
	// This is set to 2.0 to accommodate providers like Gemini.
	MaxTemperature = 2.0
	// MinTopP is the minimum allowed value for Top-P sampling.
	MinTopP = 0.0
	// MaxTopP is the maximum allowed value for Top-P sampling.
	MaxTopP = 1.0
	// MinPenalty is the minimum allowed value for frequency or presence penalties.
	MinPenalty = -2.0
	// MaxPenalty is the maximum allowed value for frequency or presence penalties.
	MaxPenalty = 2.0
	// MinMaxTokens is the minimum allowed value for the maximum number of tokens.
	MinMaxTokens = 1
	// MinTimeout is the minimum allowed duration for a request timeout.
	MinTimeout = 1 * time.Second
	// MaxTimeout is the maximum allowed duration for a request timeout.
	MaxTimeout = 10 * time.Minute
)

// IsValidTemperature checks if the temperature is within the valid range [0.0, 2.0].
func IsValidTemperature(val float64) bool {
	return val >= MinTemperature && val <= MaxTemperature
}

// IsValidTopP checks if the top_p value is within the valid range [0.0, 1.0].
func IsValidTopP(val float64) bool {
	return val >= MinTopP && val <= MaxTopP
}

// IsValidPenalty checks if the penalty value is within the valid range [-2.0, 2.0].
func IsValidPenalty(val float64) bool {
	return val >= MinPenalty && val <= MaxPenalty
}

// IsPositiveInt checks if the integer value is positive.
func IsPositiveInt(val int) bool {
	return val > 0
}

// IsNonEmptyString checks if the string is non-empty.
func IsNonEmptyString(val string) bool {
	return val != ""
}

// ValidateBaseURL validates and normalizes a base URL string.
// It ensures the URL has a valid scheme (http or https) and a host.
// An empty string is considered valid and returns no error, allowing for default URLs.
func ValidateBaseURL(baseURL string) (string, error) {
	if baseURL == "" {
		// An empty URL is valid; it signifies that the default provider URL should be used.
		return "", nil
	}

	parsedURL, err := url.Parse(baseURL)
	if err != nil {
		return "", fmt.Errorf("invalid URL format: %w", err)
	}

	if parsedURL.Scheme == "" {
		return "", fmt.Errorf("URL must include a scheme (e.g., http:// or https://)")
	}

	if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
		return "", fmt.Errorf("URL scheme must be http or https, but got: %s", parsedURL.Scheme)
	}

	if parsedURL.Host == "" {
		return "", fmt.Errorf("URL must include a host")
	}

	return parsedURL.String(), nil
}

// ValidateTimeout ensures the timeout is within a reasonable range.
// If the timeout is zero or negative, it returns zero to indicate that the default should be used.
// If it's outside the [MinTimeout, MaxTimeout] range, it clamps it to the nearest boundary.
func ValidateTimeout(timeout time.Duration) time.Duration {
	if timeout <= 0 {
		// A zero or negative timeout indicates that the system default should be used.
		return 0
	}
	if timeout < MinTimeout {
		return MinTimeout
	}
	if timeout > MaxTimeout {
		return MaxTimeout
	}
	return timeout
}

// SafeFloat32 safely converts a numeric value of type any to a float32.
// It returns the converted value and a boolean indicating success.
// The conversion fails if the value is out of the float32 range or would lose precision.
func SafeFloat32(value any) (float32, bool) {
	switch v := value.(type) {
	case float32:
		return v, true
	case float64:
		// Ensure the float64 value fits within the range of a float32.
		if v > 3.4e38 || v < -3.4e38 {
			return 0, false
		}
		return float32(v), true
	case int:
		return float32(v), true
	case int64:
		// Ensure that converting from int64 to float32 does not lose significant precision.
		// The threshold 16777216 is 2^24, the number of integers that can be exactly represented.
		if v > 16777216 || v < -16777216 {
			return 0, false
		}
		return float32(v), true
	default:
		return 0, false
	}
}

// SafeInt safely converts a numeric value of type any to an int.
// It returns the converted value and a boolean indicating success.
// The conversion fails if the value is out of the int range or is not a number (NaN).
func SafeInt(value any) (int, bool) {
	switch v := value.(type) {
	case int:
		return v, true
	case int64:
		// Check for potential overflow when converting from int64 to int.
		if int64(int(v)) != v {
			return 0, false
		}
		return int(v), true
	case float32:
		// A NaN value cannot be converted to an integer.
		if v != v {
			return 0, false
		}
		return int(v), true
	case float64:
		// A NaN value cannot be converted to an integer.
		if v != v {
			return 0, false
		}
		// Check if the float64 value is within the valid range for an integer.
		const maxInt = int(^uint(0) >> 1)
		const minInt = -maxInt - 1
		if v > float64(maxInt) || v < float64(minInt) {
			return 0, false
		}
		return int(v), true
	default:
		return 0, false
	}
}

// ClampFloat64 clamps a float64 value to be within the specified min and max range.
func ClampFloat64(val, min, max float64) float64 {
	if val < min {
		return min
	}
	if val > max {
		return max
	}
	return val
}

// ClampInt clamps an int value to be within the specified min and max range.
func ClampInt(val, min, max int) int {
	if val < min {
		return min
	}
	if val > max {
		return max
	}
	return val
}
