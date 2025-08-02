package application

import (
	"fmt"
	"slices"
	"strings"

	"github.com/go-playground/validator/v10"
	"gopkg.in/yaml.v3"
)

// ValidateUnitParameters validates the parameters for a specific unit type,
// ensuring all required fields are present and values meet domain constraints.
// ValidateUnitParameters supports llm_judge, code_analyzer,
// metrics_collector, and custom unit types with type-specific validation rules.
// ValidateUnitParameters returns an error if parameter decoding fails
// or if any validation rule is violated.
func ValidateUnitParameters(unitType string, params yaml.Node) error {
	var paramMap map[string]any
	if err := params.Decode(&paramMap); err != nil {
		return fmt.Errorf("failed to decode parameters: %w", err)
	}

	switch unitType {
	case "llm_judge":
		return validateLLMJudgeParams(paramMap)
	case "code_analyzer":
		return validateCodeAnalyzerParams(paramMap)
	case "metrics_collector":
		return validateMetricsCollectorParams(paramMap)
	case "custom":
		// Custom units have flexible validation
		return nil
	default:
		return fmt.Errorf("unknown unit type: %s", unitType)
	}
}

// validateLLMJudgeParams validates parameters for LLM judge evaluation units,
// ensuring required prompt is provided and optional parameters meet constraints.
// validateLLMJudgeParams checks for required prompt field,
// validates temperature is between 0 and 2,
// and ensures model name is non-empty if specified.
// validateLLMJudgeParams returns an error if any validation rule fails.
func validateLLMJudgeParams(params map[string]any) error {
	// Required fields
	if _, ok := params["prompt"]; !ok {
		return fmt.Errorf("llm_judge requires 'prompt' parameter")
	}

	// Validate prompt is a string
	prompt, ok := params["prompt"].(string)
	if !ok {
		return fmt.Errorf("prompt must be a string")
	}
	if prompt == "" {
		return fmt.Errorf("prompt cannot be empty")
	}

	// Optional temperature validation
	if temp, ok := params["temperature"]; ok {
		switch v := temp.(type) {
		case float64:
			if v < 0 || v > 2 {
				return fmt.Errorf("temperature must be between 0 and 2")
			}
		case int:
			if v < 0 || v > 2 {
				return fmt.Errorf("temperature must be between 0 and 2")
			}
		default:
			return fmt.Errorf("temperature must be a number")
		}
	}

	// Optional model validation
	if model, ok := params["model"]; ok {
		modelStr, ok := model.(string)
		if !ok {
			return fmt.Errorf("model must be a string")
		}
		if modelStr == "" {
			return fmt.Errorf("model cannot be empty")
		}
	}

	return nil
}

// validateCodeAnalyzerParams validates parameters for code analyzer units,
// ensuring the specified programming language is supported
// and analysis rules are properly formatted.
// validateCodeAnalyzerParams requires a language parameter
// from the supported list and validates optional rules array
// contains only string elements.
// validateCodeAnalyzerParams returns an error if language is unsupported
// or rules are malformed.
func validateCodeAnalyzerParams(params map[string]any) error {
	// Required fields
	if _, ok := params["language"]; !ok {
		return fmt.Errorf("code_analyzer requires 'language' parameter")
	}

	// Validate language
	lang, ok := params["language"].(string)
	if !ok {
		return fmt.Errorf("language must be a string")
	}

	supportedLangs := []string{"go", "python", "javascript", "typescript", "java", "rust"}
	if !slices.Contains(supportedLangs, strings.ToLower(lang)) {
		return fmt.Errorf("unsupported language: %s", lang)
	}

	// Optional rules validation
	if rules, ok := params["rules"]; ok {
		rulesSlice, ok := rules.([]any)
		if !ok {
			return fmt.Errorf("rules must be an array")
		}
		for i, rule := range rulesSlice {
			if _, ok := rule.(string); !ok {
				return fmt.Errorf("rule at index %d must be a string", i)
			}
		}
	}

	return nil
}

// validateMetricsCollectorParams validates parameters for metrics collection
// units, ensuring at least one supported metric type is specified.
// validateMetricsCollectorParams requires a non-empty metrics array
// where each element must be a supported metric type
// (complexity, coverage, performance, security, maintainability).
// validateMetricsCollectorParams returns an error if metrics array is
// empty or contains unsupported metric types.
func validateMetricsCollectorParams(params map[string]any) error {
	// Required fields
	if _, ok := params["metrics"]; !ok {
		return fmt.Errorf("metrics_collector requires 'metrics' parameter")
	}

	// Validate metrics array
	metrics, ok := params["metrics"].([]any)
	if !ok {
		return fmt.Errorf("metrics must be an array")
	}

	if len(metrics) == 0 {
		return fmt.Errorf("metrics array cannot be empty")
	}

	supportedMetrics := []string{"complexity", "coverage", "performance", "security", "maintainability"}
	for i, metric := range metrics {
		metricStr, ok := metric.(string)
		if !ok {
			return fmt.Errorf("metric at index %d must be a string", i)
		}
		if !slices.Contains(supportedMetrics, strings.ToLower(metricStr)) {
			return fmt.Errorf("unsupported metric: %s", metricStr)
		}
	}

	return nil
}

// ValidateConditionParameters validates parameters for edge condition types,
// ensuring condition logic is properly configured for graph execution flow.
// ValidateConditionParameters supports verdict_pass, score_threshold,
// and custom condition types with type-specific parameter validation.
// ValidateConditionParameters returns an error if parameter decoding fails
// or condition-specific validation rules are violated.
func ValidateConditionParameters(condType string, params yaml.Node) error {
	var paramMap map[string]any
	if err := params.Decode(&paramMap); err != nil {
		return fmt.Errorf("failed to decode parameters: %w", err)
	}

	switch condType {
	case "verdict_pass":
		return validateVerdictPassParams(paramMap)
	case "score_threshold":
		return validateScoreThresholdParams(paramMap)
	case "custom":
		// Custom conditions have flexible validation
		return nil
	default:
		return fmt.Errorf("unknown condition type: %s", condType)
	}
}

// validateVerdictPassParams validates parameters for verdict-based edge
// conditions that control execution flow based on evaluation outcomes.
// validateVerdictPassParams accepts optional level parameter
// that must be one of: pass, fail, warning, error.
// validateVerdictPassParams returns an error if the level value is invalid.
func validateVerdictPassParams(params map[string]any) error {
	// Optional verdict level.
	if level, ok := params["level"]; ok {
		levelStr, ok := level.(string)
		if !ok {
			return fmt.Errorf("level must be a string")
		}
		validLevels := []string{"pass", "fail", "warning", "error"}
		if !slices.Contains(validLevels, strings.ToLower(levelStr)) {
			return fmt.Errorf("invalid verdict level: %s", levelStr)
		}
	}

	return nil
}

// validateScoreThresholdParams validates parameters for score-based edge
// conditions that control execution flow based on numeric thresholds.
// validateScoreThresholdParams requires a threshold parameter between 0
// and 100, and accepts optional operator (gt, gte, lt, lte, eq, ne).
// validateScoreThresholdParams returns an error if threshold is out of
// range or operator is invalid.
func validateScoreThresholdParams(params map[string]any) error {
	// Required threshold.
	threshold, ok := params["threshold"]
	if !ok {
		return fmt.Errorf("score_threshold requires 'threshold' parameter")
	}

	// Validate threshold is a number.
	switch v := threshold.(type) {
	case float64:
		if v < 0 || v > 100 {
			return fmt.Errorf("threshold must be between 0 and 100")
		}
	case int:
		if v < 0 || v > 100 {
			return fmt.Errorf("threshold must be between 0 and 100")
		}
	default:
		return fmt.Errorf("threshold must be a number")
	}

	// Optional operator.
	if op, ok := params["operator"]; ok {
		opStr, ok := op.(string)
		if !ok {
			return fmt.Errorf("operator must be a string")
		}
		validOps := []string{"gt", "gte", "lt", "lte", "eq", "ne"}
		if !slices.Contains(validOps, strings.ToLower(opStr)) {
			return fmt.Errorf("invalid operator: %s", opStr)
		}
	}

	return nil
}

// RegisterGraphValidators registers custom validation functions with
// the validator instance for use in graph configuration validation.
// RegisterGraphValidators adds unitparams and condparams validators
// that can be referenced in struct tags for automated validation.
// RegisterGraphValidators returns an error if any validator registration
// fails.
func RegisterGraphValidators(v *validator.Validate) error {
	// Register unit parameter validator.
	if err := v.RegisterValidation("unitparams", validateUnitParametersTag); err != nil {
		return fmt.Errorf("failed to register unitparams validator: %w", err)
	}

	// Register condition parameter validator.
	if err := v.RegisterValidation("condparams", validateConditionParametersTag); err != nil {
		return fmt.Errorf("failed to register condparams validator: %w", err)
	}

	return nil
}

// validateUnitParametersTag is a validator.Func that can be used in struct
// tags to validate unit parameters, though actual validation is performed
// by ValidateUnitParameters during semantic validation.
// validateUnitParametersTag currently returns true as validation occurs
// elsewhere in the processing pipeline.
func validateUnitParametersTag(fl validator.FieldLevel) bool {
	// This would be used as a struct tag validator
	// For now, return true as the actual validation happens elsewhere
	return true
}

// validateConditionParametersTag is a validator.Func that can be used in
// struct tags to validate condition parameters, though actual validation
// is performed by ValidateConditionParameters during semantic validation.
// validateConditionParametersTag currently returns true as validation
// occurs elsewhere in the processing pipeline.
func validateConditionParametersTag(fl validator.FieldLevel) bool {
	// This would be used as a struct tag validator
	// For now, return true as the actual validation happens elsewhere
	return true
}
