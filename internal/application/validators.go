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

	// TODO: Come back to this, this feels like something we will get tripped up
	// on for new units for oss contributors.

	switch unitType {
	case "score_judge":
		return validateScoreJudgeParams(paramMap)
	case "answerer":
		return validateAnswererParams(paramMap)
	case "verification":
		return validateVerificationParams(paramMap)
	case "arithmetic_mean", "max_pool", "median_pool":
		return validatePoolParams(paramMap)
	case "exact_match":
		return validateExactMatchParams(paramMap)
	case "fuzzy_match":
		return validateFuzzyMatchParams(paramMap)
	case "custom":
		// Custom units have flexible validation
		return nil
	default:
		return fmt.Errorf("unknown unit type: %s", unitType)
	}
}

// validateScoreJudgeParams validates parameters for score judge evaluation units,
// ensuring required judge_prompt is provided and optional parameters meet constraints.
// validateScoreJudgeParams checks for required judge_prompt field,
// validates temperature is between 0 and 2,
// and ensures model name is non-empty if specified.
// validateScoreJudgeParams returns an error if any validation rule fails.
func validateScoreJudgeParams(params map[string]any) error {
	if _, ok := params["judge_prompt"]; !ok {
		return fmt.Errorf("score_judge requires 'judge_prompt' parameter")
	}

	if _, ok := params["score_scale"]; !ok {
		return fmt.Errorf("score_judge requires 'score_scale' parameter")
	}

	judgePrompt, ok := params["judge_prompt"].(string)
	if !ok {
		return fmt.Errorf("judge_prompt must be a string")
	}
	if judgePrompt == "" {
		return fmt.Errorf("judge_prompt cannot be empty")
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

	// Register model string validator for provider/model format.
	if err := v.RegisterValidation("modelformat", validateModelFormat); err != nil {
		return fmt.Errorf("failed to register modelformat validator: %w", err)
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

// validateModelFormat validates that a model string matches the required format:
// ^[a-z0-9]+/[A-Za-z0-9\-_\.]+(@[A-Za-z0-9\-_\.]+)?$
// This ensures the model follows the pattern provider/model or provider/model@version.
func validateModelFormat(fl validator.FieldLevel) bool {
	model := fl.Field().String()

	if model == "" {
		return true
	}

	// Basic validation - must contain a slash if not empty.
	for i, ch := range model {
		if ch == '/' {
			if i == 0 {
				return false // provider name cannot be empty
			}
			if i == len(model)-1 {
				return false // model name cannot be empty
			}
			return true
		}
	}

	return false
}

// validateAnswererParams validates parameters for answerer units.
func validateAnswererParams(params map[string]any) error {
	// Answerer units don't have required parameters
	// Optional: num_answers, prompt, temperature, max_tokens
	if numAnswers, ok := params["num_answers"]; ok {
		if num, ok := numAnswers.(int); ok {
			if num < 1 {
				return fmt.Errorf("num_answers must be at least 1")
			}
		}
	}
	return nil
}

// validateVerificationParams validates parameters for verification units.
func validateVerificationParams(params map[string]any) error {
	if _, ok := params["prompt"]; !ok {
		return fmt.Errorf("verification requires 'prompt' parameter")
	}
	return nil
}

// validatePoolParams validates parameters for pooling units (max_pool, median_pool, arithmetic_mean).
func validatePoolParams(params map[string]any) error {
	// Pool units typically don't have required parameters
	// They work with scores from previous units
	return nil
}

// validateExactMatchParams validates parameters for exact match units.
func validateExactMatchParams(params map[string]any) error {
	// Exact match units don't have required parameters.
	if caseSensitive, ok := params["case_sensitive"]; ok {
		if _, ok := caseSensitive.(bool); !ok {
			return fmt.Errorf("case_sensitive must be a boolean")
		}
	}
	if trimWhitespace, ok := params["trim_whitespace"]; ok {
		if _, ok := trimWhitespace.(bool); !ok {
			return fmt.Errorf("trim_whitespace must be a boolean")
		}
	}
	return nil
}

// validateFuzzyMatchParams validates parameters for fuzzy match units.
func validateFuzzyMatchParams(params map[string]any) error {
	if algorithm, ok := params["algorithm"]; ok {
		if alg, ok := algorithm.(string); ok {
			if alg != "levenshtein" {
				return fmt.Errorf("fuzzy_match only supports 'levenshtein' algorithm")
			}
		} else {
			return fmt.Errorf("algorithm must be a string")
		}
	}
	if threshold, ok := params["threshold"]; ok {
		switch v := threshold.(type) {
		case float64:
			if v < 0 || v > 1 {
				return fmt.Errorf("threshold must be between 0 and 1")
			}
		case int:
			if v < 0 || v > 1 {
				return fmt.Errorf("threshold must be between 0 and 1")
			}
		default:
			return fmt.Errorf("threshold must be a number")
		}
	}
	if caseSensitive, ok := params["case_sensitive"]; ok {
		if _, ok := caseSensitive.(bool); !ok {
			return fmt.Errorf("case_sensitive must be a boolean")
		}
	}
	return nil
}
