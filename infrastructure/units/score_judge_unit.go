package units

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"text/template"

	"github.com/go-playground/validator/v10"
	"golang.org/x/sync/errgroup"
	"gopkg.in/yaml.v3"

	"github.com/ahrav/go-gavel/internal/domain"
	"github.com/ahrav/go-gavel/internal/ports"
)

var _ ports.Unit = (*ScoreJudgeUnit)(nil)

// Configuration constants for ScoreJudgeUnit
const (
	// Score scale validation bounds
	MinScoreValue = -1000.0 // Minimum allowed score value
	MaxScoreValue = 1000.0  // Maximum allowed score value
	MinScoreRange = 0.01    // Minimum allowed range between min and max scores

	// Default configuration values
	DefaultJudgeMaxConcurrency = 5   // Default number of concurrent LLM calls for scoring
	DefaultJudgeMaxTokens      = 256 // Default maximum tokens for judge reasoning
	DefaultJudgeTemperature    = 0.0 // Default temperature for consistent scoring
)

// ScoreJudgeUnit evaluates and scores individual candidate answers using an LLM client.
// It expects answers in state and produces JudgeSummary objects with scores.
// The unit scores each answer independently against the question and scoring criteria.
// The unit is stateless and thread-safe for concurrent execution.
type ScoreJudgeUnit struct {
	// name is the unique identifier for this unit instance.
	name string
	// config contains the validated configuration parameters.
	config ScoreJudgeConfig
	// llmClient provides access to the LLM for scoring evaluation.
	llmClient ports.LLMClient
	// validator ensures configuration parameter validation.
	validator *validator.Validate
	// promptTemplate is the compiled template for safe prompt generation.
	promptTemplate *template.Template
}

// ScoreJudgeConfig defines the configuration parameters for the ScoreJudgeUnit.
// All fields are validated during unit creation and parameter unmarshaling.
type ScoreJudgeConfig struct {
	// JudgePrompt is the Go template used to score answers.
	// Should use {{.Question}} and {{.Answer}} placeholders for safe substitution.
	// Example: "Rate this answer to '{{.Question}}': {{.Answer}}"
	JudgePrompt string `yaml:"judge_prompt" json:"judge_prompt" validate:"required,min=20"`

	// ScoreScale defines the scoring range (e.g., "1-10" or "0.0-1.0").
	// Used to normalize scores and validate LLM responses.
	ScoreScale string `yaml:"score_scale" json:"score_scale" validate:"required"`

	// Temperature controls randomness in LLM scoring (0.0-1.0).
	// Lower values produce more consistent scoring.
	Temperature float64 `yaml:"temperature" json:"temperature" validate:"min=0.0,max=1.0"`

	// MaxTokens limits the length of scoring reasoning.
	// Should allow enough space for detailed explanations.
	MaxTokens int `yaml:"max_tokens" json:"max_tokens" validate:"required,min=50,max=2000"`

	// MinConfidence sets the minimum acceptable confidence score.
	// Responses below this threshold may be rejected or flagged.
	MinConfidence float64 `yaml:"min_confidence" json:"min_confidence" validate:"min=0.0,max=1.0"`

	// MaxConcurrency limits the number of concurrent LLM calls.
	// Prevents overwhelming the LLM service with too many simultaneous requests.
	// Defaults to 5 if not specified.
	MaxConcurrency int `yaml:"max_concurrency" json:"max_concurrency" validate:"min=1,max=20"`
}

// ScoreScale represents a parsed and validated scoring range.
// This value object eliminates repeated parsing and provides type safety.
type ScoreScale struct {
	Min float64
	Max float64
}

// ParseScoreScale parses a score scale string into a ScoreScale value object.
// Supports formats like "1-10", "0.0-1.0", "1-5", "-5-10", etc.
// Validates that scale values are within reasonable bounds.
func ParseScoreScale(scaleStr string) (ScoreScale, error) {
	// Split by dash and analyze the parts to detect valid vs invalid formats
	parts := strings.Split(scaleStr, "-")

	// Handle different cases based on number of parts
	var minPart, maxPart string

	switch len(parts) {
	case 2:
		// Simple case: "1-10" or invalid like "1to10"
		if parts[0] == "" {
			// String like "-5" (missing max part)
			return ScoreScale{}, fmt.Errorf("score scale must be in format 'min-max', got: %s", scaleStr)
		}
		minPart = parts[0]
		maxPart = parts[1]

	case 3:
		// Could be "-5-10" (negative min) or "1--3" (negative max) or invalid "1-2-3"
		if parts[0] == "" {
			// Starts with dash: "-5-10"
			if parts[1] == "" {
				// Two consecutive dashes at start: "--5-10" - invalid
				return ScoreScale{}, fmt.Errorf("score scale must be in format 'min-max', got: %s", scaleStr)
			}
			minPart = "-" + parts[1]
			maxPart = parts[2]
		} else if parts[1] == "" {
			// Middle part empty: "5--3" (negative max)
			minPart = parts[0]
			maxPart = "-" + parts[2]
		} else {
			// All parts non-empty: "1-2-3" - invalid format
			return ScoreScale{}, fmt.Errorf("score scale must be in format 'min-max', got: %s", scaleStr)
		}

	case 4:
		// Could be "-5--3" (both negative)
		if parts[0] == "" && parts[2] == "" {
			minPart = "-" + parts[1]
			maxPart = "-" + parts[3]
		} else {
			return ScoreScale{}, fmt.Errorf("score scale must be in format 'min-max', got: %s", scaleStr)
		}

	default:
		// No dashes, too many dashes, or other invalid formats
		return ScoreScale{}, fmt.Errorf("score scale must be in format 'min-max', got: %s", scaleStr)
	}

	minVal, err := strconv.ParseFloat(strings.TrimSpace(minPart), 64)
	if err != nil {
		return ScoreScale{}, fmt.Errorf("invalid minimum value in score scale: %w", err)
	}

	maxVal, err := strconv.ParseFloat(strings.TrimSpace(maxPart), 64)
	if err != nil {
		return ScoreScale{}, fmt.Errorf("invalid maximum value in score scale: %w", err)
	}

	// Validate reasonable bounds for score scale
	if minVal < MinScoreValue {
		return ScoreScale{}, fmt.Errorf("minimum score value %.2f is too low (must be >= %.0f)", minVal, MinScoreValue)
	}
	if maxVal > MaxScoreValue {
		return ScoreScale{}, fmt.Errorf("maximum score value %.2f is too high (must be <= %.0f)", maxVal, MaxScoreValue)
	}
	if minVal >= maxVal {
		return ScoreScale{}, fmt.Errorf("minimum value must be less than maximum value in score scale")
	}
	if maxVal-minVal < MinScoreRange {
		return ScoreScale{}, fmt.Errorf("score scale range too narrow (must be at least %.2f)", MinScoreRange)
	}

	return ScoreScale{Min: minVal, Max: maxVal}, nil
}

// Contains checks if a score falls within this scale's range.
func (s ScoreScale) Contains(score float64) bool {
	return score >= s.Min && score <= s.Max
}

// String returns the string representation of the scale.
func (s ScoreScale) String() string {
	return fmt.Sprintf("%.1f-%.1f", s.Min, s.Max)
}

// LLMJudgeResponse represents the expected JSON structure from the LLM
// when scoring answers. This ensures reliable parsing and validation.
type LLMJudgeResponse struct {
	// Score is the numerical score for the answer within the configured range.
	Score float64 `json:"score" validate:"required"`

	// Confidence represents how confident the LLM is in its scoring (0.0-1.0).
	Confidence float64 `json:"confidence" validate:"required,min=0.0,max=1.0"`

	// Reasoning provides the detailed explanation for the score.
	Reasoning string `json:"reasoning" validate:"required,min=10"`

	// Version allows for future schema evolution.
	Version int `json:"version,omitempty"`
}

// defaultScoreJudgeConfig returns a ScoreJudgeConfig with sensible defaults.
// This ensures consistent behavior when configuration values are missing.
func defaultScoreJudgeConfig() ScoreJudgeConfig {
	return ScoreJudgeConfig{
		JudgePrompt:    "Please score the following answer to the question on a scale from 1 to 10:\n\nQuestion: {{.Question}}\nAnswer: {{.Answer}}\n\nConsider accuracy, completeness, and clarity in your scoring.",
		ScoreScale:     "1-10",
		Temperature:    DefaultJudgeTemperature,
		MaxTokens:      DefaultJudgeMaxTokens,
		MinConfidence:  0.0,
		MaxConcurrency: DefaultJudgeMaxConcurrency,
	}
}

// validateConfig validates a ScoreJudgeConfig using the provided validator.
// This centralizes validation logic to avoid duplication across multiple methods.
func validateConfig(v *validator.Validate, config ScoreJudgeConfig) error {
	if err := v.Struct(config); err != nil {
		return fmt.Errorf("configuration validation failed: %w", err)
	}

	// Validate score scale format using the value object
	_, err := ParseScoreScale(config.ScoreScale)
	if err != nil {
		return fmt.Errorf("invalid score scale: %w", err)
	}

	return nil
}

// NewScoreJudgeUnit creates a new ScoreJudgeUnit with the specified configuration
// and dependencies.
// The unit validates its configuration and ensures the LLM client is available.
// Returns an error if configuration validation fails or dependencies are missing.
func NewScoreJudgeUnit(name string, llmClient ports.LLMClient, config ScoreJudgeConfig) (*ScoreJudgeUnit, error) {
	if name == "" {
		return nil, fmt.Errorf("unit name cannot be empty")
	}
	if llmClient == nil {
		return nil, fmt.Errorf("LLM client cannot be nil")
	}

	v := validator.New()
	if err := validateConfig(v, config); err != nil {
		return nil, err
	}

	// Compile the prompt template to prevent injection attacks.
	tmpl, err := template.New("judgePrompt").Parse(config.JudgePrompt)
	if err != nil {
		return nil, fmt.Errorf("failed to parse judge prompt template: %w", err)
	}

	return &ScoreJudgeUnit{
		name:           name,
		config:         config,
		llmClient:      llmClient,
		validator:      v,
		promptTemplate: tmpl,
	}, nil
}

// Name returns the unique identifier for this unit instance.
// The name is used for logging, debugging, and graph node referencing.
func (sju *ScoreJudgeUnit) Name() string { return sju.name }

// Execute scores candidate answers by calling the LLM client for individual evaluation.
// It retrieves answers using KeyAnswers, scores each answer independently against the question,
// and stores JudgeSummary objects in state using KeyJudgeScores.
// Returns updated state with scoring results or an error if scoring fails.
func (sju *ScoreJudgeUnit) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	question, ok := state.GetString(domain.KeyQuestion)
	if !ok {
		return state, fmt.Errorf("unit %s: question not found in state with key %s", sju.name, domain.KeyQuestion)
	}

	answersRaw, ok := state.Get(domain.KeyAnswers)
	if !ok {
		return state, fmt.Errorf("unit %s: answers not found in state with key %s", sju.name, domain.KeyAnswers)
	}

	answers, ok := answersRaw.([]domain.Answer)
	if !ok {
		return state, fmt.Errorf("unit %s: answers in state are not of type []domain.Answer, got %T", sju.name, answersRaw)
	}

	if len(answers) == 0 {
		return state, fmt.Errorf("unit %s: no answers to score", sju.name)
	}

	// Score each answer concurrently for better performance.
	judgeSummaries := make([]domain.JudgeSummary, len(answers))
	var mu sync.Mutex // Protect judgeSummaries slice

	g, gctx := errgroup.WithContext(ctx)

	// Limit concurrent LLM calls to avoid overwhelming the service.
	// Use configured max concurrency or default if not set.
	maxConcurrency := sju.config.MaxConcurrency
	if maxConcurrency <= 0 {
		maxConcurrency = DefaultJudgeMaxConcurrency // Fallback to reasonable default
	}
	g.SetLimit(maxConcurrency)

	for i, answer := range answers {
		answerContent := answer.Content

		g.Go(func() error {
			// Create scoring prompt with question and answer using template for safe generation.
			var promptBuf bytes.Buffer
			templateData := struct {
				Question string
				Answer   string
			}{
				Question: question,
				Answer:   answerContent,
			}
			if err := sju.promptTemplate.Execute(&promptBuf, templateData); err != nil {
				return fmt.Errorf("unit %s: failed to execute prompt template for answer %d: %w",
					sju.name, i+1, err)
			}
			basePrompt := promptBuf.String()
			prompt := basePrompt + "\n\nIMPORTANT: You must respond with valid JSON in exactly this format:\n" +
				`{"score": <number>, "confidence": <0.0-1.0>, "reasoning": "<detailed explanation>", "version": 1}`

			// Prepare LLM options with JSON response format if supported.
			options := map[string]any{
				"temperature": sju.config.Temperature,
				"max_tokens":  sju.config.MaxTokens,
			}

			// Request JSON output format if the provider supports it
			if supportsJSONMode(sju.llmClient) {
				options["response_format"] = map[string]string{"type": "json_object"}
			}

			// Call LLM to score the answer.
			response, err := sju.llmClient.Complete(gctx, prompt, options)
			if err != nil {
				return fmt.Errorf("unit %s: LLM call failed for answer %d (content length: %d chars): %w",
					sju.name, i+1, len(answerContent), err)
			}

			// Parse the LLM response to extract score, reasoning, and confidence.
			summary, err := sju.parseLLMResponse(response, fmt.Sprintf("%s_judge_%d", sju.name, i+1))
			if err != nil {
				return fmt.Errorf("unit %s: failed to parse LLM response for answer %d (response length: %d chars): %w",
					sju.name, i+1, len(response), err)
			}

			// Validate minimum confidence requirement.
			if summary.Confidence < sju.config.MinConfidence {
				return fmt.Errorf("unit %s: answer %d confidence %.3f below minimum %.3f (score: %.3f, reasoning length: %d)",
					sju.name, i+1, summary.Confidence, sju.config.MinConfidence, summary.Score, len(summary.Reasoning))
			}

			// Store the result in the correct position (thread-safe)
			mu.Lock()
			judgeSummaries[i] = summary
			mu.Unlock()

			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return state, err
	}

	return state.With(domain.KeyJudgeScores, judgeSummaries), nil
}

// Validate checks if the unit is properly configured and ready for execution.
// It validates the configuration parameters and verifies LLM client availability.
// Returns nil if validation passes, or an error describing what is invalid.
func (sju *ScoreJudgeUnit) Validate() error {
	if sju.llmClient == nil {
		return fmt.Errorf("unit %s: LLM client is not configured", sju.name)
	}

	// Use centralized validation logic
	if err := validateConfig(sju.validator, sju.config); err != nil {
		return fmt.Errorf("unit %s: %w", sju.name, err)
	}

	// Verify LLM client is functional by checking model.
	model := sju.llmClient.GetModel()
	if model == "" {
		return fmt.Errorf("unit %s: LLM client model is not configured", sju.name)
	}

	return nil
}

// supportsJSONMode checks if the LLM client supports structured JSON output.
// This is a simple heuristic - in production, you might check the client type
// or have the client expose this capability directly.
func supportsJSONMode(client ports.LLMClient) bool {
	// This is a simplified check. In practice, you'd check the model type
	// or have the client interface expose this capability.
	model := client.GetModel()
	return strings.Contains(strings.ToLower(model), "gpt") ||
		strings.Contains(strings.ToLower(model), "claude")
}

// parseLLMResponse extracts score, reasoning, and confidence from LLM JSON response.
// Expects structured JSON output with score, confidence, reasoning, and version fields.
func (sju *ScoreJudgeUnit) parseLLMResponse(
	response string,
	judgeID string,
) (domain.JudgeSummary, error) {
	jsonStr := extractJSON(response)
	if jsonStr == "" {
		return domain.JudgeSummary{}, fmt.Errorf("judge %s: no valid JSON found in LLM response (response length: %d chars)",
			judgeID, len(response))
	}

	var llmResponse LLMJudgeResponse
	if err := json.Unmarshal([]byte(jsonStr), &llmResponse); err != nil {
		return domain.JudgeSummary{}, fmt.Errorf("judge %s: failed to parse JSON response (JSON length: %d chars): %w",
			judgeID, len(jsonStr), err)
	}

	if err := sju.validator.Struct(llmResponse); err != nil {
		return domain.JudgeSummary{}, fmt.Errorf("judge %s: invalid response structure (score: %.3f, confidence: %.3f): %w",
			judgeID, llmResponse.Score, llmResponse.Confidence, err)
	}

	if err := sju.validateScoreInRange(llmResponse.Score); err != nil {
		return domain.JudgeSummary{}, fmt.Errorf("judge %s: score out of range (scale: %s): %w",
			judgeID, sju.config.ScoreScale, err)
	}

	return domain.JudgeSummary{
		Reasoning:  llmResponse.Reasoning,
		Confidence: llmResponse.Confidence,
		Score:      llmResponse.Score,
	}, nil
}

// extractJSON attempts to extract JSON from a response that might contain
// additional text before or after the JSON object.
// It handles various response formats including markdown code blocks and
// text surrounding the JSON object.
func extractJSON(response string) string {
	response = strings.TrimSpace(response)

	// First, try to extract from markdown code blocks
	if strings.Contains(response, "```json") {
		start := strings.Index(response, "```json")
		if start != -1 {
			start += 7 // Move past "```json"
			end := strings.Index(response[start:], "```")
			if end != -1 {
				return strings.TrimSpace(response[start : start+end])
			}
		}
	}

	// Also check for generic code blocks
	if strings.Contains(response, "```") {
		start := strings.Index(response, "```")
		if start != -1 {
			start += 3 // Move past "```"
			// Skip any language identifier
			newlineIdx := strings.Index(response[start:], "\n")
			if newlineIdx != -1 {
				start += newlineIdx + 1
			}
			end := strings.Index(response[start:], "```")
			if end != -1 {
				candidate := strings.TrimSpace(response[start : start+end])
				// Check if it looks like JSON
				if strings.HasPrefix(candidate, "{") {
					return candidate
				}
			}
		}
	}

	// Look for JSON object boundaries
	start := strings.Index(response, "{")
	if start == -1 {
		return ""
	}

	// Find the matching closing brace, handling nested objects and strings
	braceCount := 0
	inString := false
	escapeNext := false

	for i := start; i < len(response); i++ {
		char := response[i]

		// Handle escape sequences
		if escapeNext {
			escapeNext = false
			continue
		}

		if char == '\\' {
			escapeNext = true
			continue
		}

		// Track string boundaries
		if char == '"' && !escapeNext {
			inString = !inString
			continue
		}

		// Only count braces outside of strings
		if !inString {
			switch char {
			case '{':
				braceCount++
			case '}':
				braceCount--
				if braceCount == 0 {
					return response[start : i+1]
				}
			}
		}
	}

	return ""
}

// validateScoreInRange checks if the score falls within the configured scale.
func (sju *ScoreJudgeUnit) validateScoreInRange(score float64) error {
	scale, err := ParseScoreScale(sju.config.ScoreScale)
	if err != nil {
		return fmt.Errorf("invalid score scale: %w", err)
	}

	if !scale.Contains(score) {
		return fmt.Errorf("score %.2f not in range [%.2f, %.2f]", score, scale.Min, scale.Max)
	}

	return nil
}

// UnmarshalParameters deserializes YAML configuration parameters and returns
// a new ScoreJudgeUnit instance with the updated configuration.
// This method maintains thread-safety by not mutating the existing unit.
// Returns a new unit instance or an error if YAML parsing fails or validation fails.
func (sju *ScoreJudgeUnit) UnmarshalParameters(params yaml.Node) (*ScoreJudgeUnit, error) {
	var config ScoreJudgeConfig

	// Use strict decoding to catch unknown fields.
	if err := params.Decode(&config); err != nil {
		return nil, fmt.Errorf("failed to decode parameters: %w", err)
	}

	// Validate the decoded configuration using centralized logic.
	if err := validateConfig(sju.validator, config); err != nil {
		return nil, err
	}

	// Compile the prompt template to prevent injection attacks.
	tmpl, err := template.New("judgePrompt").Parse(config.JudgePrompt)
	if err != nil {
		return nil, fmt.Errorf("failed to parse judge prompt template: %w", err)
	}

	// Return a new instance with the updated configuration to maintain thread safety.
	return &ScoreJudgeUnit{
		name:           sju.name,
		config:         config,
		llmClient:      sju.llmClient,
		validator:      sju.validator,
		promptTemplate: tmpl,
	}, nil
}

// CreateScoreJudgeUnit is a factory function that creates a ScoreJudgeUnit
// from a configuration map, following the UnitFactory pattern.
// This function is used by the UnitRegistry for dynamic unit creation.
func CreateScoreJudgeUnit(id string, config map[string]any) (*ScoreJudgeUnit, error) {
	// Extract LLM client from config.
	llmClient, ok := config["llm_client"].(ports.LLMClient)
	if !ok {
		return nil, fmt.Errorf("llm_client is required and must implement ports.LLMClient")
	}

	// Start with sensible defaults and merge user-provided values.
	judgeConfig := defaultScoreJudgeConfig()

	if judgePrompt, ok := config["judge_prompt"].(string); ok {
		judgeConfig.JudgePrompt = judgePrompt
	}

	if scoreScale, ok := config["score_scale"].(string); ok {
		judgeConfig.ScoreScale = scoreScale
	}

	if temperature, ok := config["temperature"]; ok {
		if val, ok := temperature.(float64); ok {
			judgeConfig.Temperature = val
		} else if val, ok := temperature.(int); ok {
			// Handle integer temperature values (e.g., 0, 1)
			judgeConfig.Temperature = float64(val)
		} else if val, ok := temperature.(string); ok {
			if parsed, err := strconv.ParseFloat(val, 64); err == nil {
				judgeConfig.Temperature = parsed
			}
		}
	}

	if maxTokens, ok := config["max_tokens"]; ok {
		if val, ok := maxTokens.(int); ok {
			judgeConfig.MaxTokens = val
		} else if val, ok := maxTokens.(float64); ok {
			// YAML numbers often come as float64, convert to int
			judgeConfig.MaxTokens = int(val)
		} else if val, ok := maxTokens.(string); ok {
			if parsed, err := strconv.Atoi(val); err == nil {
				judgeConfig.MaxTokens = parsed
			}
		}
	}

	if minConfidence, ok := config["min_confidence"]; ok {
		if val, ok := minConfidence.(float64); ok {
			judgeConfig.MinConfidence = val
		} else if val, ok := minConfidence.(int); ok {
			// Handle integer confidence values (e.g., 0, 1)
			judgeConfig.MinConfidence = float64(val)
		} else if val, ok := minConfidence.(string); ok {
			if parsed, err := strconv.ParseFloat(val, 64); err == nil {
				judgeConfig.MinConfidence = parsed
			}
		}
	}

	if maxConcurrency, ok := config["max_concurrency"]; ok {
		if val, ok := maxConcurrency.(int); ok {
			judgeConfig.MaxConcurrency = val
		} else if val, ok := maxConcurrency.(float64); ok {
			// YAML numbers often come as float64, convert to int
			judgeConfig.MaxConcurrency = int(val)
		} else if val, ok := maxConcurrency.(string); ok {
			if parsed, err := strconv.Atoi(val); err == nil {
				judgeConfig.MaxConcurrency = parsed
			}
		}
	}

	return NewScoreJudgeUnit(id, llmClient, judgeConfig)
}
