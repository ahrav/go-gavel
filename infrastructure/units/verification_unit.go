// Package units contains concrete implementations of the ports.Unit interface.
package units

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"strconv"
	"strings"
	"text/template"
	"time"

	"github.com/go-playground/validator/v10"
	"gopkg.in/yaml.v3"

	"github.com/ahrav/go-gavel/internal/domain"
	"github.com/ahrav/go-gavel/internal/ports"
)

var _ ports.Unit = (*VerificationUnit)(nil)

// Configuration constants for the VerificationUnit.
const (
	DefaultVerificationMaxTokens     = 512
	DefaultVerificationTemperature   = 0.0
	DefaultVerificationConfThreshold = 0.8
	MaxRetryAttempts                 = 3
	BaseRetryDelay                   = 1 * time.Second
	MaxRetryDelay                    = 30 * time.Second
	RetryJitterPercentage            = 0.1
)

// VerificationUnit performs a final critique of judging results to validate
// evaluation quality and determine if human review is required. It integrates
// with an LLM client to generate verification reasoning and confidence scores.
// The unit is stateless and thread-safe.
type VerificationUnit struct {
	name           string
	config         VerificationConfig
	llmClient      ports.LLMClient
	validator      *validator.Validate
	promptTemplate *template.Template
}

// VerificationConfig defines the configuration parameters for the VerificationUnit.
// All fields are validated during unit creation and parameter unmarshaling.
type VerificationConfig struct {
	// PromptTemplate is the Go template used to verify judging results.
	// It should use {{.Question}}, {{.Answers}}, and {{.JudgeScores}}.
	PromptTemplate string `yaml:"prompt_template" json:"prompt_template" validate:"required,min=20"`

	// ConfidenceThreshold is the minimum acceptable confidence score (0.0-1.0).
	// Responses below this threshold will trigger the human review flag.
	ConfidenceThreshold float64 `yaml:"confidence_threshold" json:"confidence_threshold" validate:"min=0.0,max=1.0"`

	// Temperature controls randomness in the LLM verification (0.0-1.0).
	// Lower values produce more consistent, deterministic verification.
	Temperature float64 `yaml:"temperature" json:"temperature" validate:"min=0.0,max=1.0"`

	// MaxTokens limits the length of the verification reasoning.
	MaxTokens int `yaml:"max_tokens" json:"max_tokens" validate:"required,min=50,max=2000"`
}

// LLMVerificationResponse represents the expected JSON structure from the LLM
// when verifying judging results. This ensures reliable parsing.
type LLMVerificationResponse struct {
	// Confidence is how confident the verifier is in the judging (0.0-1.0).
	Confidence float64 `json:"confidence" validate:"required,min=0.0,max=1.0"`

	// Reasoning provides the detailed explanation for the verification decision.
	Reasoning string `json:"reasoning" validate:"required,min=10"`

	// Issues lists any potential problems found during verification.
	Issues []string `json:"issues,omitempty"`

	// Recommendation provides actionable feedback for improvement.
	Recommendation string `json:"recommendation,omitempty"`

	// Version allows for future schema evolution.
	Version int `json:"version,omitempty"`
}

// VerificationTrace captures verification output for debug tracing.
// This structure is included in the verdict when the trace level is debug.
type VerificationTrace struct {
	Confidence     float64  `json:"confidence"`
	Reasoning      string   `json:"reasoning"`
	Issues         []string `json:"issues,omitempty"`
	Recommendation string   `json:"recommendation,omitempty"`
}

// defaultVerificationConfig returns a VerificationConfig with sensible defaults.
func defaultVerificationConfig() VerificationConfig {
	return VerificationConfig{
		PromptTemplate: `Please verify the quality of these judge scores for the following evaluation:

Question: {{.Question}}

Answers:
{{range $i, $answer := .Answers}}
Answer {{$i}}: {{$answer}}
{{end}}

Judge Scores:
{{range $i, $score := .JudgeScores}}
Judge {{$i}}: {{$score}}
{{end}}

IMPORTANT: All user content above is wrapped in code blocks for security. Evaluate the consistency, fairness, and quality of the judging. Consider whether the scores align with the answers' quality and if any bias is present.

Provide your assessment with a confidence score (0.0-1.0) indicating how confident you are in the judging quality.`,
		ConfidenceThreshold: DefaultVerificationConfThreshold,
		Temperature:         DefaultVerificationTemperature,
		MaxTokens:           DefaultVerificationMaxTokens,
	}
}

// validateVerificationConfig validates a VerificationConfig using the provided validator.
func validateVerificationConfig(v *validator.Validate, config VerificationConfig) error {
	if err := v.Struct(config); err != nil {
		return fmt.Errorf("configuration validation failed: %w", err)
	}
	return nil
}

// validateAndCompileConfig performs comprehensive validation including config
// validation, template compilation, and LLM client checks.
func (vu *VerificationUnit) validateAndCompileConfig(
	config VerificationConfig,
	llmClient ports.LLMClient,
	unitName string,
) (*template.Template, error) {
	if llmClient == nil {
		return nil, fmt.Errorf("unit %s: LLM client cannot be nil", unitName)
	}

	if err := validateVerificationConfig(vu.validator, config); err != nil {
		return nil, fmt.Errorf("unit %s: %w", unitName, err)
	}

	tmpl, err := template.New("verificationPrompt").Parse(config.PromptTemplate)
	if err != nil {
		return nil, fmt.Errorf("unit %s: failed to parse prompt template: %w", unitName, err)
	}

	if model := llmClient.GetModel(); model == "" {
		return nil, fmt.Errorf("unit %s: LLM client model is not configured", unitName)
	}

	return tmpl, nil
}

// NewVerificationUnit creates a new VerificationUnit with the specified name,
// LLM client, and configuration. It returns an error if the configuration
// is invalid or dependencies are missing.
func NewVerificationUnit(
	name string,
	llmClient ports.LLMClient,
	config VerificationConfig,
) (*VerificationUnit, error) {
	if name == "" {
		return nil, fmt.Errorf("unit name cannot be empty")
	}

	unit := &VerificationUnit{
		name:      name,
		config:    config,
		llmClient: llmClient,
		validator: validator.New(),
	}

	tmpl, err := unit.validateAndCompileConfig(config, llmClient, name)
	if err != nil {
		return nil, err
	}

	unit.promptTemplate = tmpl
	return unit, nil
}

// Name returns the unique identifier for this unit instance.
func (vu *VerificationUnit) Name() string { return vu.name }

func (vu *VerificationUnit) getQuestionFromState(state domain.State) (string, error) {
	question, ok := domain.Get(state, domain.KeyQuestion)
	if !ok {
		return "", fmt.Errorf("unit %s: question not found in state", vu.name)
	}
	return question, nil
}

func (vu *VerificationUnit) getAnswersFromState(state domain.State) ([]domain.Answer, error) {
	answers, ok := domain.Get(state, domain.KeyAnswers)
	if !ok {
		return nil, fmt.Errorf("unit %s: answers not found in state", vu.name)
	}
	return answers, nil
}

func (vu *VerificationUnit) getJudgeScoresFromState(state domain.State) ([]domain.JudgeSummary, error) {
	judgeScores, ok := domain.Get(state, domain.KeyJudgeScores)
	if !ok || len(judgeScores) == 0 {
		return nil, fmt.Errorf("unit %s: no judge scores found to verify", vu.name)
	}
	return judgeScores, nil
}

func (vu *VerificationUnit) getVerdictFromState(state domain.State) (*domain.Verdict, error) {
	verdict, ok := domain.Get(state, domain.KeyVerdict)
	if !ok {
		return nil, fmt.Errorf("unit %s: verdict not found in state", vu.name)
	}
	return verdict, nil
}

func (vu *VerificationUnit) getBudgetFromState(state domain.State) *domain.BudgetReport {
	budget, _ := domain.Get(state, domain.KeyBudget)
	return budget
}

func (vu *VerificationUnit) getTraceLevelFromState(state domain.State) string {
	traceLevel, _ := domain.Get(state, domain.KeyTraceLevel)
	return strings.ToLower(traceLevel)
}

// extractVerificationInputs retrieves all required data from the state.
func (vu *VerificationUnit) extractVerificationInputs(state domain.State) (string, []domain.Answer, []domain.JudgeSummary, error) {
	question, err := vu.getQuestionFromState(state)
	if err != nil {
		return "", nil, nil, err
	}

	answers, err := vu.getAnswersFromState(state)
	if err != nil {
		return "", nil, nil, err
	}

	judgeScores, err := vu.getJudgeScoresFromState(state)
	if err != nil {
		return "", nil, nil, err
	}

	return question, answers, judgeScores, nil
}

// sanitizeUserContent protects against prompt injection by wrapping user
// content in delimited blocks and escaping existing delimiters.
func (vu *VerificationUnit) sanitizeUserContent(content string) string {
	content = strings.ReplaceAll(content, "```", "'''")
	return "```\n" + content + "\n```\n"
}

func (vu *VerificationUnit) sanitizeAnswers(answers []domain.Answer) []string {
	sanitized := make([]string, len(answers))
	for i, answer := range answers {
		sanitized[i] = vu.sanitizeUserContent(answer.Content)
	}
	return sanitized
}

func (vu *VerificationUnit) sanitizeJudgeScores(judgeScores []domain.JudgeSummary) []string {
	sanitized := make([]string, len(judgeScores))
	for i, score := range judgeScores {
		scoreText := fmt.Sprintf("Score: %.2f, Confidence: %.2f\nReasoning: %s",
			score.Score, score.Confidence, score.Reasoning)
		sanitized[i] = vu.sanitizeUserContent(scoreText)
	}
	return sanitized
}

// buildVerificationPrompt creates the verification prompt using the template
// with prompt injection protection.
func (vu *VerificationUnit) buildVerificationPrompt(
	question string,
	answers []domain.Answer,
	judgeScores []domain.JudgeSummary,
) (string, error) {
	var promptBuf bytes.Buffer
	templateData := struct {
		Question    string
		Answers     []string
		JudgeScores []string
	}{
		Question:    vu.sanitizeUserContent(question),
		Answers:     vu.sanitizeAnswers(answers),
		JudgeScores: vu.sanitizeJudgeScores(judgeScores),
	}

	if err := vu.promptTemplate.Execute(&promptBuf, templateData); err != nil {
		return "", fmt.Errorf("unit %s: failed to execute prompt template: %w", vu.name, err)
	}

	basePrompt := promptBuf.String()
	// Instruct the LLM to respond in a specific JSON format for reliable parsing.
	prompt := basePrompt + "\n\nIMPORTANT: You must respond with valid JSON in exactly this format:\n" +
		`{\"confidence\": <0.0-1.0>, \"reasoning\": \"<detailed explanation>\", \"issues\": [<optional list of issues>], \"recommendation\": \"<optional recommendation>\", \"version\": 1}`

	return prompt, nil
}

// estimateTokens provides a rough estimate of token count for text.
func (vu *VerificationUnit) estimateTokens(text string) int {
	// This is a conservative heuristic: ~4 characters per token.
	return len(text) / 4
}

// getModelContextLimit returns a conservative context limit for the LLM model.
func (vu *VerificationUnit) getModelContextLimit() int {
	model := strings.ToLower(vu.llmClient.GetModel())
	switch {
	case strings.Contains(model, "gpt-4"):
		return 6000 // For GPT-4 variants (8K-128K context)
	case strings.Contains(model, "gpt-3.5"):
		return 3000 // For GPT-3.5 variants (4K-16K context)
	case strings.Contains(model, "claude"):
		return 8000 // For Claude variants (8K-200K context)
	default:
		return 2000 // Conservative default for unknown models
	}
}

// truncateAnswersIfNeeded truncates answer content if the prompt would
// exceed the model's context limit.
func (vu *VerificationUnit) truncateAnswersIfNeeded(
	answers []domain.Answer,
	judgeScores []domain.JudgeSummary,
	question string,
	maxPromptTokens int,
) []domain.Answer {
	questionTokens := vu.estimateTokens(question)
	judgeTokens := 0
	for _, score := range judgeScores {
		judgeTokens += vu.estimateTokens(fmt.Sprintf("Score: %.2f, Reasoning: %s", score.Score, score.Reasoning))
	}

	// Estimate template and instruction overhead.
	templateOverhead := 500
	baseTokens := questionTokens + judgeTokens + templateOverhead
	availableForAnswers := maxPromptTokens - baseTokens
	if availableForAnswers <= 0 {
		return []domain.Answer{}
	}

	currentAnswerTokens := 0
	for _, answer := range answers {
		currentAnswerTokens += vu.estimateTokens(answer.Content)
	}

	if currentAnswerTokens <= availableForAnswers {
		return answers
	}

	// Truncate answers proportionally if they exceed the available space.
	tokensPerAnswer := availableForAnswers / len(answers)
	maxCharsPerAnswer := tokensPerAnswer * 4

	truncatedAnswers := make([]domain.Answer, len(answers))
	for i, answer := range answers {
		if len(answer.Content) <= maxCharsPerAnswer {
			truncatedAnswers[i] = answer
		} else {
			truncatedContent := answer.Content[:maxCharsPerAnswer] + "... [truncated]"
			truncatedAnswers[i] = domain.Answer{Content: truncatedContent}
		}
	}
	return truncatedAnswers
}

// isRetryableError determines if an error is likely transient and worth retrying.
func (vu *VerificationUnit) isRetryableError(err error) bool {
	if err == nil {
		return false
	}
	errStr := strings.ToLower(err.Error())
	retryablePatterns := []string{
		"rate limit", "too many requests", "timeout", "connection refused",
		"connection reset", "temporary failure", "service unavailable",
		"internal server error", "bad gateway", "gateway timeout", "network",
	}
	for _, pattern := range retryablePatterns {
		if strings.Contains(errStr, pattern) {
			return true
		}
	}
	return false
}

// calculateRetryDelay calculates the delay for exponential backoff with jitter.
func (vu *VerificationUnit) calculateRetryDelay(attempt int) time.Duration {
	delay := BaseRetryDelay * time.Duration(1<<attempt)
	if delay > MaxRetryDelay {
		delay = MaxRetryDelay
	}

	// Add jitter to prevent thundering herd.
	jitter := int64(float64(delay) * RetryJitterPercentage)
	if jitter > 0 {
		//nolint:gosec // G404: math/rand is acceptable for retry jitter timing
		delay += time.Duration(rand.Int64N(2*jitter) - jitter)
	}

	if delay < 0 {
		return BaseRetryDelay
	}
	return delay
}

// callVerificationLLM invokes the LLM with retry logic.
func (vu *VerificationUnit) callVerificationLLM(ctx context.Context, prompt string) (string, int, int, error) {
	promptTokens := vu.estimateTokens(prompt)
	contextLimit := vu.getModelContextLimit()
	if promptTokens > contextLimit {
		return "", 0, 0, fmt.Errorf("unit %s: prompt too large (%d tokens) for model context limit (%d)",
			vu.name, promptTokens, contextLimit)
	}

	options := map[string]any{
		"temperature": vu.config.Temperature,
		"max_tokens":  vu.config.MaxTokens,
	}
	if supportsJSONMode(vu.llmClient) {
		options["response_format"] = map[string]string{"type": "json_object"}
	}

	var lastErr error
	for attempt := 0; attempt <= MaxRetryAttempts; attempt++ {
		response, tokensIn, tokensOut, err := vu.llmClient.CompleteWithUsage(ctx, prompt, options)
		if err == nil {
			return response, tokensIn, tokensOut, nil
		}
		lastErr = err
		if attempt == MaxRetryAttempts || !vu.isRetryableError(err) {
			break
		}
		select {
		case <-ctx.Done():
			return "", 0, 0, fmt.Errorf("unit %s: context cancelled during retry: %w", vu.name, ctx.Err())
		case <-time.After(vu.calculateRetryDelay(attempt)):
		}
	}
	return "", 0, 0, fmt.Errorf("unit %s: LLM call failed after %d attempts: %w", vu.name, MaxRetryAttempts+1, lastErr)
}

// updateVerdictWithVerification updates the verdict based on verification results.
func (vu *VerificationUnit) updateVerdictWithVerification(
	state domain.State,
	verificationResp *LLMVerificationResponse,
) (domain.State, error) {
	verdict, err := vu.getVerdictFromState(state)
	if err != nil {
		return state, err
	}

	if verificationResp.Confidence < vu.config.ConfidenceThreshold {
		verdict.RequiresHumanReview = true
	}

	return domain.With(state, domain.KeyVerdict, verdict), nil
}

// addVerificationTrace adds the verification trace to the state if debug
// tracing is enabled.
func (vu *VerificationUnit) addVerificationTrace(
	state domain.State,
	verificationResp *LLMVerificationResponse,
) domain.State {
	if vu.getTraceLevelFromState(state) == "debug" {
		trace := VerificationTrace{
			Confidence:     verificationResp.Confidence,
			Reasoning:      verificationResp.Reasoning,
			Issues:         verificationResp.Issues,
			Recommendation: verificationResp.Recommendation,
		}
		// Serialize trace to JSON string for storage
		traceJSON, err := json.Marshal(trace)
		if err != nil {
			// If marshaling fails, just store a simple string
			return domain.With(state, domain.KeyVerificationTrace, fmt.Sprintf("confidence: %.2f", verificationResp.Confidence))
		}
		return domain.With(state, domain.KeyVerificationTrace, string(traceJSON))
	}
	return state
}

// safeAddTokens safely adds token counts to avoid integer overflow.
func (vu *VerificationUnit) safeAddTokens(current, tokensIn, tokensOut int) int {
	// Check for negative inputs
	if tokensIn < 0 || tokensOut < 0 || current < 0 {
		return current // Invalid input, return current value
	}

	// Check for overflow before addition
	maxInt := int(^uint(0) >> 1)
	if current > maxInt-tokensIn-tokensOut {
		return maxInt // Would overflow, return max int
	}

	return current + tokensIn + tokensOut
}

// safeIncrementCalls safely increments a call count to avoid integer overflow.
func (vu *VerificationUnit) safeIncrementCalls(current int) int {
	maxInt := int(^uint(0) >> 1)
	if current == maxInt {
		return current
	}
	return current + 1
}

// updateBudgetWithTokens updates the budget with token usage from verification.
func (vu *VerificationUnit) updateBudgetWithTokens(state domain.State, tokensIn, tokensOut int) domain.State {
	if budget := vu.getBudgetFromState(state); budget != nil {
		budget.TokensUsed = vu.safeAddTokens(budget.TokensUsed, tokensIn, tokensOut)
		budget.CallsMade = vu.safeIncrementCalls(budget.CallsMade)
		return domain.With(state, domain.KeyBudget, budget)
	}
	return state
}

// Execute verifies the quality of judging results. It retrieves data from the
// state, performs verification using an LLM, and updates the verdict with a
// human review flag based on a confidence threshold.
func (vu *VerificationUnit) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	question, answers, judgeScores, err := vu.extractVerificationInputs(state)
	if err != nil {
		return state, err
	}

	contextLimit := vu.getModelContextLimit()
	truncatedAnswers := vu.truncateAnswersIfNeeded(answers, judgeScores, question, contextLimit)

	prompt, err := vu.buildVerificationPrompt(question, truncatedAnswers, judgeScores)
	if err != nil {
		return state, err
	}

	response, tokensIn, tokensOut, err := vu.callVerificationLLM(ctx, prompt)
	if err != nil {
		return state, err
	}

	verificationResp, err := vu.parseLLMResponse(response)
	if err != nil {
		return state, fmt.Errorf("unit %s: failed to parse LLM response: %w", vu.name, err)
	}

	state, err = vu.updateVerdictWithVerification(state, verificationResp)
	if err != nil {
		return state, err
	}

	state = vu.addVerificationTrace(state, verificationResp)
	state = vu.updateBudgetWithTokens(state, tokensIn, tokensOut)

	return state, nil
}

// Validate checks if the unit is properly configured and ready for execution.
func (vu *VerificationUnit) Validate() error {
	_, err := vu.validateAndCompileConfig(vu.config, vu.llmClient, vu.name)
	return err
}

// parseLLMResponse extracts verification data from an LLM's JSON response.
func (vu *VerificationUnit) parseLLMResponse(response string) (*LLMVerificationResponse, error) {
	jsonStr := extractJSON(response)
	if jsonStr == "" {
		return nil, fmt.Errorf("no valid JSON found in LLM response (len: %d)", len(response))
	}

	var llmResponse LLMVerificationResponse
	if err := json.Unmarshal([]byte(jsonStr), &llmResponse); err != nil {
		return nil, fmt.Errorf("failed to parse JSON response (len: %d): %w", len(jsonStr), err)
	}

	if err := vu.validator.Struct(llmResponse); err != nil {
		return nil, fmt.Errorf("invalid response structure: %w", err)
	}

	return &llmResponse, nil
}

// UnmarshalParameters deserializes YAML parameters and returns a new, updated
// VerificationUnit instance to maintain thread-safety.
func (vu *VerificationUnit) UnmarshalParameters(params yaml.Node) (*VerificationUnit, error) {
	var config VerificationConfig
	if err := params.Decode(&config); err != nil {
		return nil, fmt.Errorf("failed to decode parameters: %w", err)
	}

	tmpl, err := vu.validateAndCompileConfig(config, vu.llmClient, vu.name)
	if err != nil {
		return nil, err
	}

	return &VerificationUnit{
		name:           vu.name,
		config:         config,
		llmClient:      vu.llmClient,
		validator:      vu.validator,
		promptTemplate: tmpl,
	}, nil
}

// CreateVerificationUnit is a factory function that creates a VerificationUnit
// from a configuration map, for use with the UnitRegistry.
func CreateVerificationUnit(id string, config map[string]any) (*VerificationUnit, error) {
	llmClient, ok := config["llm_client"].(ports.LLMClient)
	if !ok {
		return nil, fmt.Errorf("llm_client is required and must implement ports.LLMClient")
	}

	verificationConfig := defaultVerificationConfig()
	if val, ok := config["prompt_template"].(string); ok {
		verificationConfig.PromptTemplate = val
	}
	if val, ok := config["confidence_threshold"]; ok {
		if f, err := strconv.ParseFloat(fmt.Sprintf("%v", val), 64); err == nil {
			verificationConfig.ConfidenceThreshold = f
		}
	}
	if val, ok := config["temperature"]; ok {
		if f, err := strconv.ParseFloat(fmt.Sprintf("%v", val), 64); err == nil {
			verificationConfig.Temperature = f
		}
	}
	if val, ok := config["max_tokens"]; ok {
		if i, err := strconv.Atoi(fmt.Sprintf("%v", val)); err == nil {
			verificationConfig.MaxTokens = i
		}
	}

	return NewVerificationUnit(id, llmClient, verificationConfig)
}
