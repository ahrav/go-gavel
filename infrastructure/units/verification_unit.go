// Package units contains concrete implementations of the ports.Unit interface.
package units

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"text/template"
	"time"

	"github.com/go-playground/validator/v10"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
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
	tracer         trace.Tracer
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
// when verifying judging results. All fields are validated using struct tags
// to ensure response integrity and prevent malformed data from affecting
// the verification process.
type LLMVerificationResponse struct {
	// Confidence is how confident the verifier is in the judging (0.0-1.0).
	// Values below the configured threshold trigger human review flags.
	Confidence float64 `json:"confidence" validate:"required,min=0.0,max=1.0"`

	// Reasoning provides the detailed explanation for the verification decision.
	// Minimum 10 characters required to ensure meaningful feedback.
	Reasoning string `json:"reasoning" validate:"required,min=10"`

	// Issues lists any potential problems found during verification.
	// Empty slice indicates no issues detected.
	Issues []string `json:"issues,omitempty"`

	// Recommendation provides actionable feedback for improvement.
	// Optional field that may contain suggestions for better evaluation.
	Recommendation string `json:"recommendation,omitempty"`

	// Version allows for future schema evolution.
	// Currently optional but recommended for forward compatibility.
	Version int `json:"version,omitempty"`
}

// VerificationTrace captures verification output for debug tracing.
// This structure is included in the verdict when the trace level is debug,
// providing detailed verification information for analysis and debugging.
// The trace is serialized to JSON and stored in the state under KeyVerificationTrace.
type VerificationTrace struct {
	// Confidence from the LLM verification response.
	Confidence float64 `json:"confidence"`
	// Reasoning explanation from the LLM verification.
	Reasoning string `json:"reasoning"`
	// Issues found during verification, if any.
	Issues []string `json:"issues,omitempty"`
	// Recommendation for improvement, if provided.
	Recommendation string `json:"recommendation,omitempty"`
}

// defaultVerificationConfig returns a VerificationConfig with sensible defaults
// for production use. The default prompt template includes security protections
// against prompt injection and provides comprehensive evaluation criteria.
// Default values prioritize reliable verification with conservative token usage.
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

// validateVerificationConfig validates a VerificationConfig using struct tags
// and the provided validator instance. Ensures all required fields are present,
// numeric values are within acceptable ranges, and the prompt template contains
// minimum required content for effective verification.
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

	tmpl, err := template.New("verificationPrompt").Funcs(GetTemplateFuncMap()).Parse(config.PromptTemplate)
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

	if llmClient == nil {
		return nil, fmt.Errorf("unit %s: LLM client cannot be nil", name)
	}

	unit := &VerificationUnit{
		name:      name,
		config:    config,
		llmClient: llmClient,
		validator: validator.New(),
		tracer:    otel.Tracer("verification-unit"),
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

// getQuestionFromState extracts the evaluation question from the state.
// Returns an error with unit context if the question is not found.
func (vu *VerificationUnit) getQuestionFromState(state domain.State) (string, error) {
	question, ok := domain.Get(state, domain.KeyQuestion)
	if !ok {
		return "", fmt.Errorf("unit %s: question not found in state", vu.name)
	}
	return question, nil
}

// getAnswersFromState extracts the candidate answers from the state.
// Returns an error with unit context if answers are not found.
func (vu *VerificationUnit) getAnswersFromState(state domain.State) ([]domain.Answer, error) {
	answers, ok := domain.Get(state, domain.KeyAnswers)
	if !ok {
		return nil, fmt.Errorf("unit %s: answers not found in state", vu.name)
	}
	return answers, nil
}

// getJudgeScoresFromState extracts judge scoring results from the state.
// Returns an error if no judge scores are found, as verification requires
// existing judgments to analyze.
func (vu *VerificationUnit) getJudgeScoresFromState(state domain.State) ([]domain.JudgeSummary, error) {
	judgeScores, ok := domain.Get(state, domain.KeyJudgeScores)
	if !ok || len(judgeScores) == 0 {
		return nil, fmt.Errorf("unit %s: no judge scores found to verify", vu.name)
	}
	return judgeScores, nil
}

// getVerdictFromState extracts the current verdict from the state.
// The verdict will be updated with human review flags based on verification results.
func (vu *VerificationUnit) getVerdictFromState(state domain.State) (*domain.Verdict, error) {
	verdict, ok := domain.Get(state, domain.KeyVerdict)
	if !ok {
		return nil, fmt.Errorf("unit %s: verdict not found in state", vu.name)
	}
	return verdict, nil
}

// getBudgetFromState extracts the budget report from the state.
// Returns nil if no budget tracking is configured. Used for token usage accounting.
func (vu *VerificationUnit) getBudgetFromState(state domain.State) *domain.BudgetReport {
	budget, _ := domain.Get(state, domain.KeyBudget)
	return budget
}

// getTraceLevelFromState extracts the trace level setting from the state.
// Returns empty string if not configured. Debug level enables verification tracing.
func (vu *VerificationUnit) getTraceLevelFromState(state domain.State) string {
	traceLevel, _ := domain.Get(state, domain.KeyTraceLevel)
	return strings.ToLower(traceLevel)
}

// extractVerificationInputs retrieves all required data from the state
// for verification analysis. Returns the question, answers, and judge scores
// or an error if any required component is missing.
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

// sanitizeUserContent protects against prompt injection attacks by wrapping
// user-provided content in markdown code blocks and escaping existing delimiters.
// This security measure prevents malicious inputs from breaking out of their
// designated content areas and injecting commands into the verification prompt.
func (vu *VerificationUnit) sanitizeUserContent(content string) string {
	content = strings.ReplaceAll(content, "```", "'''")
	return "```\n" + content + "\n```\n"
}

// sanitizeAnswers applies security sanitization to all answer content
// to prevent prompt injection attacks.
func (vu *VerificationUnit) sanitizeAnswers(answers []domain.Answer) []string {
	sanitized := make([]string, len(answers))
	for i, answer := range answers {
		sanitized[i] = vu.sanitizeUserContent(answer.Content)
	}
	return sanitized
}

// sanitizeJudgeScores formats and sanitizes judge scoring data
// for safe inclusion in verification prompts.
func (vu *VerificationUnit) sanitizeJudgeScores(judgeScores []domain.JudgeSummary) []string {
	sanitized := make([]string, len(judgeScores))
	for i, score := range judgeScores {
		scoreText := fmt.Sprintf("Score: %.2f, Confidence: %.2f\nReasoning: %s",
			score.Score, score.Confidence, score.Reasoning)
		sanitized[i] = vu.sanitizeUserContent(scoreText)
	}
	return sanitized
}

// buildVerificationPrompt creates the verification prompt using the Go template
// with sanitized user content to prevent prompt injection attacks.
// The function applies security protections to all user inputs and appends
// JSON format instructions to ensure reliable LLM response parsing.
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

// estimateTokens provides a conservative estimate of token count for text
// using a heuristic of approximately 4 characters per token.
// This estimation is used for context limit checking and prompt truncation.
// Actual token counts may vary based on model tokenizer and content type.
func (vu *VerificationUnit) estimateTokens(text string) int {
	// This is a conservative heuristic: ~4 characters per token.
	return len(text) / 4
}

// getModelContextLimit returns a conservative context limit for the LLM model
// based on model name heuristics. These limits are intentionally conservative
// to prevent context overflow and ensure reliable prompt processing.
// Production systems should expose actual context limits through the client interface.
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

// truncateAnswersIfNeeded truncates answer content proportionally when
// the complete prompt would exceed the model's context limit.
// Preserves all answers but reduces their content length to fit within
// available token budget after accounting for question, judge scores, and template overhead.
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

// callVerificationLLM invokes the LLM client to perform verification analysis.
// Configures temperature, max tokens, and JSON response format when supported.
// Returns the response text along with input/output token counts for budget tracking.
// Retry logic is handled by the RetryingLLMClient middleware.
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

	// The retry logic is now handled by the RetryingLLMClient middleware
	return vu.llmClient.CompleteWithUsage(ctx, prompt, options)
}

// updateVerdictWithVerification updates the verdict's RequiresHumanReview flag
// based on the verification confidence score compared to the configured threshold.
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

// addVerificationTrace adds detailed verification information to the state
// when debug tracing is enabled. The trace is serialized to JSON for storage.
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

// safeAddTokens safely adds token counts with overflow protection.
// Validates input parameters and prevents integer overflow when accumulating
// token usage across multiple LLM calls. Returns the maximum integer value
// if overflow would occur, ensuring budget tracking remains stable.
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

// safeIncrementCalls safely increments a call count with overflow protection.
// Prevents integer overflow when tracking LLM API calls in long-running processes.
// Returns the current value if incrementing would cause overflow.
func (vu *VerificationUnit) safeIncrementCalls(current int) int {
	maxInt := int(^uint(0) >> 1)
	if current == maxInt {
		return current
	}
	return current + 1
}

// updateBudgetWithTokens updates the budget report with token usage
// and call count from the verification LLM request. Uses safe arithmetic
// to prevent integer overflow in long-running processes.
func (vu *VerificationUnit) updateBudgetWithTokens(state domain.State, tokensIn, tokensOut int) domain.State {
	if budget := vu.getBudgetFromState(state); budget != nil {
		budget.TokensUsed = vu.safeAddTokens(budget.TokensUsed, tokensIn, tokensOut)
		budget.CallsMade = vu.safeIncrementCalls(budget.CallsMade)
		return domain.With(state, domain.KeyBudget, budget)
	}
	return state
}

// Execute verifies the quality of judging results by analyzing them with an LLM.
// It extracts the question, answers, and judge scores from the state, builds a
// verification prompt with security protections, and calls the LLM for analysis.
//
// The method updates the verdict's RequiresHumanReview flag when the LLM's
// confidence score falls below the configured threshold. Token usage is tracked
// in the budget, and debug traces are added when trace level is set to "debug".
//
// Context cancellation is supported throughout the LLM call chain.
// Returns an error if required state data is missing or LLM analysis fails.
func (vu *VerificationUnit) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	_, span := vu.tracer.Start(ctx, "VerificationUnit.Execute",
		trace.WithAttributes(
			attribute.String("unit.type", "verification"),
			attribute.String("unit.id", vu.name),
			attribute.Float64("config.confidence_threshold", vu.config.ConfidenceThreshold),
			attribute.Float64("config.temperature", vu.config.Temperature),
			attribute.Int("config.max_tokens", vu.config.MaxTokens),
		),
	)
	defer span.End()

	start := time.Now()

	question, answers, judgeScores, err := vu.extractVerificationInputs(state)
	if err != nil {
		span.RecordError(err)
		return state, err
	}

	contextLimit := vu.getModelContextLimit()
	truncatedAnswers := vu.truncateAnswersIfNeeded(answers, judgeScores, question, contextLimit)

	prompt, err := vu.buildVerificationPrompt(question, truncatedAnswers, judgeScores)
	if err != nil {
		span.RecordError(err)
		return state, err
	}

	response, tokensIn, tokensOut, err := vu.callVerificationLLM(ctx, prompt)
	if err != nil {
		err := fmt.Errorf("unit %s: LLM call failed: %w", vu.name, err)
		span.RecordError(err)
		return state, err
	}

	verificationResp, err := vu.parseLLMResponse(response)
	if err != nil {
		err := fmt.Errorf("unit %s: failed to parse LLM response: %w", vu.name, err)
		span.RecordError(err)
		return state, err
	}

	state, err = vu.updateVerdictWithVerification(state, verificationResp)
	if err != nil {
		span.RecordError(err)
		return state, err
	}

	state = vu.addVerificationTrace(state, verificationResp)
	state = vu.updateBudgetWithTokens(state, tokensIn, tokensOut)

	latency := time.Since(start)
	span.SetAttributes(
		attribute.Int64("eval.latency_ms", latency.Milliseconds()),
		attribute.Int("eval.answers_count", len(answers)),
		attribute.Int("eval.judge_scores_count", len(judgeScores)),
		attribute.Int("eval.question_length", len(question)),
		attribute.Float64("eval.verification_confidence", verificationResp.Confidence),
		attribute.Bool("eval.requires_human_review", verificationResp.Confidence < vu.config.ConfidenceThreshold),
		attribute.Int("eval.tokens_in", tokensIn),
		attribute.Int("eval.tokens_out", tokensOut),
		attribute.Bool("no_llm_cost", false), // LLM-based units have cost
	)

	return state, nil
}

// Validate checks if the unit is properly configured and ready for execution.
// Verifies that the LLM client is available, configuration is valid,
// and the prompt template compiles successfully. This method should be called
// before using the unit in an evaluation pipeline.
func (vu *VerificationUnit) Validate() error {
	_, err := vu.validateAndCompileConfig(vu.config, vu.llmClient, vu.name)
	return err
}

// parseLLMResponse extracts and validates verification data from an LLM's JSON response.
// Uses extractJSON to handle various response formats (markdown blocks, plain JSON)
// and validates the parsed structure using struct tags to ensure data integrity.
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

// UnmarshalParameters deserializes YAML parameters and returns a new
// VerificationUnit instance with the updated configuration.
// This method maintains immutability and thread-safety by creating a new
// instance rather than modifying the existing one. The new instance shares
// the same LLM client but uses the updated configuration and recompiled template.
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
		tracer:         otel.Tracer("verification-unit"),
	}, nil
}

// NewVerificationFromConfig creates a VerificationUnit from a configuration map.
// This is the boundary adapter for YAML/JSON configuration.
// Verification requires an LLM client for quality verification.
func NewVerificationFromConfig(id string, config map[string]any, llm ports.LLMClient) (ports.Unit, error) {
	if llm == nil {
		return nil, fmt.Errorf("LLM client cannot be nil")
	}

	// Use yaml marshaling for clean conversion
	data, err := yaml.Marshal(config)
	if err != nil {
		return nil, fmt.Errorf("marshal config: %w", err)
	}

	// Start with defaults, then overlay user config
	cfg := defaultVerificationConfig()
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	return NewVerificationUnit(id, llm, cfg)
}
