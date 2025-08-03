package testutils

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"regexp"
	"strings"
	"sync"
	"time"
)

// BenchmarkMockLLMClient is a specialized mock LLM client for benchmark testing.
// It provides deterministic but realistic scoring based on comparing answers to ground truth.
type BenchmarkMockLLMClient struct {
	*MockLLMClient

	// groundTruthMap maps question IDs to their correct answer IDs
	groundTruthMap map[string]string

	// questionAnswerMap maps question content to answer content and IDs
	questionAnswerMap map[string]map[string]string // question -> answer content -> answer ID

	// answerToQuestion maps answer content back to question for context
	answerToQuestion map[string]string // answer content -> question content

	// questionContentToID maps question content to question ID
	questionContentToID map[string]string // question content -> question ID

	// judgePersonality defines how this judge scores answers
	judgePersonality JudgePersonality

	// config contains all configurable parameters for judge behavior
	config JudgeConfig

	// biasPosition adds positional bias (0 = no bias, positive = favor later answers)
	biasPosition float64

	// mu protects rng access for thread safety
	mu sync.Mutex

	// rng is the instance-specific random number generator for thread safety
	rng *rand.Rand

	// ensembleConfig is optional and used for correlated errors
	ensembleConfig *EnsembleConfig

	// sharedErrorState tracks errors made by other judges for correlation
	sharedErrorState map[string]float64 // questionID -> error amount

	// Catastrophic failure simulation fields
	timeoutDelay       time.Duration // If non-zero, Complete will timeout after this delay
	simulatePartial    bool          // If true, return incomplete JSON
	simulateMalformed  bool          // If true, return malformed JSON
	networkFailureRate float64       // Probability of network failure (0.0-1.0)
	rateLimitAfter     int           // Simulate rate limiting after N requests
	requestCount       int           // Track number of requests for rate limiting
}

// JudgePersonality defines different scoring behaviors for judges.
type JudgePersonality string

const (
	// ConservativeJudge tends to score lower but is more accurate
	ConservativeJudge JudgePersonality = "conservative"

	// ComprehensiveJudge balances multiple factors
	ComprehensiveJudge JudgePersonality = "comprehensive"

	// AnalyticalJudge focuses on logical reasoning
	AnalyticalJudge JudgePersonality = "analytical"

	// BiasedJudge has systematic bias toward certain positions
	BiasedJudge JudgePersonality = "biased"
)

// NewBenchmarkMockLLMClient creates a mock LLM client for benchmark testing.
func NewBenchmarkMockLLMClient(
	model string, dataset *BenchmarkDataset, personality JudgePersonality,
) *BenchmarkMockLLMClient {
	groundTruthMap := make(map[string]string)
	questionAnswerMap := make(map[string]map[string]string)
	answerToQuestion := make(map[string]string)
	questionContentToID := make(map[string]string)

	for _, q := range dataset.Questions {
		groundTruthMap[q.ID] = q.GroundTruthID
		questionContentToID[q.Question] = q.ID

		answerMap := make(map[string]string)
		for _, answer := range q.Answers {
			answerMap[answer.Content] = answer.ID
			answerToQuestion[answer.Content] = q.Question
		}
		questionAnswerMap[q.Question] = answerMap
	}

	var seed int64
	for _, c := range model {
		seed += int64(c)
	}
	for _, c := range string(personality) {
		seed += int64(c)
	}

	return &BenchmarkMockLLMClient{
		MockLLMClient:       NewMockLLMClient(model),
		groundTruthMap:      groundTruthMap,
		questionAnswerMap:   questionAnswerMap,
		answerToQuestion:    answerToQuestion,
		questionContentToID: questionContentToID,
		judgePersonality:    personality,
		config:              DefaultJudgeConfig(),
		biasPosition:        0.0,
		// G404: Intentionally using weak RNG for deterministic test behavior
		rng: rand.New(rand.NewSource(seed)), //nolint:gosec // Fixed seed for reproducible tests
	}
}

// NewBenchmarkMockLLMClientWithConfig creates a mock LLM client with custom configuration.
func NewBenchmarkMockLLMClientWithConfig(
	model string, dataset *BenchmarkDataset, personality JudgePersonality, config JudgeConfig,
) *BenchmarkMockLLMClient {
	client := NewBenchmarkMockLLMClient(model, dataset, personality)
	client.config = config
	return client
}

// NewBiasedBenchmarkMockLLMClient creates a mock with intentional bias for testing bias mitigation.
func NewBiasedBenchmarkMockLLMClient(
	model string,
	dataset *BenchmarkDataset,
	biasStrength float64,
) *BenchmarkMockLLMClient {
	client := NewBenchmarkMockLLMClient(model, dataset, BiasedJudge)
	client.biasPosition = biasStrength // Positive values favor later answers
	return client
}

// Complete provides realistic scoring based on ground truth with personality variations.
func (m *BenchmarkMockLLMClient) Complete(
	ctx context.Context, prompt string, options map[string]any,
) (string, error) {
	m.mu.Lock()
	m.requestCount++
	requestCount := m.requestCount
	m.mu.Unlock()

	// Check for catastrophic failures first
	if err := m.checkCatastrophicFailures(ctx); err != nil {
		return "", err
	}

	// Check for rate limiting
	m.mu.Lock()
	rateLimitAfter := m.rateLimitAfter
	m.mu.Unlock()
	if rateLimitAfter > 0 && requestCount > rateLimitAfter {
		return "", fmt.Errorf("rate limit exceeded: too many requests")
	}

	if m.overrideError != nil {
		return "", m.overrideError
	}

	if m.overrideResponse != "" {
		return m.overrideResponse, nil
	}

	if prompt == "" {
		return "", fmt.Errorf("prompt cannot be empty")
	}

	// Check if this is a scoring request
	if strings.Contains(prompt, "Rate") || strings.Contains(prompt, "score") ||
		strings.Contains(prompt, "evaluate") || strings.Contains(prompt, "Score") {
		response, err := m.generateScoreResponse(prompt)
		if err != nil {
			return "", err
		}

		// Apply catastrophic response modifications if enabled
		return m.applyCatastrophicResponseModifications(response), nil
	}

	// For non-scoring requests, fall back to default behavior
	return m.MockLLMClient.Complete(ctx, prompt, options)
}

// generateScoreResponse creates a realistic score based on ground truth and judge personality.
func (m *BenchmarkMockLLMClient) generateScoreResponse(prompt string) (string, error) {

	questionContent, answerContent := m.extractQuestionAndAnswer(prompt)

	var answerID string
	if answerMap, hasQuestion := m.questionAnswerMap[questionContent]; hasQuestion {
		if id, hasAnswer := answerMap[answerContent]; hasAnswer {
			answerID = id
		}
	}

	// For now, we don't need the question ID since we're using content-based matching

	baseScore := m.calculateBaseScore(questionContent, answerContent, answerID)

	score := m.applyPersonalityModifiers(baseScore, answerID)

	score = m.applyNoise(score)

	// Ensure score is in valid range (MUST be done after all modifications)
	score = clamp(score, 0.0, 1.0)

	// Generate confidence based on score distance from 0.5
	confidence := 0.7 + 0.3*abs(score-0.5)*2
	confidence = clamp(confidence, 0.0, 1.0) // Ensure confidence is also valid

	// Create reasoning based on personality
	reasoning := m.generateReasoning(score, m.judgePersonality)

	response := map[string]any{
		"score":      score,
		"confidence": confidence,
		"reasoning":  reasoning,
		"version":    1,
	}

	responseJSON, _ := json.Marshal(response)

	return string(responseJSON), nil
}

// CompleteWithUsage provides the same functionality as Complete but also returns token usage.
func (m *BenchmarkMockLLMClient) CompleteWithUsage(
	ctx context.Context, prompt string, options map[string]any,
) (output string, tokensIn, tokensOut int, err error) {
	response, err := m.Complete(ctx, prompt, options)
	if err != nil {
		return "", 0, 0, err
	}

	tokensIn, _ = m.EstimateTokens(prompt)
	tokensOut, _ = m.EstimateTokens(response)

	return response, tokensIn, tokensOut, nil
}

// extractQuestionAndAnswer uses regex to robustly extract question and answer from prompt text.
func (m *BenchmarkMockLLMClient) extractQuestionAndAnswer(prompt string) (questionContent, answerContent string) {
	// Define regex patterns for more flexible matching
	// These patterns handle various formats and are case-insensitive
	questionRegex := regexp.MustCompile(`(?is)question:\s*(.+?)(?:\n\s*answer:|$)`)
	answerRegex := regexp.MustCompile(`(?is)answer:\s*(.+?)(?:\n\n|consider|score|provide|evaluate|$)`)

	// Extract question content
	if matches := questionRegex.FindStringSubmatch(prompt); len(matches) > 1 {
		questionContent = strings.TrimSpace(matches[1])
	}

	// Extract answer content
	if matches := answerRegex.FindStringSubmatch(prompt); len(matches) > 1 {
		answerContent = strings.TrimSpace(matches[1])
	}

	// Fallback to simpler patterns if needed
	if questionContent == "" {
		// Try simpler pattern for question
		simpleQuestionRegex := regexp.MustCompile(`(?i)(?:q:|question:)\s*([^\n]+)`)
		if matches := simpleQuestionRegex.FindStringSubmatch(prompt); len(matches) > 1 {
			questionContent = strings.TrimSpace(matches[1])
		}
	}

	if answerContent == "" {
		// Try simpler pattern for answer
		simpleAnswerRegex := regexp.MustCompile(`(?i)(?:a:|answer:)\s*([^\n]+)`)
		if matches := simpleAnswerRegex.FindStringSubmatch(prompt); len(matches) > 1 {
			answerContent = strings.TrimSpace(matches[1])
		}
	}

	return questionContent, answerContent
}

// calculateBaseScore returns a base score based on whether the answer is correct.
func (m *BenchmarkMockLLMClient) calculateBaseScore(questionContent, answerContent, answerID string) float64 {
	// Step 1: Get the question ID from content
	questionID, hasQuestion := m.questionContentToID[questionContent]
	if !hasQuestion {
		// Can't find the question, use heuristics
		if strings.Contains(answerContent, "comprehensive") ||
			strings.Contains(answerContent, "correct answer") ||
			len(answerContent) > 80 {
			return 0.70
		}
		return 0.35
	}

	// Step 2: Get the correct answer ID for this question
	correctAnswerID := m.groundTruthMap[questionID]

	// Step 3: Check if the provided answer is correct
	isCorrect := false
	if answerID != "" && answerID == correctAnswerID {
		isCorrect = true
	} else if answerMap, exists := m.questionAnswerMap[questionContent]; exists {
		for content, id := range answerMap {
			if content == answerContent && id == correctAnswerID {
				isCorrect = true
				break
			}
		}
	}

	// Step 4: Simulate realistic judging errors
	// Create a local RNG seeded with question ID for consistency
	errorSeed := int64(0)
	for _, c := range questionID {
		errorSeed += int64(c)
	}
	// Add personality to seed
	for _, c := range string(m.judgePersonality) {
		errorSeed += int64(c)
	}
	localRng := rand.New(rand.NewSource(errorSeed)) //nolint:gosec // Fixed seed for reproducible tests

	// Generate base score
	var baseScore float64
	if isCorrect {
		// Correct answers usually get high scores, but not always
		m.mu.Lock()
		correctAccuracy := m.config.CorrectAnswerAccuracy
		m.mu.Unlock()
		if localRng.Float64() < correctAccuracy {
			baseScore = 0.75 + localRng.Float64()*0.20 // Score between 0.75-0.95
		} else {
			// Sometimes judges miss correct answers - this is an error
			baseError := -0.40 // Negative error for missing correct answer
			correlatedError := m.calculateCorrelatedError(baseError, questionID)
			baseScore = 0.55 + correlatedError // Apply correlated error
		}
	} else {
		// Incorrect answers usually get low scores, but not always
		m.mu.Lock()
		incorrectAccuracy := m.config.IncorrectAnswerAccuracy
		m.mu.Unlock()
		if localRng.Float64() < incorrectAccuracy {
			baseScore = 0.25 + localRng.Float64()*0.25 // Score between 0.25-0.50
		} else {
			// Sometimes judges wrongly favor incorrect answers - this is an error
			baseError := 0.35 // Positive error for favoring incorrect answer
			correlatedError := m.calculateCorrelatedError(baseError, questionID)
			baseScore = 0.50 + correlatedError // Apply correlated error
		}
	}

	return baseScore
}

// applyPersonalityModifiers adjusts score based on judge personality.
func (m *BenchmarkMockLLMClient) applyPersonalityModifiers(baseScore float64, answerID string) float64 {
	score := baseScore

	// Add some randomness based on personality to simulate real judge variance
	personalityNoise := 0.0

	// Apply personality strength scaling
	m.mu.Lock()
	strength := m.config.PersonalityStrength
	m.mu.Unlock()

	switch m.judgePersonality {
	case ConservativeJudge:
		// Conservative judges are significantly more cautious
		conservativeFactor := 0.85 + (0.15 * (1.0 - strength)) // 85% to 100% based on strength
		score *= conservativeFactor

		// Stronger penalty for uncertain answers
		if baseScore < 0.6 {
			uncertaintyPenalty := 0.75 + (0.25 * (1.0 - strength)) // 75% to 100% based on strength
			score *= uncertaintyPenalty
		}

		// Conservative judges rarely give perfect scores
		if score > 0.85 {
			score = 0.85 + (score-0.85)*0.5
		}

	case ComprehensiveJudge:
		// Comprehensive judges are well-balanced but consider answer length
		m.mu.Lock()
		personalityNoise = (m.rng.Float64() - 0.5) * 0.05
		m.mu.Unlock()

		// Slight preference for longer, more comprehensive answers
		answerLength := len(answerID) // This is a proxy; in real scenario would check actual content
		if answerLength > 2 {         // Assuming longer IDs correlate with longer answers
			score += 0.05 * strength
		}

	case AnalyticalJudge:
		// Analytical judges have strong preference for extremes
		if score > 0.6 {
			// Boost high scores more significantly
			boostFactor := 1.15 - (0.15 * (1.0 - strength)) // 1.0 to 1.15 based on strength
			score = score * boostFactor
			if score > 1.0 {
				score = 1.0
			}
		} else {
			// Penalize low scores more significantly
			penaltyFactor := 0.85 + (0.15 * (1.0 - strength)) // 85% to 100% based on strength
			score *= penaltyFactor
		}

		// Analytical judges are more likely to give very high or very low scores
		if score > 0.5 && score < 0.7 {
			// Push away from middle scores
			m.mu.Lock()
			randomValue := m.rng.Float64()
			m.mu.Unlock()
			if randomValue < 0.5 {
				score -= 0.1 * strength
			} else {
				score += 0.1 * strength
			}
		}

	case BiasedJudge:
		// Apply multiple types of bias
		position := m.getAnswerPosition(answerID)

		// Position bias (favor later answers)
		positionBias := m.biasPosition * float64(position) * 0.1 * strength
		score += positionBias

		// Length bias (favor longer answers) - using answerID as proxy
		if len(answerID) > 2 {
			score += 0.05 * strength
		}

		// Recency bias (slight preference for answers that appear to be newer)
		if position >= 2 { // Last two positions
			score += 0.03 * strength
		}
	}

	return score + personalityNoise
}

// getAnswerPosition returns the position of an answer (0-based).
func (m *BenchmarkMockLLMClient) getAnswerPosition(answerID string) int {
	switch answerID {
	case "a1":
		return 0
	case "a2":
		return 1
	case "a3":
		return 2
	case "a4":
		return 3
	default:
		return 0
	}
}

// applyNoise adds controlled randomness to simulate real judge variance.
func (m *BenchmarkMockLLMClient) applyNoise(score float64) float64 {
	m.mu.Lock()
	noise := (m.rng.Float64() - 0.5) * 2 * m.config.NoiseFactor

	// Occasionally add slightly larger errors to simulate real judge mistakes
	if m.rng.Float64() < m.config.LargeErrorProbability {
		noise *= m.config.LargeErrorMultiplier
	}
	m.mu.Unlock()

	return score + noise
}

// generateReasoning creates personality-appropriate reasoning text.
func (m *BenchmarkMockLLMClient) generateReasoning(score float64, personality JudgePersonality) string {
	switch personality {
	case ConservativeJudge:
		if score > 0.7 {
			return "This answer demonstrates solid understanding with accurate information."
		}
		return "The answer lacks sufficient accuracy or detail for a higher score."

	case ComprehensiveJudge:
		if score > 0.7 {
			return "The response addresses the question well with good clarity and completeness."
		}
		return "The answer could be improved with more comprehensive coverage of the topic."

	case AnalyticalJudge:
		if score > 0.7 {
			return "Strong logical structure and reasoning support this answer effectively."
		}
		return "The logical flow and analytical depth need improvement."

	case BiasedJudge:
		if score > 0.7 {
			return "This answer aligns well with expected patterns and shows good understanding."
		}
		return "The response deviates from expected patterns and needs refinement."

	default:
		return "Evaluated based on accuracy, completeness, and relevance to the question."
	}
}

// Helper functions

func clamp(value, min, max float64) float64 {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// SetEnsembleConfig configures the client for ensemble behavior with correlated errors.
func (m *BenchmarkMockLLMClient) SetEnsembleConfig(config *EnsembleConfig) {
	m.ensembleConfig = config
}

// SetSharedErrorState allows sharing error patterns between judges for correlation.
func (m *BenchmarkMockLLMClient) SetSharedErrorState(state map[string]float64) {
	m.sharedErrorState = state
}

// calculateCorrelatedError adjusts errors based on other judges' errors.
func (m *BenchmarkMockLLMClient) calculateCorrelatedError(baseError float64, questionID string) float64 {
	if m.ensembleConfig == nil || m.ensembleConfig.ErrorCorrelation == 0 || m.sharedErrorState == nil {
		return baseError
	}

	// Check if other judges have made errors on this question
	if otherError, exists := m.sharedErrorState[questionID]; exists {
		// Blend base error with other judge's error based on correlation
		correlation := m.ensembleConfig.ErrorCorrelation
		correlatedError := baseError*(1-correlation) + otherError*correlation
		return correlatedError
	}

	// Store this judge's error for other judges to potentially correlate with
	m.sharedErrorState[questionID] = baseError

	return baseError
}

// CreateBenchmarkEnsembleMocks creates a set of mock LLM clients for ensemble testing.
// Returns multiple clients with different personalities to simulate diverse judges.
func CreateBenchmarkEnsembleMocks(dataset *BenchmarkDataset) map[string]*BenchmarkMockLLMClient {
	return map[string]*BenchmarkMockLLMClient{
		"conservative":  NewBenchmarkMockLLMClient("benchmark-conservative-v1", dataset, ConservativeJudge),
		"comprehensive": NewBenchmarkMockLLMClient("benchmark-comprehensive-v1", dataset, ComprehensiveJudge),
		"analytical":    NewBenchmarkMockLLMClient("benchmark-analytical-v1", dataset, AnalyticalJudge),
		"biased":        NewBiasedBenchmarkMockLLMClient("benchmark-biased-v1", dataset, 0.3), // 30% bias toward later answers
	}
}

// CreateCorrelatedBenchmarkEnsembleMocks creates ensemble mocks with correlated errors.
func CreateCorrelatedBenchmarkEnsembleMocks(
	dataset *BenchmarkDataset, ensembleConfig EnsembleConfig,
) map[string]*BenchmarkMockLLMClient {
	mocks := CreateBenchmarkEnsembleMocks(dataset)

	sharedErrorState := make(map[string]float64)

	// Configure each mock with ensemble config and shared state
	for _, mock := range mocks {
		mock.SetEnsembleConfig(&ensembleConfig)
		mock.SetSharedErrorState(sharedErrorState)
	}

	return mocks
}

// Catastrophic failure simulation methods

// SimulateTimeout makes the judge timeout after a delay.
func (m *BenchmarkMockLLMClient) SimulateTimeout(delay time.Duration) {
	m.mu.Lock()
	m.timeoutDelay = delay
	m.mu.Unlock()
}

// SimulatePartialResponse returns incomplete JSON.
func (m *BenchmarkMockLLMClient) SimulatePartialResponse() {
	m.mu.Lock()
	m.simulatePartial = true
	m.mu.Unlock()
}

// SimulateMalformedJSON returns invalid JSON.
func (m *BenchmarkMockLLMClient) SimulateMalformedJSON() {
	m.mu.Lock()
	m.simulateMalformed = true
	m.mu.Unlock()
}

// SimulateNetworkFailure sets the probability of network failures.
func (m *BenchmarkMockLLMClient) SimulateNetworkFailure(rate float64) {
	m.mu.Lock()
	m.networkFailureRate = rate
	m.mu.Unlock()
}

// SimulateRateLimiting triggers rate limiting after N requests.
func (m *BenchmarkMockLLMClient) SimulateRateLimiting(afterRequests int) {
	m.mu.Lock()
	m.rateLimitAfter = afterRequests
	m.mu.Unlock()
}

// ResetFailureSimulation clears all failure simulation settings.
func (m *BenchmarkMockLLMClient) ResetFailureSimulation() {
	m.mu.Lock()
	m.timeoutDelay = 0
	m.simulatePartial = false
	m.simulateMalformed = false
	m.networkFailureRate = 0
	m.rateLimitAfter = 0
	m.requestCount = 0
	m.mu.Unlock()
}

// checkCatastrophicFailures checks and triggers various failure scenarios.
func (m *BenchmarkMockLLMClient) checkCatastrophicFailures(ctx context.Context) error {
	// Check for timeout
	m.mu.Lock()
	timeoutDelay := m.timeoutDelay
	networkFailureRate := m.networkFailureRate
	m.mu.Unlock()

	if timeoutDelay > 0 {
		select {
		case <-time.After(timeoutDelay):
			return fmt.Errorf("request timeout after %v", timeoutDelay)
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	// Check for network failure
	if networkFailureRate > 0 {
		m.mu.Lock()
		randomValue := m.rng.Float64()
		m.mu.Unlock()
		if randomValue < networkFailureRate {
			return fmt.Errorf("network error: connection refused")
		}
	}

	return nil
}

// applyCatastrophicResponseModifications modifies responses for failure testing.
func (m *BenchmarkMockLLMClient) applyCatastrophicResponseModifications(response string) string {
	m.mu.Lock()
	simulatePartial := m.simulatePartial
	simulateMalformed := m.simulateMalformed
	m.mu.Unlock()

	if simulatePartial {
		// Return only part of the response
		if len(response) > 20 {
			return response[:len(response)/2] + "..."
		}
		return response[:10]
	}

	if simulateMalformed {
		// Corrupt the JSON
		return strings.Replace(response, `"score"`, `"scor`, 1)
	}

	return response
}

// SetConfig sets the judge configuration with proper synchronization.
func (m *BenchmarkMockLLMClient) SetConfig(config JudgeConfig) {
	m.mu.Lock()
	m.config = config
	m.mu.Unlock()
}
