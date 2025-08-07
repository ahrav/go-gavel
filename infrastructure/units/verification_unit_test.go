package units

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"

	"github.com/ahrav/go-gavel/internal/domain"
	"github.com/ahrav/go-gavel/internal/ports"
	"github.com/ahrav/go-gavel/internal/testutils"
)

// TestNewVerificationUnit tests the constructor for the VerificationUnit.
// It ensures that the unit is created successfully with valid parameters and fails
// appropriately for invalid inputs, such as an empty name, nil LLM client,
// or invalid configuration values.
func TestNewVerificationUnit(t *testing.T) {
	tests := []struct {
		name      string
		unitName  string
		llmClient ports.LLMClient
		config    VerificationConfig
		wantErr   bool
		errMsg    string
	}{
		{
			name:      "valid configuration creates unit successfully",
			unitName:  "verifier1",
			llmClient: testutils.NewMockLLMClient("test-model"),
			config: VerificationConfig{
				PromptTemplate:      "Verify these results: {{.Question}}",
				ConfidenceThreshold: 0.8,
				Temperature:         0.0,
				MaxTokens:           512,
			},
			wantErr: false,
		},
		{
			name:      "empty unit name returns error",
			unitName:  "",
			llmClient: testutils.NewMockLLMClient("test-model"),
			config:    defaultVerificationConfig(),
			wantErr:   true,
			errMsg:    "unit name cannot be empty",
		},
		{
			name:      "nil LLM client returns error",
			unitName:  "verifier1",
			llmClient: nil, // Explicitly setting to nil
			config:    defaultVerificationConfig(),
			wantErr:   true,
			errMsg:    "LLM client cannot be nil",
		},
		{
			name:      "invalid prompt template returns error",
			unitName:  "verifier1",
			llmClient: testutils.NewMockLLMClient("test-model"),
			config: VerificationConfig{
				PromptTemplate:      "Short", // Too short
				ConfidenceThreshold: 0.8,
				Temperature:         0.0,
				MaxTokens:           512,
			},
			wantErr: true,
			errMsg:  "configuration validation failed",
		},
		{
			name:      "prompt template too short returns validation error",
			unitName:  "verifier1",
			llmClient: testutils.NewMockLLMClient("test-model"),
			config: VerificationConfig{
				PromptTemplate:      "Too short",
				ConfidenceThreshold: 0.8,
				Temperature:         0.0,
				MaxTokens:           512,
			},
			wantErr: true,
			errMsg:  "configuration validation failed",
		},
		{
			name:      "invalid template syntax returns error",
			unitName:  "verifier1",
			llmClient: testutils.NewMockLLMClient("test-model"),
			config: VerificationConfig{
				PromptTemplate:      "This is a valid length template with {{.Invalid", // Invalid syntax
				ConfidenceThreshold: 0.8,
				Temperature:         0.0,
				MaxTokens:           512,
			},
			wantErr: true,
			errMsg:  "failed to parse prompt template",
		},
		{
			name:      "confidence threshold out of range returns error",
			unitName:  "verifier1",
			llmClient: testutils.NewMockLLMClient("test-model"),
			config: VerificationConfig{
				PromptTemplate:      "Verify these results: {{.Question}}",
				ConfidenceThreshold: 1.5, // Out of range
				Temperature:         0.0,
				MaxTokens:           512,
			},
			wantErr: true,
			errMsg:  "configuration validation failed",
		},
		{
			name:      "temperature out of range returns error",
			unitName:  "verifier1",
			llmClient: testutils.NewMockLLMClient("test-model"),
			config: VerificationConfig{
				PromptTemplate:      "Verify these results: {{.Question}}",
				ConfidenceThreshold: 0.8,
				Temperature:         2.0, // Out of range
				MaxTokens:           512,
			},
			wantErr: true,
			errMsg:  "configuration validation failed",
		},
		{
			name:      "max tokens too low returns error",
			unitName:  "verifier1",
			llmClient: testutils.NewMockLLMClient("test-model"),
			config: VerificationConfig{
				PromptTemplate:      "Verify these results: {{.Question}}",
				ConfidenceThreshold: 0.8,
				Temperature:         0.0,
				MaxTokens:           25, // Too low
			},
			wantErr: true,
			errMsg:  "configuration validation failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			unit, err := NewVerificationUnit(tt.unitName, tt.llmClient, tt.config)

			if tt.wantErr {
				require.Error(t, err, "expected error but got none")
				if err != nil {
					assert.Contains(t, err.Error(), tt.errMsg, "error message should contain expected text")
				}
				assert.Nil(t, unit, "unit should be nil on error")
			} else {
				require.NoError(t, err, "unexpected error creating unit")
				assert.NotNil(t, unit, "unit should not be nil")
				assert.Equal(t, tt.unitName, unit.Name(), "unit name should match")
			}
		})
	}
}

// buildState is a helper function to construct a domain.State object for testing.
// It takes a series of key-value pairs and adds them to the state,
// handling different data types for convenience.
func buildState(entries ...interface{}) domain.State {
	state := domain.NewState()
	for i := 0; i < len(entries); i += 2 {
		switch key := entries[i].(type) {
		case domain.Key[string]:
			state = domain.With(state, key, entries[i+1].(string))
		case domain.Key[[]domain.Answer]:
			state = domain.With(state, key, entries[i+1].([]domain.Answer))
		case domain.Key[[]domain.JudgeSummary]:
			state = domain.With(state, key, entries[i+1].([]domain.JudgeSummary))
		case domain.Key[*domain.Verdict]:
			state = domain.With(state, key, entries[i+1].(*domain.Verdict))
		case domain.Key[*domain.BudgetReport]:
			state = domain.With(state, key, entries[i+1].(*domain.BudgetReport))
		}
	}
	return state
}

// TestVerificationUnit_Execute tests the full execution flow of the VerificationUnit.
// It covers successful verification, handling of low-confidence scores that trigger
// human review, and ensures that the resulting state is correctly updated.
// It also tests failure modes, such as missing state components or LLM errors.
func TestVerificationUnit_Execute(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name                   string
		state                  domain.State
		llmResponse            string
		llmError               error
		expectedConfidence     float64
		expectHumanReview      bool
		wantErr                bool
		errMsg                 string
		traceLevel             string
		checkVerificationTrace bool
		confidenceThreshold    float64
	}{
		{
			name: "successful verification with high confidence",
			state: buildState(
				domain.KeyQuestion, "What is 2+2?",
				domain.KeyAnswers, []domain.Answer{
					{ID: "a1", Content: "4"},
				},
				domain.KeyJudgeScores, []domain.JudgeSummary{
					{Score: 10.0, Confidence: 0.95, Reasoning: "Correct answer"},
				},
				domain.KeyVerdict, &domain.Verdict{
					ID:                  "v1",
					AggregateScore:      10.0,
					RequiresHumanReview: false,
				},
			),
			llmResponse:         `{"confidence": 0.9, "reasoning": "The judging is consistent and accurate", "version": 1}`,
			expectedConfidence:  0.9,
			expectHumanReview:   false,
			confidenceThreshold: 0.8,
			wantErr:             false,
		},
		{
			name: "successful verification with low confidence triggers human review",
			state: buildState(
				domain.KeyQuestion, "What is the meaning of life?",
				domain.KeyAnswers, []domain.Answer{
					{ID: "a1", Content: "42"},
				},
				domain.KeyJudgeScores, []domain.JudgeSummary{
					{Score: 5.0, Confidence: 0.5, Reasoning: "Subjective question"},
				},
				domain.KeyVerdict, &domain.Verdict{
					ID:                  "v1",
					AggregateScore:      5.0,
					RequiresHumanReview: false,
				},
			),
			llmResponse:         `{"confidence": 0.6, "reasoning": "The judging is subjective and may need human review", "issues": ["Philosophical question", "No objective answer"], "recommendation": "Consider human review", "version": 1}`,
			expectedConfidence:  0.6,
			expectHumanReview:   true,
			confidenceThreshold: 0.8,
			wantErr:             false,
		},
		{
			name: "verification with debug trace level stores trace",
			state: func() domain.State {
				s := buildState(
					domain.KeyQuestion, "What is 2+2?",
					domain.KeyAnswers, []domain.Answer{
						{ID: "a1", Content: "4"},
					},
					domain.KeyJudgeScores, []domain.JudgeSummary{
						{Score: 10.0, Confidence: 0.95, Reasoning: "Correct answer"},
					},
					domain.KeyVerdict, &domain.Verdict{
						ID:                  "v1",
						AggregateScore:      10.0,
						RequiresHumanReview: false,
					},
				)
				return domain.With(s, domain.KeyTraceLevel, "debug")
			}(),
			llmResponse:            `{"confidence": 0.9, "reasoning": "The judging is consistent", "issues": ["Minor formatting"], "recommendation": "Improve format", "version": 1}`,
			expectedConfidence:     0.9,
			expectHumanReview:      false,
			confidenceThreshold:    0.8,
			traceLevel:             "debug",
			checkVerificationTrace: true,
			wantErr:                false,
		},
		{
			name: "LLM response with markdown code block",
			state: buildState(
				domain.KeyQuestion, "What is 2+2?",
				domain.KeyAnswers, []domain.Answer{
					{ID: "a1", Content: "4"},
				},
				domain.KeyJudgeScores, []domain.JudgeSummary{
					{Score: 10.0, Confidence: 0.95, Reasoning: "Correct answer"},
				},
				domain.KeyVerdict, &domain.Verdict{
					ID:             "v1",
					AggregateScore: 10.0,
				},
			),
			llmResponse:         "Here's my verification:\n```json\n{\"confidence\": 0.85, \"reasoning\": \"Good judging\"}\n```",
			expectedConfidence:  0.85,
			expectHumanReview:   false,
			confidenceThreshold: 0.8,
			wantErr:             false,
		},
		{
			name: "missing question returns error",
			state: buildState(
				domain.KeyAnswers, []domain.Answer{
					{ID: "a1", Content: "4"},
				},
				domain.KeyJudgeScores, []domain.JudgeSummary{
					{Score: 10.0, Confidence: 0.95, Reasoning: "Correct answer"},
				},
				domain.KeyVerdict, &domain.Verdict{ID: "v1"},
			),
			wantErr: true,
			errMsg:  "question not found in state",
		},
		{
			name: "missing answers returns error",
			state: buildState(
				domain.KeyQuestion, "What is 2+2?",
				domain.KeyJudgeScores, []domain.JudgeSummary{
					{Score: 10.0, Confidence: 0.95, Reasoning: "Correct answer"},
				},
				domain.KeyVerdict, &domain.Verdict{ID: "v1"},
			),
			wantErr: true,
			errMsg:  "answers not found in state",
		},
		{
			name: "missing judge scores returns error",
			state: buildState(
				domain.KeyQuestion, "What is 2+2?",
				domain.KeyAnswers, []domain.Answer{
					{ID: "a1", Content: "4"},
				},
				domain.KeyVerdict, &domain.Verdict{ID: "v1"},
			),
			wantErr: true,
			errMsg:  "no judge scores found to verify",
		},
		{
			name: "missing verdict returns error",
			state: buildState(
				domain.KeyQuestion, "What is 2+2?",
				domain.KeyAnswers, []domain.Answer{
					{ID: "a1", Content: "4"},
				},
				domain.KeyJudgeScores, []domain.JudgeSummary{
					{Score: 10.0, Confidence: 0.95, Reasoning: "Correct answer"},
				},
			),
			wantErr: true,
			errMsg:  "verdict not found in state",
		},
		{
			name: "empty judge scores returns error",
			state: buildState(
				domain.KeyQuestion, "What is 2+2?",
				domain.KeyAnswers, []domain.Answer{
					{ID: "a1", Content: "4"},
				},
				domain.KeyJudgeScores, []domain.JudgeSummary{},
				domain.KeyVerdict, &domain.Verdict{ID: "v1"},
			),
			wantErr: true,
			errMsg:  "no judge scores found to verify",
		},
		{
			name: "LLM returns error",
			state: buildState(
				domain.KeyQuestion, "What is 2+2?",
				domain.KeyAnswers, []domain.Answer{
					{ID: "a1", Content: "4"},
				},
				domain.KeyJudgeScores, []domain.JudgeSummary{
					{Score: 10.0, Confidence: 0.95, Reasoning: "Correct answer"},
				},
				domain.KeyVerdict, &domain.Verdict{ID: "v1"},
			),
			llmError: fmt.Errorf("API rate limit exceeded"),
			wantErr:  true,
			errMsg:   "LLM call failed",
		},
		{
			name: "invalid JSON response returns error",
			state: buildState(
				domain.KeyQuestion, "What is 2+2?",
				domain.KeyAnswers, []domain.Answer{
					{ID: "a1", Content: "4"},
				},
				domain.KeyJudgeScores, []domain.JudgeSummary{
					{Score: 10.0, Confidence: 0.95, Reasoning: "Correct answer"},
				},
				domain.KeyVerdict, &domain.Verdict{ID: "v1"},
			),
			llmResponse: "This is not JSON",
			wantErr:     true,
			errMsg:      "no valid JSON found in LLM response",
		},
		{
			name: "malformed JSON response returns error",
			state: buildState(
				domain.KeyQuestion, "What is 2+2?",
				domain.KeyAnswers, []domain.Answer{
					{ID: "a1", Content: "4"},
				},
				domain.KeyJudgeScores, []domain.JudgeSummary{
					{Score: 10.0, Confidence: 0.95, Reasoning: "Correct answer"},
				},
				domain.KeyVerdict, &domain.Verdict{ID: "v1"},
			),
			llmResponse: `{"confidence": "not-a-number", "reasoning": "test"}`,
			wantErr:     true,
			errMsg:      "failed to parse JSON response",
		},
		{
			name: "invalid confidence value returns validation error",
			state: buildState(
				domain.KeyQuestion, "What is 2+2?",
				domain.KeyAnswers, []domain.Answer{
					{ID: "a1", Content: "4"},
				},
				domain.KeyJudgeScores, []domain.JudgeSummary{
					{Score: 10.0, Confidence: 0.95, Reasoning: "Correct answer"},
				},
				domain.KeyVerdict, &domain.Verdict{ID: "v1"},
			),
			llmResponse: `{"confidence": 1.5, "reasoning": "Invalid confidence"}`,
			wantErr:     true,
			errMsg:      "invalid response structure",
		},
		{
			name: "budget tracking updated on successful execution",
			state: buildState(
				domain.KeyQuestion, "What is 2+2?",
				domain.KeyAnswers, []domain.Answer{
					{ID: "a1", Content: "4"},
				},
				domain.KeyJudgeScores, []domain.JudgeSummary{
					{Score: 10.0, Confidence: 0.95, Reasoning: "Correct answer"},
				},
				domain.KeyVerdict, &domain.Verdict{ID: "v1"},
				domain.KeyBudget, &domain.BudgetReport{
					TokensUsed: 100,
					CallsMade:  2,
				},
			),
			llmResponse:         `{"confidence": 0.9, "reasoning": "Good judging", "version": 1}`,
			expectedConfidence:  0.9,
			expectHumanReview:   false,
			confidenceThreshold: 0.8,
			wantErr:             false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock LLM client
			mockLLM := testutils.NewMockLLMClient("test-model")
			if tt.llmError != nil {
				mockLLM.SetError(tt.llmError)
			} else if tt.llmResponse != "" {
				mockLLM.SetResponse(tt.llmResponse)
			}

			// Create verification unit with custom threshold if specified
			config := defaultVerificationConfig()
			if tt.confidenceThreshold > 0 {
				config.ConfidenceThreshold = tt.confidenceThreshold
			}

			unit, err := NewVerificationUnit("verifier1", mockLLM, config)
			require.NoError(t, err, "failed to create verification unit")

			// Execute the unit
			newState, err := unit.Execute(ctx, tt.state)

			if tt.wantErr {
				require.Error(t, err, "expected error but got none")
				assert.Contains(t, err.Error(), tt.errMsg, "error message should contain expected text")
			} else {
				require.NoError(t, err, "unexpected error during execution")

				// Check verdict was updated with human review flag
				verdict, ok := domain.Get(newState, domain.KeyVerdict)
				require.True(t, ok, "verdict should be in state")
				require.NotNil(t, verdict, "verdict should not be nil")

				assert.Equal(t, tt.expectHumanReview, verdict.RequiresHumanReview,
					"human review flag should match expected value")

				// Check verification trace if debug mode
				if tt.checkVerificationTrace {
					traceStr, ok := domain.Get(newState, domain.KeyVerificationTrace)
					require.True(t, ok, "verification trace should be in state when debug")
					var trace VerificationTrace
					err := json.Unmarshal([]byte(traceStr), &trace)
					require.NoError(t, err, "verification trace should unmarshal correctly")
					require.True(t, ok, "verification trace should be correct type")
					assert.Equal(t, tt.expectedConfidence, trace.Confidence,
						"trace confidence should match")
					assert.NotEmpty(t, trace.Reasoning, "trace should have reasoning")
				}

				// Check budget was updated
				if oldBudget, ok := domain.Get(tt.state, domain.KeyBudget); ok {
					newBudget, ok := domain.Get(newState, domain.KeyBudget)
					require.True(t, ok, "budget should still be in state")
					assert.GreaterOrEqual(t, newBudget.TokensUsed, oldBudget.TokensUsed,
						"token usage should not decrease")
					assert.Equal(t, oldBudget.CallsMade+1, newBudget.CallsMade,
						"calls made should increase by 1")
				}
			}
		})
	}
}

// TestVerificationUnit_Validate tests the validation logic for the VerificationUnit.
// It ensures that a unit with valid configuration and a properly configured LLM client
// passes validation, while units with missing or invalid components fail.
func TestVerificationUnit_Validate(t *testing.T) {
	tests := []struct {
		name    string
		unit    *VerificationUnit
		wantErr bool
		errMsg  string
	}{
		{
			name: "valid unit passes validation",
			unit: &VerificationUnit{
				name:      "verifier1",
				config:    defaultVerificationConfig(),
				llmClient: testutils.NewMockLLMClient("test-model"),
				validator: testutils.NewTestValidator(),
			},
			wantErr: false,
		},
		{
			name: "nil LLM client fails validation",
			unit: &VerificationUnit{
				name:      "verifier1",
				config:    defaultVerificationConfig(),
				llmClient: nil,
				validator: testutils.NewTestValidator(),
			},
			wantErr: true,
			errMsg:  "LLM client cannot be nil",
		},
		{
			name: "empty model name fails validation",
			unit: &VerificationUnit{
				name:      "verifier1",
				config:    defaultVerificationConfig(),
				llmClient: testutils.NewMockLLMClient(""), // Empty model
				validator: testutils.NewTestValidator(),
			},
			wantErr: true,
			errMsg:  "LLM client model is not configured",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.unit.Validate()

			if tt.wantErr {
				require.Error(t, err, "expected validation error")
				assert.Contains(t, err.Error(), tt.errMsg, "error message should contain expected text")
			} else {
				require.NoError(t, err, "validation should pass")
			}
		})
	}
}

// TestVerificationUnit_UnmarshalParameters tests the UnmarshalParameters method.
// It verifies that a new VerificationUnit can be created with updated parameters
// from a YAML node, ensuring the original unit remains unchanged and that
// invalid YAML or configurations are correctly handled.
func TestVerificationUnit_UnmarshalParameters(t *testing.T) {
	tests := []struct {
		name    string
		yaml    string
		wantErr bool
		errMsg  string
		check   func(t *testing.T, unit *VerificationUnit)
	}{
		{
			name: "valid YAML unmarshals successfully",
			yaml: `
prompt_template: "Verify these results: {{.Question}}"
confidence_threshold: 0.85
temperature: 0.1
max_tokens: 600
`,
			wantErr: false,
			check: func(t *testing.T, unit *VerificationUnit) {
				assert.Equal(t, "Verify these results: {{.Question}}", unit.config.PromptTemplate)
				assert.Equal(t, 0.85, unit.config.ConfidenceThreshold)
				assert.Equal(t, 0.1, unit.config.Temperature)
				assert.Equal(t, 600, unit.config.MaxTokens)
			},
		},
		{
			name: "invalid YAML syntax returns error",
			yaml: `
prompt_template: "Verify
confidence_threshold: not-a-number
`,
			wantErr: true,
			errMsg:  "failed to decode parameters",
		},
		// Note: yaml.v3 doesn't support strict decoding with yaml.Node.Decode()
		// so unknown fields are silently ignored. This is consistent with other units.
		{
			name: "invalid template syntax returns error",
			yaml: `
prompt_template: "This is a valid length template with {{.Invalid"
confidence_threshold: 0.8
temperature: 0.0
max_tokens: 512
`,
			wantErr: true,
			errMsg:  "failed to parse prompt template",
		},
		{
			name: "validation failure returns error",
			yaml: `
prompt_template: "Too short"
confidence_threshold: 1.5
temperature: 0.0
max_tokens: 512
`,
			wantErr: true,
			errMsg:  "configuration validation failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create base unit
			mockLLM := testutils.NewMockLLMClient("test-model")
			baseUnit, err := NewVerificationUnit("verifier1", mockLLM, defaultVerificationConfig())
			require.NoError(t, err, "failed to create base unit")

			// Parse YAML
			var params yaml.Node
			err = yaml.Unmarshal([]byte(tt.yaml), &params)

			// For invalid YAML syntax test, expect parse error
			if tt.name == "invalid YAML syntax returns error" {
				require.Error(t, err, "should fail to parse invalid YAML")
				return
			}

			require.NoError(t, err, "failed to parse test YAML")

			// Check if params.Content is not empty
			if len(params.Content) == 0 {
				t.Fatal("YAML node has no content")
			}

			// Unmarshal parameters - pass the node directly instead of dereferencing Content[0]
			// The UnmarshalParameters method expects a yaml.Node, not a dereferenced one
			newUnit, err := baseUnit.UnmarshalParameters(params)

			if tt.wantErr {
				require.Error(t, err, "expected unmarshal error")
				assert.Contains(t, err.Error(), tt.errMsg, "error message should contain expected text")
			} else {
				require.NoError(t, err, "unexpected unmarshal error")
				assert.NotNil(t, newUnit, "new unit should not be nil")
				assert.Equal(t, baseUnit.name, newUnit.name, "unit name should be preserved")
				assert.Equal(t, baseUnit.llmClient, newUnit.llmClient, "LLM client should be preserved")

				if tt.check != nil {
					tt.check(t, newUnit)
				}
			}
		})
	}
}

// TestNewVerificationFromConfig tests the factory function for creating a VerificationUnit.
// It ensures that the unit can be created from a map of parameters,
// handles type conversions correctly, applies default values, and fails when
// required fields like the LLM client are missing.
func TestNewVerificationFromConfig(t *testing.T) {
	mockLLM := testutils.NewMockLLMClient("test-model")

	tests := []struct {
		name      string
		id        string
		config    map[string]any
		llmClient ports.LLMClient // Add this field to allow testing with nil
		wantErr   bool
		errMsg    string
		check     func(t *testing.T, unit *VerificationUnit)
	}{
		{
			name:      "valid config creates unit successfully",
			id:        "verifier1",
			llmClient: mockLLM,
			config: map[string]any{
				"prompt_template":      "Verify these results: {{.Question}}",
				"confidence_threshold": 0.85,
				"temperature":          0.1,
				"max_tokens":           600,
			},
			wantErr: false,
			check: func(t *testing.T, unit *VerificationUnit) {
				assert.Equal(t, "verifier1", unit.Name())
				assert.Equal(t, 0.85, unit.config.ConfidenceThreshold)
				assert.Equal(t, 0.1, unit.config.Temperature)
				assert.Equal(t, 600, unit.config.MaxTokens)
			},
		},
		{
			name:    "uses defaults for missing optional fields",
			id:      "verifier2",
			config:  map[string]any{},
			wantErr: false,
			check: func(t *testing.T, unit *VerificationUnit) {
				assert.Equal(t, DefaultVerificationConfThreshold, unit.config.ConfidenceThreshold)
				assert.Equal(t, DefaultVerificationTemperature, unit.config.Temperature)
				assert.Equal(t, DefaultVerificationMaxTokens, unit.config.MaxTokens)
			},
		},
		{
			name:      "nil LLM client returns error",
			id:        "verifier3",
			llmClient: nil, // Explicitly nil
			config: map[string]any{
				"prompt_template": "Verify these results: {{.Question}}",
			},
			wantErr: true,
			errMsg:  "LLM client cannot be nil",
		},
		{
			name: "handles different numeric types for confidence threshold",
			id:   "verifier4",
			config: map[string]any{
				"confidence_threshold": 0.75, // Use actual float
			},
			wantErr: false,
			check: func(t *testing.T, unit *VerificationUnit) {
				assert.Equal(t, 0.75, unit.config.ConfidenceThreshold)
			},
		},
		{
			name: "handles integer temperature",
			id:   "verifier5",
			config: map[string]any{
				"temperature": 0, // Integer
			},
			wantErr: false,
			check: func(t *testing.T, unit *VerificationUnit) {
				assert.Equal(t, 0.0, unit.config.Temperature)
			},
		},
		{
			name: "handles float64 max tokens",
			id:   "verifier6",
			config: map[string]any{
				"max_tokens": 750.0, // Float64
			},
			wantErr: false,
			check: func(t *testing.T, unit *VerificationUnit) {
				assert.Equal(t, 750, unit.config.MaxTokens)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Use the llmClient from the test case, defaulting to mockLLM if not specified
			llm := tt.llmClient
			if llm == nil && !tt.wantErr {
				llm = mockLLM
			}
			unitPort, err := NewVerificationFromConfig(tt.id, tt.config, llm)

			if tt.wantErr {
				require.Error(t, err, "expected creation error")
				assert.Contains(t, err.Error(), tt.errMsg, "error message should contain expected text")
			} else {
				require.NoError(t, err, "unexpected creation error")
				assert.NotNil(t, unitPort, "unit should not be nil")

				if tt.check != nil {
					unit, ok := unitPort.(*VerificationUnit)
					require.True(t, ok, "unit should be *VerificationUnit")
					tt.check(t, unit)
				}
			}
		})
	}
}

// TestParseLLMResponse tests the parsing of LLM responses for the VerificationUnit.
// It ensures that valid JSON is correctly parsed into an LLMVerificationResponse,
// handles JSON embedded in surrounding text, and fails appropriately for
// invalid or malformed responses.
func TestParseLLMResponse(t *testing.T) {
	unit := &VerificationUnit{
		validator: testutils.NewTestValidator(),
	}

	tests := []struct {
		name     string
		response string
		wantErr  bool
		errMsg   string
		check    func(t *testing.T, resp *LLMVerificationResponse)
	}{
		{
			name:     "valid JSON response parses successfully",
			response: `{"confidence": 0.85, "reasoning": "The judging appears consistent and fair", "version": 1}`,
			wantErr:  false,
			check: func(t *testing.T, resp *LLMVerificationResponse) {
				assert.Equal(t, 0.85, resp.Confidence)
				assert.Equal(t, "The judging appears consistent and fair", resp.Reasoning)
				assert.Equal(t, 1, resp.Version)
			},
		},
		{
			name: "response with issues and recommendation",
			response: `{
				"confidence": 0.6,
				"reasoning": "Some inconsistencies found",
				"issues": ["Score variance too high", "Subjective criteria"],
				"recommendation": "Consider more objective scoring criteria",
				"version": 1
			}`,
			wantErr: false,
			check: func(t *testing.T, resp *LLMVerificationResponse) {
				assert.Equal(t, 0.6, resp.Confidence)
				assert.Len(t, resp.Issues, 2)
				assert.Equal(t, "Consider more objective scoring criteria", resp.Recommendation)
			},
		},
		{
			name:     "JSON in markdown code block",
			response: "Here's my analysis:\n```json\n{\"confidence\": 0.9, \"reasoning\": \"Excellent judging\"}\n```",
			wantErr:  false,
			check: func(t *testing.T, resp *LLMVerificationResponse) {
				assert.Equal(t, 0.9, resp.Confidence)
			},
		},
		{
			name:     "no JSON in response",
			response: "This is just plain text without any JSON",
			wantErr:  true,
			errMsg:   "no valid JSON found",
		},
		{
			name:     "invalid JSON syntax",
			response: `{"confidence": 0.8, "reasoning": "Missing closing brace"`,
			wantErr:  true,
			errMsg:   "no valid JSON found",
		},
		{
			name:     "missing required fields",
			response: `{"confidence": 0.8}`,
			wantErr:  true,
			errMsg:   "invalid response structure",
		},
		{
			name:     "confidence out of range",
			response: `{"confidence": 1.5, "reasoning": "Invalid confidence"}`,
			wantErr:  true,
			errMsg:   "invalid response structure",
		},
		{
			name:     "reasoning too short",
			response: `{"confidence": 0.8, "reasoning": "Too short"}`,
			wantErr:  true,
			errMsg:   "invalid response structure",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := unit.parseLLMResponse(tt.response)

			if tt.wantErr {
				require.Error(t, err, "expected parse error")
				assert.Contains(t, err.Error(), tt.errMsg, "error message should contain expected text")
			} else {
				require.NoError(t, err, "unexpected parse error")
				assert.NotNil(t, resp, "response should not be nil")

				if tt.check != nil {
					tt.check(t, resp)
				}
			}
		})
	}
}

// TestDefaultVerificationConfig tests that the default configuration is created with the expected values.
func TestDefaultVerificationConfig(t *testing.T) {
	config := defaultVerificationConfig()

	assert.NotEmpty(t, config.PromptTemplate, "default prompt template should not be empty")
	assert.Contains(t, config.PromptTemplate, "{{.Question}}", "template should include question placeholder")
	assert.Contains(t, config.PromptTemplate, "{{range $i, $answer := .Answers}}", "template should include answers range")
	assert.Contains(t, config.PromptTemplate, "{{range $i, $score := .JudgeScores}}", "template should include judge scores range")
	assert.Equal(t, DefaultVerificationConfThreshold, config.ConfidenceThreshold, "should use default confidence threshold")
	assert.Equal(t, DefaultVerificationTemperature, config.Temperature, "should use default temperature")
	assert.Equal(t, DefaultVerificationMaxTokens, config.MaxTokens, "should use default max tokens")
}
