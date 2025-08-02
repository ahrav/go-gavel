// Package units provides domain-specific evaluation units that implement
// the ports.Unit interface for the go-gavel evaluation engine.
// Each unit encapsulates a specific evaluation capability such as answer
// generation, scoring, or validation.
package units

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"strconv"
	"text/template"
	"time"

	"github.com/go-playground/validator/v10"
	"golang.org/x/sync/errgroup"
	"gopkg.in/yaml.v3"

	"github.com/ahrav/go-gavel/internal/domain"
	"github.com/ahrav/go-gavel/internal/ports"
)

var _ ports.Unit = (*AnswererUnit)(nil)

// Shared validator instance to reduce allocations across unit instances.
var answererValidator = validator.New()

// Configuration constants
const (
	// DefaultMaxConcurrency is the default number of concurrent LLM calls.
	DefaultMaxConcurrency = 5
	// DefaultNumAnswers is the default number of candidate answers to generate.
	DefaultNumAnswers = 3
	// DefaultMaxTokens is the default maximum tokens per generated answer.
	DefaultMaxTokens = 500
	// DefaultTemperature is the default LLM temperature for answer generation.
	DefaultTemperature = 0.7
	// DefaultTimeoutSeconds is the default timeout for LLM calls in seconds.
	DefaultTimeoutSeconds = 30
)

// Sentinel errors for better error handling and testing.
var (
	ErrQuestionMissing   = errors.New("question not found in state")
	ErrQuestionEmpty     = errors.New("question cannot be empty")
	ErrUnitNameEmpty     = errors.New("unit name cannot be empty")
	ErrLLMClientNil      = errors.New("LLM client cannot be nil")
	ErrConfigValidation  = errors.New("configuration validation failed")
	ErrTemplateExecution = errors.New("failed to execute prompt template")
	ErrLLMCallFailed     = errors.New("LLM call failed")
)

// LLMOptions contains typed options for LLM client calls to improve type safety.
type LLMOptions struct {
	Temperature float64 `json:"temperature"`
	MaxTokens   int     `json:"max_tokens"`
}

// AnswererUnit generates candidate answers by calling an LLM client with
// the provided question prompt.
// It expects a question in state and produces a list of Answer objects.
// The unit is stateless and thread-safe for concurrent execution.
type AnswererUnit struct {
	// name is the unique identifier for this unit instance.
	name string
	// config contains the validated configuration parameters.
	config AnswererConfig
	// llmClient provides access to the LLM for answer generation.
	llmClient ports.LLMClient
	// promptTemplate is the compiled template for safe prompt generation.
	promptTemplate *template.Template
}

// AnswererConfig defines the configuration parameters for the AnswererUnit.
// All fields are validated during unit creation and parameter unmarshaling.
type AnswererConfig struct {
	// NumAnswers specifies how many candidate answers to generate.
	// Must be between 1 and 10 for reasonable performance.
	NumAnswers int `yaml:"num_answers" json:"num_answers" validate:"required,min=1,max=10"`

	// Prompt is the Go template used to generate answers from the question.
	// Should use {{.Question}} placeholder for safe question substitution.
	// Example: "Please answer this question: {{.Question}}"
	Prompt string `yaml:"prompt" json:"prompt" validate:"required,min=10"`

	// Temperature controls randomness in LLM generation (0.0-1.0).
	// Lower values produce more deterministic responses.
	Temperature float64 `yaml:"temperature" json:"temperature" validate:"min=0.0,max=1.0"`

	// MaxTokens limits the length of each generated answer.
	// Should be set based on expected answer complexity.
	MaxTokens int `yaml:"max_tokens" json:"max_tokens" validate:"required,min=10,max=16000"`

	// Timeout specifies the maximum duration for LLM calls.
	// Prevents hanging on slow or unresponsive LLM services.
	Timeout time.Duration `yaml:"timeout" json:"timeout" validate:"required,min=1s,max=300s"`

	// MaxConcurrency limits the number of concurrent LLM calls.
	// Prevents overwhelming the LLM service with too many simultaneous requests.
	MaxConcurrency int `yaml:"max_concurrency" json:"max_concurrency" validate:"required,min=1,max=20"`
}

// NewAnswererUnit creates a new AnswererUnit with the specified configuration
// and dependencies.
// The unit validates its configuration and ensures the LLM client is available.
// Returns an error if configuration validation fails or dependencies are missing.
func NewAnswererUnit(
	name string,
	llmClient ports.LLMClient,
	config AnswererConfig,
) (*AnswererUnit, error) {
	if name == "" {
		return nil, ErrUnitNameEmpty
	}
	if llmClient == nil {
		return nil, ErrLLMClientNil
	}

	if err := answererValidator.Struct(config); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrConfigValidation, err)
	}

	// Compile the prompt template to prevent injection attacks.
	tmpl, err := template.New("prompt").Parse(config.Prompt)
	if err != nil {
		return nil, fmt.Errorf("failed to parse prompt template: %w", err)
	}

	return &AnswererUnit{
		name:           name,
		config:         config,
		llmClient:      llmClient,
		promptTemplate: tmpl,
	}, nil
}

// Name returns the unique identifier for this unit instance.
// The name is used for logging, debugging, and graph node referencing.
func (au *AnswererUnit) Name() string { return au.name }

// Execute generates candidate answers by calling the LLM client multiple times
// with the question from state.
// It retrieves the question using KeyQuestion, generates the specified number
// of answers, and stores them in state using KeyAnswers.
// Returns updated state with generated answers or an error if generation fails.
func (au *AnswererUnit) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	question, ok := domain.Get(state, domain.KeyQuestion)
	if !ok {
		return state, ErrQuestionMissing
	}

	if question == "" {
		return state, ErrQuestionEmpty
	}

	ctx, cancel := context.WithTimeout(ctx, au.config.Timeout)
	defer cancel()

	// Pre-compile prompt to avoid repeated template execution.
	var promptBuf bytes.Buffer
	if err := au.promptTemplate.Execute(&promptBuf, struct{ Question string }{Question: question}); err != nil {
		return state, fmt.Errorf("%w: %v", ErrTemplateExecution, err)
	}
	prompt := promptBuf.String()

	llmOpts := LLMOptions{
		Temperature: au.config.Temperature,
		MaxTokens:   au.config.MaxTokens,
	}

	options := map[string]any{
		"temperature": llmOpts.Temperature,
		"max_tokens":  llmOpts.MaxTokens,
	}

	answers := make([]domain.Answer, au.config.NumAnswers)
	g, ctx := errgroup.WithContext(ctx)

	// Limit concurrent LLM calls to avoid overwhelming the service.
	// Use configured max concurrency or default if not set.
	maxConcurrency := au.config.MaxConcurrency
	if maxConcurrency <= 0 {
		maxConcurrency = DefaultMaxConcurrency // Fallback to reasonable default
	}
	g.SetLimit(maxConcurrency)

	for idx := 0; idx < au.config.NumAnswers; idx++ {
		g.Go(func() error {
			response, err := au.llmClient.Complete(ctx, prompt, options)
			if err != nil {
				return fmt.Errorf("%w for answer %d: %v", ErrLLMCallFailed, idx+1, err)
			}

			answers[idx] = domain.Answer{
				ID:      fmt.Sprintf("%s_answer_%d", au.name, idx+1),
				Content: response,
			}

			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return state, au.aggregateErrors(err, "answer generation")
	}

	return domain.With(state, domain.KeyAnswers, answers), nil
}

// Validate checks if the unit is properly configured and ready for execution.
// It validates the configuration parameters and verifies LLM client availability.
// Returns nil if validation passes, or an error describing what is invalid.
func (au *AnswererUnit) Validate() error {
	if au.llmClient == nil {
		return fmt.Errorf("LLM client is not configured")
	}

	if err := answererValidator.Struct(au.config); err != nil {
		return fmt.Errorf("configuration validation failed: %w", err)
	}

	// Verify LLM client is functional by checking model.
	model := au.llmClient.GetModel()
	if model == "" {
		return fmt.Errorf("LLM client model is not configured")
	}

	return nil
}

// aggregateErrors provides enhanced error context for concurrent operations.
// It wraps the error with additional context about the operation that failed,
// making it easier to diagnose issues in production.
func (au *AnswererUnit) aggregateErrors(err error, operation string) error {
	if err == nil {
		return nil
	}

	if errors.Is(err, ErrLLMCallFailed) {
		return err
	}

	return fmt.Errorf("unit %s: %s failed: %w", au.name, operation, err)
}

// UnmarshalParameters deserializes YAML configuration parameters and returns
// a new AnswererUnit instance with the updated configuration.
// This method maintains thread-safety by not mutating the existing unit.
// Returns a new unit instance or an error if YAML parsing fails or validation fails.
func (au *AnswererUnit) UnmarshalParameters(params yaml.Node) (*AnswererUnit, error) {
	var config AnswererConfig

	// Use strict decoding to catch unknown fields.
	if err := params.Decode(&config); err != nil {
		return nil, fmt.Errorf("failed to decode parameters: %w", err)
	}

	// Validate the decoded configuration.
	if err := answererValidator.Struct(config); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrConfigValidation, err)
	}

	// Compile the prompt template to prevent injection attacks.
	tmpl, err := template.New("prompt").Parse(config.Prompt)
	if err != nil {
		return nil, fmt.Errorf("failed to parse prompt template: %w", err)
	}

	// Return a new instance with the updated configuration to maintain thread safety.
	return &AnswererUnit{
		name:           au.name,
		config:         config,
		llmClient:      au.llmClient,
		promptTemplate: tmpl,
	}, nil
}

// defaultAnswererConfig returns an AnswererConfig with sensible defaults.
// This ensures consistent behavior when configuration values are missing.
func defaultAnswererConfig() AnswererConfig {
	return AnswererConfig{
		NumAnswers:     DefaultNumAnswers,
		Prompt:         "Please provide a comprehensive answer to: {{.Question}}",
		Temperature:    DefaultTemperature,
		MaxTokens:      DefaultMaxTokens,
		Timeout:        DefaultTimeoutSeconds * time.Second,
		MaxConcurrency: DefaultMaxConcurrency,
	}
}

// CreateAnswererUnit is a factory function that creates an AnswererUnit
// from a configuration map, following the UnitFactory pattern.
// This function is used by the UnitRegistry for dynamic unit creation.
func CreateAnswererUnit(id string, config map[string]any) (*AnswererUnit, error) {
	// Extract LLM client from config.
	llmClient, ok := config["llm_client"].(ports.LLMClient)
	if !ok {
		return nil, fmt.Errorf("llm_client is required and must implement ports.LLMClient")
	}

	// Check if this looks like an incomplete configuration that should fail.
	// Only fail if user provided some config but missed ALL the critical operational fields.
	_, hasPrompt := config["prompt"]
	_, hasMaxTokens := config["max_tokens"]
	_, hasTimeout := config["timeout"]
	_, hasMaxConcurrency := config["max_concurrency"]

	// Count non-llm_client config keys
	nonClientConfigCount := 0
	for key := range config {
		if key != "llm_client" {
			nonClientConfigCount++
		}
	}

	// If user provided some config but none of the operational fields, it's likely incomplete.
	if nonClientConfigCount > 0 && !hasPrompt && !hasMaxTokens && !hasTimeout && !hasMaxConcurrency {
		return nil, fmt.Errorf("failed to create unit: incomplete configuration missing operational fields")
	}

	// Start with sensible defaults and merge user-provided values.
	answererConfig := defaultAnswererConfig()

	if numAnswers, ok := config["num_answers"]; ok {
		if val, ok := numAnswers.(int); ok {
			answererConfig.NumAnswers = val
		} else if val, ok := numAnswers.(float64); ok {
			// YAML numbers often come as float64, convert to int
			answererConfig.NumAnswers = int(val)
		} else if val, ok := numAnswers.(string); ok {
			if parsed, err := strconv.Atoi(val); err == nil {
				answererConfig.NumAnswers = parsed
			}
		}
	}

	if prompt, ok := config["prompt"].(string); ok {
		answererConfig.Prompt = prompt
	}

	if temperature, ok := config["temperature"]; ok {
		if val, ok := temperature.(float64); ok {
			answererConfig.Temperature = val
		} else if val, ok := temperature.(int); ok {
			// Handle integer temperature values (e.g., 0, 1)
			answererConfig.Temperature = float64(val)
		} else if val, ok := temperature.(string); ok {
			parsed, err := strconv.ParseFloat(val, 64)
			if err != nil {
				return nil, fmt.Errorf("failed to unmarshal answerer config: invalid temperature value '%v': %w", val, err)
			}
			answererConfig.Temperature = parsed
		} else {
			return nil, fmt.Errorf("failed to unmarshal answerer config: temperature must be a number, got %T", temperature)
		}
	}

	if maxTokens, ok := config["max_tokens"]; ok {
		if val, ok := maxTokens.(int); ok {
			answererConfig.MaxTokens = val
		} else if val, ok := maxTokens.(float64); ok {
			// YAML numbers often come as float64, convert to int
			answererConfig.MaxTokens = int(val)
		} else if val, ok := maxTokens.(string); ok {
			if parsed, err := strconv.Atoi(val); err == nil {
				answererConfig.MaxTokens = parsed
			}
		}
	}

	if timeout, ok := config["timeout"]; ok {
		if val, ok := timeout.(time.Duration); ok {
			answererConfig.Timeout = val
		} else if val, ok := timeout.(string); ok {
			parsed, err := time.ParseDuration(val)
			if err != nil {
				return nil, fmt.Errorf("failed to unmarshal answerer config: invalid timeout duration '%v': %w", val, err)
			}
			answererConfig.Timeout = parsed
		} else if val, ok := timeout.(float64); ok {
			// Handle as seconds if it's a number
			answererConfig.Timeout = time.Duration(val) * time.Second
		} else {
			return nil, fmt.Errorf("failed to unmarshal answerer config: timeout must be a duration string or number, got %T", timeout)
		}
	}

	if maxConcurrency, ok := config["max_concurrency"]; ok {
		if val, ok := maxConcurrency.(int); ok {
			answererConfig.MaxConcurrency = val
		} else if val, ok := maxConcurrency.(float64); ok {
			// YAML numbers often come as float64, convert to int
			answererConfig.MaxConcurrency = int(val)
		} else if val, ok := maxConcurrency.(string); ok {
			if parsed, err := strconv.Atoi(val); err == nil {
				answererConfig.MaxConcurrency = parsed
			}
		}
	}

	return NewAnswererUnit(id, llmClient, answererConfig)
}
