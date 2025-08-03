// Package units provides domain-specific evaluation units that implement
// the ports.Unit interface for the go-gavel evaluation engine.
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

// Shared validator instance to reduce allocations.
var answererValidator = validator.New()

// Configuration constants for the AnswererUnit.
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

// Sentinel errors for clear, testable error conditions.
var (
	ErrQuestionMissing   = errors.New("question not found in state")
	ErrQuestionEmpty     = errors.New("question cannot be empty")
	ErrUnitNameEmpty     = errors.New("unit name cannot be empty")
	ErrLLMClientNil      = errors.New("LLM client cannot be nil")
	ErrConfigValidation  = errors.New("configuration validation failed")
	ErrTemplateExecution = errors.New("failed to execute prompt template")
	ErrLLMCallFailed     = errors.New("LLM call failed")
)

// LLMOptions contains typed options for LLM client calls.
type LLMOptions struct {
	Temperature float64 `json:"temperature"`
	MaxTokens   int     `json:"max_tokens"`
}

// AnswererUnit generates candidate answers by calling an LLM client with a
// given question. It expects a question in the state and produces a list of
// Answer objects. The unit is stateless and thread-safe.
type AnswererUnit struct {
	name           string
	config         AnswererConfig
	llmClient      ports.LLMClient
	promptTemplate *template.Template
}

// AnswererConfig defines the configuration parameters for the AnswererUnit.
// All fields are validated during unit creation and parameter unmarshaling.
type AnswererConfig struct {
	// NumAnswers specifies how many candidate answers to generate.
	NumAnswers int `yaml:"num_answers" json:"num_answers" validate:"required,min=1,max=10"`

	// Prompt is the Go template used to generate answers from the question.
	// It should use the {{.Question}} placeholder for safe substitution.
	Prompt string `yaml:"prompt" json:"prompt" validate:"required,min=10"`

	// Temperature controls randomness in LLM generation (0.0-1.0).
	Temperature float64 `yaml:"temperature" json:"temperature" validate:"min=0.0,max=1.0"`

	// MaxTokens limits the length of each generated answer.
	MaxTokens int `yaml:"max_tokens" json:"max_tokens" validate:"required,min=10,max=16000"`

	// Timeout specifies the maximum duration for each LLM call.
	Timeout time.Duration `yaml:"timeout" json:"timeout" validate:"required,min=1s,max=300s"`

	// MaxConcurrency limits the number of concurrent LLM calls to avoid
	// overwhelming the service.
	MaxConcurrency int `yaml:"max_concurrency" json:"max_concurrency" validate:"required,min=1,max=20"`
}

// NewAnswererUnit creates a new AnswererUnit with the specified configuration
// and dependencies. It returns an error if the configuration is invalid or
// dependencies are missing.
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
func (au *AnswererUnit) Name() string { return au.name }

// Execute generates candidate answers by calling the LLM client concurrently.
// It retrieves the question from the state, generates the specified number of
// answers, and stores them back into the state.
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

	var promptBuf bytes.Buffer
	if err := au.promptTemplate.Execute(&promptBuf, struct{ Question string }{Question: question}); err != nil {
		return state, fmt.Errorf("%w: %v", ErrTemplateExecution, err)
	}
	prompt := promptBuf.String()

	options := map[string]any{
		"temperature": au.config.Temperature,
		"max_tokens":  au.config.MaxTokens,
	}

	answers := make([]domain.Answer, au.config.NumAnswers)
	g, ctx := errgroup.WithContext(ctx)
	g.SetLimit(au.config.MaxConcurrency)

	for i := 0; i < au.config.NumAnswers; i++ {
		idx := i
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
func (au *AnswererUnit) Validate() error {
	if au.llmClient == nil {
		return fmt.Errorf("LLM client is not configured")
	}
	if err := answererValidator.Struct(au.config); err != nil {
		return fmt.Errorf("configuration validation failed: %w", err)
	}
	if model := au.llmClient.GetModel(); model == "" {
		return fmt.Errorf("LLM client model is not configured")
	}
	return nil
}

// aggregateErrors provides enhanced error context for concurrent operations.
func (au *AnswererUnit) aggregateErrors(err error, operation string) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, ErrLLMCallFailed) {
		return err
	}
	return fmt.Errorf("unit %s: %s failed: %w", au.name, operation, err)
}

// UnmarshalParameters deserializes YAML parameters and returns a new, updated
// AnswererUnit instance to maintain thread-safety.
func (au *AnswererUnit) UnmarshalParameters(params yaml.Node) (*AnswererUnit, error) {
	var config AnswererConfig
	if err := params.Decode(&config); err != nil {
		return nil, fmt.Errorf("failed to decode parameters: %w", err)
	}

	if err := answererValidator.Struct(config); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrConfigValidation, err)
	}

	tmpl, err := template.New("prompt").Parse(config.Prompt)
	if err != nil {
		return nil, fmt.Errorf("failed to parse prompt template: %w", err)
	}

	return &AnswererUnit{
		name:           au.name,
		config:         config,
		llmClient:      au.llmClient,
		promptTemplate: tmpl,
	}, nil
}

// defaultAnswererConfig returns an AnswererConfig with sensible defaults.
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
// from a configuration map, for use with the UnitRegistry.
func CreateAnswererUnit(id string, config map[string]any) (*AnswererUnit, error) {
	llmClient, ok := config["llm_client"].(ports.LLMClient)
	if !ok {
		return nil, fmt.Errorf("llm_client is required and must implement ports.LLMClient")
	}

	// Check for required fields first
	requiredFields := []string{"num_answers", "prompt", "max_tokens", "timeout", "max_concurrency"}
	for _, field := range requiredFields {
		if _, ok := config[field]; !ok {
			return nil, fmt.Errorf("failed to unmarshal answerer config: missing required field '%s'", field)
		}
	}

	answererConfig := defaultAnswererConfig()

	// Parse num_answers
	if val, ok := config["num_answers"]; ok {
		if i, err := strconv.Atoi(fmt.Sprintf("%v", val)); err != nil {
			return nil, fmt.Errorf("failed to unmarshal answerer config: invalid num_answers format")
		} else {
			answererConfig.NumAnswers = i
		}
	}

	// Parse prompt
	if val, ok := config["prompt"].(string); ok {
		answererConfig.Prompt = val
	} else {
		return nil, fmt.Errorf("failed to unmarshal answerer config: prompt must be a string")
	}

	// Parse temperature (optional field)
	if val, ok := config["temperature"]; ok {
		if f, err := strconv.ParseFloat(fmt.Sprintf("%v", val), 64); err != nil {
			return nil, fmt.Errorf("failed to unmarshal answerer config: invalid temperature format")
		} else {
			answererConfig.Temperature = f
		}
	}

	// Parse max_tokens
	if val, ok := config["max_tokens"]; ok {
		if i, err := strconv.Atoi(fmt.Sprintf("%v", val)); err != nil {
			return nil, fmt.Errorf("failed to unmarshal answerer config: invalid max_tokens format")
		} else {
			answererConfig.MaxTokens = i
		}
	}

	// Parse timeout
	if val, ok := config["timeout"]; ok {
		if str, ok := val.(string); ok {
			if d, err := time.ParseDuration(str); err != nil {
				return nil, fmt.Errorf("failed to unmarshal answerer config: invalid timeout duration format")
			} else {
				answererConfig.Timeout = d
			}
		} else if num, ok := val.(int); ok {
			answererConfig.Timeout = time.Duration(num) * time.Second
		} else {
			return nil, fmt.Errorf("failed to unmarshal answerer config: timeout must be a string or integer")
		}
	}

	// Parse max_concurrency
	if val, ok := config["max_concurrency"]; ok {
		if i, err := strconv.Atoi(fmt.Sprintf("%v", val)); err != nil {
			return nil, fmt.Errorf("failed to unmarshal answerer config: invalid max_concurrency format")
		} else {
			answererConfig.MaxConcurrency = i
		}
	}

	return NewAnswererUnit(id, llmClient, answererConfig)
}
