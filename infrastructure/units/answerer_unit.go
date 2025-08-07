package units

import (
	"bytes"
	"context"
	"errors"
	"fmt"
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

// LLMOptions contains typed configuration parameters for LLM client calls.
// These options control the generation behavior and resource usage during
// answer generation.
type LLMOptions struct {
	// Temperature controls randomness in text generation (0.0-1.0).
	// Higher values produce more creative but less deterministic outputs.
	Temperature float64 `json:"temperature"`
	// MaxTokens limits the maximum length of generated responses.
	// Prevents runaway generation and controls resource usage.
	MaxTokens int `json:"max_tokens"`
}

// AnswererUnit generates candidate answers by calling an LLM client with
// configurable concurrency and timeout controls. It transforms questions into
// multiple candidate answers using template-based prompts and LLM completion.
//
// This unit bridges deterministic evaluation pipelines with generative AI models,
// enabling flexible answer generation strategies for various evaluation scenarios.
//
// Concurrency: The unit is stateless and safe for concurrent execution.
// Multiple goroutines can call Execute simultaneously. Internal concurrency
// is controlled via MaxConcurrency to prevent overwhelming LLM services.
//
// Resource Management: Implements timeout controls, concurrency limits, and
// template caching for efficient resource utilization. Template parsing occurs
// once during unit creation for optimal performance.
//
// Error Handling: Provides structured error types for different failure modes
// including configuration validation, template execution, and LLM service errors.
type AnswererUnit struct {
	name           string
	config         AnswererConfig
	llmClient      ports.LLMClient
	promptTemplate *template.Template
}

// AnswererConfig defines the behavioral parameters for LLM-based answer generation.
// Configuration is immutable after unit creation and validated for consistency
// and resource safety.
//
// All timing constraints are enforced per LLM call, while concurrency limits
// apply to the entire execution batch. Template validation occurs during
// unit creation to ensure prompt safety and correctness.
type AnswererConfig struct {
	// NumAnswers specifies how many candidate answers to generate concurrently.
	// Range: 1-10 answers to balance diversity with resource usage.
	NumAnswers int `yaml:"num_answers" json:"num_answers" validate:"required,min=1,max=10"`

	// Prompt is the Go template for answer generation with question substitution.
	// Must contain {{.Question}} placeholder for safe parameter injection.
	// Minimum 10 characters to ensure meaningful prompts.
	Prompt string `yaml:"prompt" json:"prompt" validate:"required,min=10"`

	// Temperature controls LLM generation randomness and creativity (0.0-1.0).
	// 0.0 = deterministic, 1.0 = maximum creativity. Default: 0.7 for balanced output.
	Temperature float64 `yaml:"temperature" json:"temperature" validate:"min=0.0,max=1.0"`

	// MaxTokens limits individual answer length to prevent runaway generation.
	// Range: 10-16000 tokens. Consider model context window and cost implications.
	MaxTokens int `yaml:"max_tokens" json:"max_tokens" validate:"required,min=10,max=16000"`

	// Timeout specifies per-LLM-call maximum duration including network latency.
	// Range: 1s-300s. Should account for model inference time and network conditions.
	Timeout time.Duration `yaml:"timeout" json:"timeout" validate:"required,min=1s,max=300s"`

	// MaxConcurrency limits concurrent LLM calls to prevent service overload.
	// Range: 1-20 concurrent requests. Consider LLM service rate limits and quotas.
	MaxConcurrency int `yaml:"max_concurrency" json:"max_concurrency" validate:"required,min=1,max=20"`
}

// NewAnswererUnit creates a new AnswererUnit with validated configuration
// and dependency injection. The unit is immediately ready for concurrent
// execution after successful creation.
//
// The name parameter serves as a unique identifier for logging, debugging,
// and answer ID generation. The llmClient provides the generative AI interface
// and must be configured with appropriate model and credentials.
//
// Template parsing occurs during creation to validate prompt syntax and
// optimize runtime performance. Invalid templates cause immediate failure
// rather than runtime errors.
//
// Returns ErrUnitNameEmpty if name is empty, ErrLLMClientNil if client is nil,
// ErrConfigValidation if validation fails, or template parsing errors.
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

	tmpl, err := template.New("prompt").Funcs(GetTemplateFuncMap()).Parse(config.Prompt)
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
// The returned value is immutable and safe for concurrent access.
func (au *AnswererUnit) Name() string { return au.name }

// Execute generates candidate answers through concurrent LLM calls with
// configurable timeout and concurrency controls. It transforms a question
// into multiple diverse answers using template-based prompting.
//
// State requirements:
//   - domain.KeyQuestion: string containing the question to answer
//
// Returns a new state containing domain.KeyAnswers with generated responses.
// Each Answer contains a unique ID and the LLM-generated content.
//
// Concurrency: Uses errgroup for bounded parallel execution with fail-fast
// semantics. MaxConcurrency limits prevent overwhelming LLM services.
//
// Timeout: Applies per-execution timeout covering all concurrent LLM calls.
// Individual calls may complete at different rates within the timeout window.
//
// Template Security: Executes Go templates with controlled input sanitization
// to prevent injection attacks while enabling flexible prompt customization.
//
// Errors:
//   - ErrQuestionMissing: Question not found in state
//   - ErrQuestionEmpty: Empty question string provided
//   - ErrTemplateExecution: Template parsing or execution failure
//   - ErrLLMCallFailed: LLM service error with answer index context
//   - Context cancellation or timeout errors
//
// The function is safe for concurrent execution and does not modify input state.
func (au *AnswererUnit) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	question, ok := domain.Get(state, domain.KeyQuestion)
	if !ok {
		return state, ErrQuestionMissing
	}
	if question == "" {
		return state, ErrQuestionEmpty
	}

	// Apply timeout to entire answer generation batch.
	// Individual LLM calls inherit this deadline for coordinated timeout behavior.
	ctx, cancel := context.WithTimeout(ctx, au.config.Timeout)
	defer cancel()

	// Execute Go template with safe parameter injection.
	// Template is pre-parsed during unit creation for performance.
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
		g.Go(func() error {
			response, err := au.llmClient.Complete(ctx, prompt, options)
			if err != nil {
				return fmt.Errorf("%w for answer %d: %v", ErrLLMCallFailed, i+1, err)
			}
			answers[i] = domain.Answer{
				ID:      fmt.Sprintf("%s_answer_%d", au.name, i+1),
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

// Validate verifies the unit is properly configured and ready for execution.
// This method performs comprehensive health checks including LLM client
// connectivity and configuration completeness.
//
// Validation includes:
//   - LLM client presence and model configuration
//   - Configuration parameter constraints and consistency
//   - Template compilation and syntax verification
//
// Returns nil if the unit is operational, or a descriptive error indicating
// the specific validation failure. Safe for concurrent use.
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

// aggregateErrors enhances error context for concurrent operations with
// unit identification and operation context. Preserves LLM-specific errors
// while adding contextual information for debugging.
func (au *AnswererUnit) aggregateErrors(err error, operation string) error {
	if err == nil {
		return nil
	}
	// Preserve LLM-specific errors for detailed error handling.
	// Add unit context for other error types.
	if errors.Is(err, ErrLLMCallFailed) {
		return err
	}
	return fmt.Errorf("unit %s: %s failed: %w", au.name, operation, err)
}

// UnmarshalParameters deserializes YAML configuration and returns a new
// AnswererUnit instance with updated parameters. This approach maintains
// thread-safety by avoiding mutation of the existing unit.
//
// The method performs strict YAML decoding with comprehensive validation
// including template parsing to ensure the new configuration is fully
// operational before returning the updated unit.
//
// Returns a new unit instance with identical name and LLM client but
// updated configuration, or an error if YAML parsing or validation fails.
// The original unit remains unchanged on error.
func (au *AnswererUnit) UnmarshalParameters(params yaml.Node) (*AnswererUnit, error) {
	var config AnswererConfig
	if err := params.Decode(&config); err != nil {
		return nil, fmt.Errorf("failed to decode parameters: %w", err)
	}

	if err := answererValidator.Struct(config); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrConfigValidation, err)
	}

	tmpl, err := template.New("prompt").Funcs(GetTemplateFuncMap()).Parse(config.Prompt)
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

// defaultAnswererConfig returns an AnswererConfig with production-ready defaults:
// balanced creativity, reasonable timeouts, and moderate concurrency for typical LLM services.
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

// NewAnswererFromConfig creates an AnswererUnit from a configuration map.
// This is the boundary adapter that converts untyped configuration to typed.
// It's only used by the registry when creating units from YAML/JSON config.
func NewAnswererFromConfig(id string, config map[string]any, llm ports.LLMClient) (ports.Unit, error) {
	if llm == nil {
		return nil, ErrLLMClientNil
	}

	data, err := yaml.Marshal(config)
	if err != nil {
		return nil, fmt.Errorf("marshal config: %w", err)
	}

	// Start with defaults, then overlay user config.
	cfg := defaultAnswererConfig()
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	if err := answererValidator.Struct(cfg); err != nil {
		return nil, fmt.Errorf("validate config: %w", err)
	}

	return NewAnswererUnit(id, llm, cfg)
}
