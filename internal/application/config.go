package application

import (
	"gopkg.in/yaml.v3"
)

// GraphConfig defines the complete specification for an evaluation graph
// and serves as the primary configuration entry point for the system.
// Use GraphConfig when defining complex evaluation workflows that require
// structured execution of multiple analysis units with specific topologies.
type GraphConfig struct {
	// Version specifies the configuration schema version using semantic
	// versioning to ensure compatibility across system updates.
	Version string `yaml:"version" validate:"required,semver"`
	// Metadata contains descriptive information about the evaluation graph
	// including name, tags, and labels for organization and discovery.
	Metadata Metadata `yaml:"metadata" validate:"required"`
	// Units defines the individual evaluation components that will execute
	// within this graph, each with their own configuration and constraints.
	Units []UnitConfig `yaml:"units" validate:"required,min=1,dive"`
	// Graph specifies the execution topology that determines how units
	// are connected and the order in which they execute.
	Graph GraphTopology `yaml:"graph" validate:"required"`
}

// Metadata provides descriptive information about an evaluation graph
// to support organization, discovery, and operational management.
// Use Metadata to categorize graphs and provide context for operators
// and automated systems that need to identify graph characteristics.
type Metadata struct {
	// Name is the human-readable identifier for this evaluation graph
	// and must be unique within the deployment scope.
	Name string `yaml:"name" validate:"required,min=1,max=255"`
	// Description provides a detailed explanation of the graph's purpose
	// and intended use cases for documentation and discovery.
	Description string `yaml:"description" validate:"max=1000"`
	// Tags are categorical labels that enable filtering and grouping
	// of graphs by functional domain or operational characteristics.
	Tags []string `yaml:"tags" validate:"max=20,dive,min=1,max=50"`
	// Labels are arbitrary key-value pairs that provide flexible metadata
	// for integration with external systems and custom categorization.
	Labels map[string]string `yaml:"labels" validate:"max=50"`
}

// UnitConfig defines the specification for a single evaluation unit
// within an evaluation graph, including its behavior, constraints,
// and error handling policies.
// Use UnitConfig to define atomic evaluation components that can be
// composed into complex evaluation workflows.
type UnitConfig struct {
	// ID is the unique identifier for this unit within the graph
	// and must be alphanumeric for safe referencing in topologies.
	ID string `yaml:"id" validate:"required,alphanum,min=1,max=100"`
	// Type specifies the evaluation unit implementation to instantiate,
	// determining the available parameters and execution behavior.
	Type string `yaml:"type" validate:"required,oneof=answerer score_judge verification arithmetic_mean max_pool median_pool exact_match fuzzy_match custom"`
	// Model specifies the LLM provider and model to use for this unit
	// in the format "provider/model" or "provider/model@version".
	// When omitted, the unit will use the default provider configured
	// in the engine. Must match pattern: ^[a-z0-9]+/[A-Za-z0-9\-_\.]+(@[A-Za-z0-9\-_\.]+)?$
	Model string `yaml:"model,omitempty" validate:"omitempty,modelformat"`
	// Budget defines resource constraints that limit the unit's
	// consumption of tokens, cost, time, and retry attempts.
	Budget BudgetConfig `yaml:"budget" validate:"required"`
	// Parameters contains type-specific configuration as flexible YAML
	// that will be validated according to the unit type requirements.
	Parameters yaml.Node `yaml:"parameters"` // Flexible parameters for unit-specific validation
	// Retry configures the error recovery behavior including backoff
	// strategies and maximum attempt limits for transient failures.
	Retry RetryConfig `yaml:"retry"`
	// Timeout defines execution time limits and graceful shutdown
	// behavior to prevent units from consuming excessive resources.
	Timeout TimeoutConfig `yaml:"timeout"`
}

// BudgetConfig establishes resource consumption limits for evaluation units
// to prevent runaway costs and ensure predictable resource usage.
// Use BudgetConfig to enforce organizational policies on token usage,
// monetary costs, and execution time across evaluation workloads.
type BudgetConfig struct {
	// MaxTokens limits the total number of tokens that can be consumed
	// by this unit, preventing excessive API usage in language model calls.
	MaxTokens int64 `yaml:"max_tokens" validate:"omitempty,min=1,max=1000000"`
	// MaxCost sets the maximum monetary cost in dollars that this unit
	// is allowed to incur, providing cost control for expensive operations.
	MaxCost float64 `yaml:"max_cost" validate:"omitempty,min=0,max=10000"`
	// MaxCalls limits the number of API calls that can be made by this unit,
	// providing direct control over API usage and associated costs.
	MaxCalls int64 `yaml:"max_calls" validate:"omitempty,min=0,max=1000"`
	// TimeoutSeconds specifies the maximum execution time in seconds
	// before the unit is forcibly terminated to prevent resource exhaustion.
	TimeoutSeconds int `yaml:"timeout_seconds" validate:"omitempty,min=1,max=3600"`
}

// RetryConfig specifies the error recovery strategy for evaluation units
// when transient failures occur during execution.
// Use RetryConfig to define resilient behavior that can recover from
// temporary network issues, rate limiting, or service unavailability.
type RetryConfig struct {
	// MaxAttempts defines the total number of execution attempts including
	// the initial attempt, where 0 disables retries entirely.
	MaxAttempts int `yaml:"max_attempts" validate:"min=0,max=10"`
	// BackoffType determines the delay calculation strategy between retry
	// attempts to balance recovery speed with system load considerations.
	BackoffType string `yaml:"backoff_type" validate:"omitempty,oneof=constant exponential linear"`
	// InitialWait specifies the base delay in milliseconds before the
	// first retry attempt, serving as the foundation for backoff calculations.
	InitialWait int `yaml:"initial_wait_ms" validate:"omitempty,min=0,max=60000"`
	// MaxWait caps the maximum delay in milliseconds between retry attempts
	// to prevent excessively long waits in exponential backoff strategies.
	MaxWait int `yaml:"max_wait_ms" validate:"omitempty,min=0,max=300000"`
}

// TimeoutConfig controls execution time limits and shutdown behavior
// for evaluation units to ensure responsive system operation.
// Use TimeoutConfig to prevent units from consuming excessive time
// while allowing for graceful cleanup of resources and state.
type TimeoutConfig struct {
	// ExecutionTimeout specifies the maximum time in seconds that a unit
	// is allowed to execute before being interrupted and marked as failed.
	ExecutionTimeout int `yaml:"execution_timeout_seconds" validate:"omitempty,min=1,max=3600"`
	// GracefulShutdown defines the additional time in seconds allowed
	// for the unit to clean up resources after receiving a termination signal.
	GracefulShutdown int `yaml:"graceful_shutdown_seconds" validate:"omitempty,min=0,max=300"`
}

// GraphTopology specifies the structural organization and execution flow
// of units within an evaluation graph, supporting both sequential and
// parallel execution patterns.
// Use GraphTopology to define complex evaluation workflows that require
// specific ordering, parallelism, and conditional execution logic.
type GraphTopology struct {
	// Pipelines define sequential execution chains where units execute
	// in strict order, with each unit's output feeding to the next.
	Pipelines []PipelineConfig `yaml:"pipelines" validate:"dive"`
	// Layers define parallel execution groups where multiple units
	// can execute simultaneously to improve overall throughput.
	Layers []LayerConfig `yaml:"layers" validate:"dive"`
	// Edges specify directed connections between units, pipelines, and
	// layers, including conditional logic that controls execution flow.
	Edges []EdgeConfig `yaml:"edges" validate:"dive"`
}

// PipelineConfig defines a sequential execution chain where units
// execute in strict order with data flowing from one unit to the next.
// Use PipelineConfig when evaluation logic requires specific sequencing
// or when units have data dependencies that must be respected.
type PipelineConfig struct {
	// ID is the unique identifier for this pipeline within the graph
	// topology, used for referencing in edges and execution planning.
	ID string `yaml:"id" validate:"required,alphanum,min=1,max=100"`
	// Units lists the evaluation unit IDs in execution order, where
	// each unit's output becomes available to the subsequent unit.
	Units []string `yaml:"units" validate:"required,min=1,dive,alphanum"`
}

// LayerConfig defines a parallel execution group where multiple units
// execute simultaneously to improve throughput and reduce total runtime.
// Use LayerConfig when units are independent and can benefit from
// concurrent execution without data dependencies between them.
type LayerConfig struct {
	// ID is the unique identifier for this layer within the graph
	// topology, used for referencing in edges and execution coordination.
	ID string `yaml:"id" validate:"required,alphanum,min=1,max=100"`
	// Units lists the evaluation unit IDs that will execute in parallel,
	// with a minimum of two units required to justify layer overhead.
	Units []string `yaml:"units" validate:"required,min=2,dive,alphanum"`
}

// EdgeConfig establishes a directed connection between execution nodes
// in the graph topology, optionally including conditional logic that
// controls when the connection is activated.
// Use EdgeConfig to define execution dependencies and implement
// branching logic based on evaluation results or system state.
type EdgeConfig struct {
	// From identifies the source node (unit, pipeline, or layer) that
	// must complete before the target node can begin execution.
	From string `yaml:"from" validate:"required,alphanum"`
	// To identifies the target node that will receive control flow
	// and potentially data from the source node upon completion.
	To string `yaml:"to" validate:"required,alphanum"`
	// Conditions define optional logical predicates that must evaluate
	// to true for this edge to be traversed during execution.
	Conditions []ConditionConfig `yaml:"conditions" validate:"dive"`
}

// ConditionConfig specifies logical predicates that control edge traversal
// and execution flow within the evaluation graph topology.
// Use ConditionConfig to implement branching logic based on evaluation
// results, scores, or custom business rules.
type ConditionConfig struct {
	// Type specifies the condition evaluation strategy, determining
	// how the parameters will be interpreted and evaluated.
	Type string `yaml:"type" validate:"required,oneof=verdict_pass score_threshold custom"`
	// Parameters contains condition-specific configuration as flexible
	// YAML that will be validated according to the condition type.
	Parameters yaml.Node `yaml:"parameters"` // Flexible for condition-specific validation
}
