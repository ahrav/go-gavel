package ports

import (
	"context"

	"github.com/ahrav/go-gavel/internal/domain"
)

// MergeStrategy defines how multiple states from parallel executions
// should be combined into a single output state.
// Implement this interface to provide custom merge logic for layers
// that accounts for your specific domain requirements and conflict resolution.
type MergeStrategy interface {
	// Merge combines multiple states from parallel executions into a single state.
	// The baseState parameter is the original input state to the layer.
	// The states parameter contains all successfully executed states from the layer.
	// The implementation must be deterministic - given the same inputs in the same
	// order, it must produce the same output.
	// The returned state should be a new instance; do not modify input states.
	Merge(baseState domain.State, states []domain.State) (domain.State, error)
}

// Executable defines the core contract for components that can be executed
// within a directed acyclic graph (DAG) evaluation system.
// Use Executable to implement evaluation units, pipelines, layers, or any
// component that needs to participate in graph-based execution workflows.
type Executable interface {
	// Execute processes the given state through this executable component
	// and returns the updated state along with any execution errors.
	// The context allows for cancellation and timeout control during execution.
	// Execute must be safe for concurrent use when called on different states.
	//
	// IMPORTANT: The input state is immutable and MUST NOT be modified.
	// domain.State uses copy-on-write semantics - use state.With() or
	// state.WithMultiple() to create a new state with modifications.
	// Multiple executables may receive the same state instance concurrently,
	// especially when running in parallel layers.
	Execute(ctx context.Context, state domain.State) (domain.State, error)

	// ID returns the unique string identifier for this executable component.
	// The ID must remain constant throughout the executable's lifetime
	// and should be unique within the scope of the containing graph.
	ID() string
}

// Pipeline defines a sequential execution container that runs multiple
// executables in strict order, where each executable's output becomes
// the input for the next executable in the sequence.
// Use Pipeline when evaluation logic requires specific ordering or when
// executables have data dependencies that must be respected.
type Pipeline interface {
	Executable

	// Add appends an executable to the end of this pipeline's execution
	// sequence, maintaining the order in which executables will be processed.
	// Add returns an error if the executable cannot be added due to
	// conflicts, validation failures, or pipeline capacity limits.
	Add(exec Executable) error

	// Executables returns the complete ordered list of executables
	// in this pipeline, preserving the sequence in which they will execute.
	// The returned slice should not be modified by callers.
	Executables() []Executable
}

// Layer defines a parallel execution container that runs multiple
// executables concurrently to improve throughput and reduce total runtime.
// Use Layer when executables are independent and can benefit from
// concurrent execution without data dependencies between them.
type Layer interface {
	Executable

	// Add includes an executable in this layer's parallel execution group.
	// All executables in a layer receive the same input state and execute
	// concurrently, with their results merged according to layer policies.
	// Add returns an error if the executable conflicts with existing
	// executables or violates layer constraints.
	Add(exec Executable) error

	// Executables returns all executables that will execute in parallel
	// within this layer, in no particular order since execution is concurrent.
	// The returned slice should not be modified by callers.
	Executables() []Executable

	// SetMergeStrategy configures how parallel execution results are combined.
	// If not set, a default last-write-wins strategy is used.
	// The merge strategy must be set before Execute is called.
	SetMergeStrategy(strategy MergeStrategy)
}

// Graph defines a directed acyclic graph (DAG) container that manages
// the execution topology and dependencies between executable components.
// Use Graph to orchestrate complex evaluation workflows that require
// specific ordering, parallelism, and conditional execution logic
// while ensuring no circular dependencies exist.
type Graph interface {
	// AddNode registers an executable component as a node in this graph.
	// The executable's ID must be unique within the graph scope.
	// AddNode returns an error if the executable's ID conflicts with
	// an existing node or if the executable is invalid.
	AddNode(exec Executable) error

	// AddEdge establishes a directed dependency relationship where the
	// target executable cannot begin until the source executable completes.
	// AddEdge returns an error if either ID is not found, if the edge
	// would create a cycle, or if the edge already exists.
	AddEdge(sourceID, targetID string) error

	// TopologicalSort computes the execution order that respects all
	// dependency relationships in the graph, returning executables in
	// an order where dependencies always execute before dependents.
	// TopologicalSort returns an error if the graph contains cycles.
	TopologicalSort() ([]Executable, error)

	// HasCycle performs cycle detection to determine if the graph
	// contains any circular dependencies that would prevent valid
	// topological ordering and execution.
	HasCycle() bool

	// GetNode retrieves an executable by its unique identifier.
	// GetNode returns the executable and true if found, or nil and false
	// if no executable with the given ID exists in the graph.
	//
	// WARNING: The returned Executable is the actual instance stored in the
	// graph. Callers MUST NOT modify the executable's internal state as
	// this could cause data races. The executable should be treated as
	// read-only. If modifications are needed, they should be synchronized
	// externally or a new graph should be constructed.
	GetNode(id string) (Executable, bool)
}
