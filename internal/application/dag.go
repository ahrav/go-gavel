package application

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"

	"github.com/ahrav/go-gavel/internal/domain"
	"github.com/ahrav/go-gavel/internal/ports"
)

// Pipeline is a sequential execution container that processes executables
// in strict order, where each executable's output becomes the input for
// the next executable in the sequence.
// Use Pipeline when evaluation logic requires specific sequencing or when
// executables have data dependencies that must be respected.
type Pipeline struct {
	// id is the unique identifier for this pipeline within the graph
	// topology, used for referencing in edges and execution planning.
	id string
	// executables contains the ordered list of components that will execute
	// sequentially, with data flowing from one to the next.
	executables []ports.Executable
	// idSet tracks executable IDs for O(1) duplicate detection.
	idSet map[string]struct{}
	// mu provides thread-safe access to the executables slice during
	// concurrent read and write operations.
	mu sync.RWMutex
}

// NewPipeline creates a new sequential execution pipeline with the specified
// identifier, ready to accept executable components.
// The pipeline will execute added components in the order they were added.
func NewPipeline(id string) *Pipeline {
	return &Pipeline{
		id:          id,
		executables: make([]ports.Executable, 0),
		idSet:       make(map[string]struct{}),
	}
}

// Execute processes all executables in this pipeline sequentially,
// passing the output state from each executable as input to the next.
// Execute respects context cancellation and returns immediately if the
// context is cancelled between executable runs.
// Execute returns an error if any executable fails, including the
// executable ID in the error message for debugging.
func (p *Pipeline) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	p.mu.RLock()
	executables := make([]ports.Executable, len(p.executables))
	copy(executables, p.executables)
	p.mu.RUnlock()

	currentState := state
	for _, exec := range executables {
		select {
		case <-ctx.Done():
			return currentState, ctx.Err()
		default:
			newState, err := exec.Execute(ctx, currentState)
			if err != nil {
				return currentState, fmt.Errorf("pipeline %s: execution failed at %s: %w", p.id, exec.ID(), err)
			}
			currentState = newState
		}
	}

	return currentState, nil
}

// ID returns the unique string identifier for this pipeline.
// The ID remains constant throughout the pipeline's lifetime and
// is used for referencing in graph topologies and error reporting.
func (p *Pipeline) ID() string {
	return p.id
}

// Add appends an executable to the end of this pipeline's execution
// sequence, maintaining the order in which executables will be processed.
// Add returns an error if the executable is nil or if an executable
// with the same ID already exists in the pipeline.
// Add is safe for concurrent use with Execute.
func (p *Pipeline) Add(exec ports.Executable) error {
	if exec == nil {
		return fmt.Errorf("cannot add nil executable to pipeline")
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	// Check for duplicate IDs - O(1) lookup.
	execID := exec.ID()
	if _, exists := p.idSet[execID]; exists {
		return fmt.Errorf("executable with ID %s already exists in pipeline", execID)
	}

	p.executables = append(p.executables, exec)
	p.idSet[execID] = struct{}{}
	return nil
}

// Executables returns a copy of the complete ordered list of executables
// in this pipeline, preserving the sequence in which they will execute.
// The returned slice is safe to modify without affecting the pipeline.
// Executables is safe for concurrent use.
func (p *Pipeline) Executables() []ports.Executable {
	p.mu.RLock()
	defer p.mu.RUnlock()

	result := make([]ports.Executable, len(p.executables))
	copy(result, p.executables)
	return result
}

// Layer is a parallel execution container that runs multiple executables
// concurrently to improve throughput and reduce total runtime.
// Use Layer when executables are independent and can benefit from
// concurrent execution without data dependencies between them.
type Layer struct {
	// id is the unique identifier for this layer within the graph
	// topology, used for referencing in edges and execution coordination.
	id string
	// executables contains the list of components that will execute
	// concurrently, all receiving the same input state.
	executables []ports.Executable
	// idSet tracks executable IDs for O(1) duplicate detection.
	idSet map[string]struct{}
	// mergeStrategy defines how to combine results from parallel executions.
	// If nil, defaultMergeStrategy is used.
	mergeStrategy ports.MergeStrategy
	// concurrencyLimit controls the maximum number of concurrent executions.
	// Defaults to runtime.NumCPU() * 2 if not set.
	concurrencyLimit int
	// mu provides thread-safe access to the executables slice during
	// concurrent read and write operations.
	mu sync.RWMutex
}

// NewLayer creates a new parallel execution layer with the specified
// identifier, ready to accept executable components that will run concurrently.
// All executables in the layer receive the same input state.
func NewLayer(id string) *Layer {
	return &Layer{
		id:               id,
		executables:      make([]ports.Executable, 0),
		idSet:            make(map[string]struct{}),
		concurrencyLimit: runtime.NumCPU() * 2, // Default to 2x CPU cores
	}
}

// Execute runs all executables in this layer concurrently, with each
// executable receiving the same input state.
// Execute collects results from all parallel executions and merges their
// output states using a last-write-wins strategy.
// Execute returns an error if any executable fails, including details
// about all failed executions for comprehensive debugging.
// Execute uses goroutines and channels for coordination.
// Note: domain.State is immutable, so concurrent access is safe.
func (l *Layer) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	l.mu.RLock()
	executables := make([]ports.Executable, len(l.executables))
	copy(executables, l.executables)
	limit := l.concurrencyLimit
	if limit <= 0 {
		limit = runtime.NumCPU() * 2
	}
	l.mu.RUnlock()

	if len(executables) == 0 {
		return state, nil
	}

	// Channel for collecting results.
	type result struct {
		state domain.State
		err   error
		id    string
	}

	resultChan := make(chan result, len(executables))
	var wg sync.WaitGroup

	// Semaphore to limit concurrency.
	semaphore := make(chan struct{}, limit)

	// Execute all units in parallel with concurrency control.
	for _, exec := range executables {
		wg.Add(1)
		go func(e ports.Executable) {
			defer wg.Done()

			// Acquire semaphore slot.
			select {
			case semaphore <- struct{}{}:
				// Slot acquired, proceed with execution.
				defer func() { <-semaphore }() // Release slot when done.
			case <-ctx.Done():
				// Context cancelled, exit early.
				return
			}

			// Execute with the immutable state.
			newState, err := e.Execute(ctx, state)

			// Send result to channel (non-blocking due to buffer).
			select {
			case resultChan <- result{
				state: newState,
				err:   err,
				id:    e.ID(),
			}:
			case <-ctx.Done():
				// Context cancelled, exit early.
				return
			}
		}(exec)
	}

	// Wait for all executions to complete.
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect results and merge states.
	var errs []error
	states := make([]domain.State, 0, len(executables))
	remaining := len(executables)

	for remaining > 0 {
		select {
		case <-ctx.Done():
			// Context cancelled, return immediately.
			return state, ctx.Err()
		case res, ok := <-resultChan:
			if !ok {
				// Channel closed, all results collected.
				break
			}
			remaining--

			if res.err != nil {
				errs = append(errs, fmt.Errorf("executable %s: %w", res.id, res.err))
			} else {
				states = append(states, res.state)
			}
		}
	}

	// If any execution failed, return aggregated errors.
	if len(errs) > 0 {
		// Use errors.Join for proper error aggregation.
		return state, fmt.Errorf("layer %s failed with %d errors: %w", l.id, len(errs), errors.Join(errs...))
	}

	// Apply merge strategy.
	strategy := l.mergeStrategy
	if strategy == nil {
		strategy = defaultMergeStrategy{}
	}

	mergedState, err := strategy.Merge(state, states)
	if err != nil {
		return state, fmt.Errorf("layer %s: merge failed: %w", l.id, err)
	}

	return mergedState, nil
}

// ID returns the unique string identifier for this layer.
// The ID remains constant throughout the layer's lifetime and
// is used for referencing in graph topologies and error reporting.
func (l *Layer) ID() string {
	return l.id
}

// Add includes an executable in this layer's parallel execution group.
// All executables in a layer receive the same input state and execute
// concurrently, with their results merged according to layer policies.
// Add returns an error if the executable is nil or if an executable
// with the same ID already exists in the layer.
// Add is safe for concurrent use with Execute.
func (l *Layer) Add(exec ports.Executable) error {
	if exec == nil {
		return fmt.Errorf("cannot add nil executable to layer")
	}

	l.mu.Lock()
	defer l.mu.Unlock()

	// Check for duplicate IDs - O(1) lookup.
	execID := exec.ID()
	if _, exists := l.idSet[execID]; exists {
		return fmt.Errorf("executable with ID %s already exists in layer", execID)
	}

	l.executables = append(l.executables, exec)
	l.idSet[execID] = struct{}{}
	return nil
}

// Executables returns a copy of all executables that will execute in
// parallel within this layer, in no particular order since execution
// is concurrent. The returned slice is safe to modify without affecting
// the layer. Executables is safe for concurrent use.
func (l *Layer) Executables() []ports.Executable {
	l.mu.RLock()
	defer l.mu.RUnlock()

	result := make([]ports.Executable, len(l.executables))
	copy(result, l.executables)
	return result
}

// SetMergeStrategy configures how parallel execution results are combined.
// If not set, a default last-write-wins strategy is used.
// The merge strategy must be set before Execute is called.
// SetMergeStrategy is safe for concurrent use.
func (l *Layer) SetMergeStrategy(strategy ports.MergeStrategy) {
	l.mu.Lock()
	defer l.mu.Unlock()

	l.mergeStrategy = strategy
}

// SetConcurrencyLimit configures the maximum number of executables that
// can run concurrently within this layer.
// If not set or set to 0 or negative, defaults to runtime.NumCPU() * 2.
// The concurrency limit should be set before Execute is called.
// SetConcurrencyLimit is safe for concurrent use.
func (l *Layer) SetConcurrencyLimit(limit int) {
	l.mu.Lock()
	defer l.mu.Unlock()

	l.concurrencyLimit = limit
}

// Graph is a directed acyclic graph (DAG) container that manages
// the execution topology and dependencies between executable components.
// Use Graph to orchestrate complex evaluation workflows that require
// specific ordering, parallelism, and conditional execution logic
// while ensuring no circular dependencies exist.
type Graph struct {
	// nodes maps executable IDs to their corresponding executable components,
	// providing fast lookup during graph operations and execution.
	nodes map[string]ports.Executable
	// edges represents the adjacency list mapping each node ID to its
	// list of dependent target node IDs for dependency tracking.
	edges map[string][]string // adjacency list: node ID -> list of target IDs.
	// edgeSet provides O(1) duplicate edge detection.
	// Key format: "sourceID->targetID"
	edgeSet map[string]struct{}
	// inDegree tracks the number of incoming edges for each node,
	// used for efficient topological sorting algorithms.
	inDegree map[string]int // for topological sort.
	// mu provides thread-safe access to all graph data structures
	// during concurrent operations.
	mu sync.RWMutex
}

// NewGraph creates a new empty directed acyclic graph ready to accept
// executable nodes and dependency edges.
// The graph maintains internal data structures for efficient topological
// sorting and cycle detection.
func NewGraph() *Graph {
	return &Graph{
		nodes:    make(map[string]ports.Executable),
		edges:    make(map[string][]string),
		edgeSet:  make(map[string]struct{}),
		inDegree: make(map[string]int),
	}
}

// AddNode registers an executable component as a node in this graph.
// The executable's ID must be unique within the graph scope.
// AddNode initializes the node's adjacency list and in-degree counter
// for dependency tracking.
// AddNode returns an error if the executable is nil or if an executable
// with the same ID already exists in the graph.
func (g *Graph) AddNode(exec ports.Executable) error {
	if exec == nil {
		return fmt.Errorf("cannot add nil executable to graph")
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	id := exec.ID()
	if _, exists := g.nodes[id]; exists {
		return fmt.Errorf("node with ID %s already exists in graph", id)
	}

	g.nodes[id] = exec
	g.edges[id] = make([]string, 0)
	g.inDegree[id] = 0

	return nil
}

// AddEdge establishes a directed dependency relationship where the
// target executable cannot begin until the source executable completes.
// AddEdge automatically performs cycle detection and rolls back the
// edge if it would create a circular dependency.
// AddEdge returns an error if either ID is not found, if the edge
// already exists, or if the edge would create a cycle.
func (g *Graph) AddEdge(sourceID, targetID string) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	// Validate nodes exist.
	if _, exists := g.nodes[sourceID]; !exists {
		return fmt.Errorf("source node %s does not exist", sourceID)
	}
	if _, exists := g.nodes[targetID]; !exists {
		return fmt.Errorf("target node %s does not exist", targetID)
	}

	// Check if edge already exists - O(1) lookup.
	edgeKey := sourceID + "->" + targetID
	if _, exists := g.edgeSet[edgeKey]; exists {
		return fmt.Errorf("edge from %s to %s already exists", sourceID, targetID)
	}

	// Add edge.
	g.edges[sourceID] = append(g.edges[sourceID], targetID)
	g.edgeSet[edgeKey] = struct{}{}
	g.inDegree[targetID]++

	// Check for cycles.
	if g.hasCycleUnsafe() {
		// Rollback the edge.
		g.edges[sourceID] = g.edges[sourceID][:len(g.edges[sourceID])-1]
		delete(g.edgeSet, edgeKey)
		g.inDegree[targetID]--
		return fmt.Errorf("adding edge from %s to %s would create a cycle", sourceID, targetID)
	}

	return nil
}

// TopologicalSort computes the execution order that respects all
// dependency relationships in the graph, returning executables in
// an order where dependencies always execute before dependents.
// TopologicalSort uses Kahn's algorithm for efficient computation
// and returns an error if the graph contains cycles.
func (g *Graph) TopologicalSort() ([]ports.Executable, error) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	// Create a copy of inDegree for the sort.
	inDegreeCopy := make(map[string]int)
	for k, v := range g.inDegree {
		inDegreeCopy[k] = v
	}

	// Find all nodes with no incoming edges.
	queue := make([]string, 0)
	for id, degree := range inDegreeCopy {
		if degree == 0 {
			queue = append(queue, id)
		}
	}

	result := make([]ports.Executable, 0, len(g.nodes))

	// Process nodes in topological order.
	for len(queue) > 0 {
		// Dequeue.
		nodeID := queue[0]
		queue = queue[1:]

		// Add to result.
		result = append(result, g.nodes[nodeID])

		// Reduce in-degree of neighbors.
		for _, neighbor := range g.edges[nodeID] {
			inDegreeCopy[neighbor]--
			if inDegreeCopy[neighbor] == 0 {
				queue = append(queue, neighbor)
			}
		}
	}

	// If we didn't process all nodes, there's a cycle.
	if len(result) != len(g.nodes) {
		return nil, fmt.Errorf("graph contains a cycle")
	}

	return result, nil
}

// HasCycle performs cycle detection to determine if the graph
// contains any circular dependencies that would prevent valid
// topological ordering and execution.
// HasCycle uses depth-first search with node coloring for detection.
func (g *Graph) HasCycle() bool {
	g.mu.RLock()
	defer g.mu.RUnlock()

	return g.hasCycleUnsafe()
}

// hasCycleUnsafe performs cycle detection using depth-first search
// with three-color node marking (white, gray, black) to identify
// back edges that indicate cycles.
// hasCycleUnsafe must be called with the graph mutex held and is
// used internally by both HasCycle and AddEdge methods.
func (g *Graph) hasCycleUnsafe() bool {
	// Use DFS with coloring for cycle detection.
	// White (0): unvisited, Gray (1): visiting, Black (2): visited.
	colors := make(map[string]int)
	for id := range g.nodes {
		colors[id] = 0 // white.
	}

	var dfs func(nodeID string) bool
	dfs = func(nodeID string) bool {
		colors[nodeID] = 1 // gray

		for _, neighbor := range g.edges[nodeID] {
			if colors[neighbor] == 1 { // gray - back edge found.
				return true
			}
			if colors[neighbor] == 0 && dfs(neighbor) {
				return true
			}
		}

		colors[nodeID] = 2 // black.
		return false
	}

	// Check each unvisited node.
	for id := range g.nodes {
		if colors[id] == 0 && dfs(id) {
			return true
		}
	}

	return false
}

// GetNode retrieves an executable by its unique identifier.
// GetNode returns the executable and true if found, or nil and false
// if no executable with the given ID exists in the graph.
// GetNode is safe for concurrent use.
func (g *Graph) GetNode(id string) (ports.Executable, bool) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	exec, exists := g.nodes[id]
	return exec, exists
}

// defaultMergeStrategy implements a simple last-write-wins merge strategy.
// This is provided as a fallback when no custom merge strategy is specified.
// Note: This strategy is deterministic only when states are processed in a
// consistent order, which may not be guaranteed with concurrent execution.
type defaultMergeStrategy struct{}

// Merge implements the MergeStrategy interface with a simple last-write-wins approach.
// Each state overwrites the previous one completely.
// For production use, consider implementing a domain-specific merge strategy
// that handles conflicts appropriately.
func (d defaultMergeStrategy) Merge(baseState domain.State, states []domain.State) (domain.State, error) {
	if len(states) == 0 {
		return baseState, nil
	}

	// Simple last-write-wins: return the last state in the slice.
	// Note: Order may not be deterministic due to concurrent execution.
	return states[len(states)-1], nil
}
