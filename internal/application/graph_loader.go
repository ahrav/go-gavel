package application

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"

	"github.com/go-playground/validator/v10"
	"golang.org/x/sync/singleflight"
	"gopkg.in/yaml.v3"

	"github.com/ahrav/go-gavel/internal/ports"
)

// GraphLoader provides YAML configuration parsing, validation, and caching
// for evaluation graphs, transforming declarative YAML specifications into
// executable graph structures.
// Use GraphLoader to load graphs from files or readers while benefiting
// from SHA256-based caching and comprehensive validation.
type GraphLoader struct {
	// validator performs struct field validation and custom validation
	// rules for graph configurations and their nested components.
	validator *validator.Validate
	// unitRegistry provides factory methods for creating evaluation units
	// based on their type and configuration parameters.
	unitRegistry ports.UnitRegistry
	// cache stores compiled graphs indexed by SHA256 hash of source YAML
	// to avoid recompilation of identical configurations.
	// WARNING: Cached graphs MUST NOT be mutated. The Graph methods
	// AddNode and AddEdge should never be called on cached graphs.
	cache map[string]*Graph // SHA256 hash -> compiled graph
	// cacheMu provides thread-safe access to the cache map during
	// concurrent read and write operations.
	cacheMu sync.RWMutex
	// sf prevents duplicate graph compilation when multiple goroutines
	// request the same graph simultaneously.
	sf singleflight.Group
}

// NewGraphLoader creates a new graph loader with validation capabilities
// and an empty cache, ready to load and compile evaluation graphs.
// NewGraphLoader registers custom validators for semantic validation
// beyond basic struct field validation.
// NewGraphLoader returns an error if validator registration fails.
func NewGraphLoader(unitRegistry ports.UnitRegistry) (*GraphLoader, error) {
	v := validator.New()

	// Register custom validators for semantic validation beyond struct tags.
	if err := registerCustomValidators(v); err != nil {
		return nil, fmt.Errorf("failed to register validators: %w", err)
	}

	return &GraphLoader{
		validator:    v,
		unitRegistry: unitRegistry,
		cache:        make(map[string]*Graph),
	}, nil
}

// load is the common implementation for loading graphs from byte data,
// utilizing singleflight to prevent duplicate compilation and SHA256-based
// caching for efficiency.
// load performs comprehensive validation and returns a new graph instance.
// WARNING: The returned graph is a pointer to a cached instance. Callers
// MUST NOT mutate the graph by calling AddNode or AddEdge methods.
func (gl *GraphLoader) load(ctx context.Context, data []byte) (*Graph, error) {
	// Parse YAML first to normalize it before hashing.
	config, err := gl.parseYAML(data)
	if err != nil {
		return nil, fmt.Errorf("failed to parse YAML: %w", err)
	}

	// Calculate hash based on normalized config, not raw bytes.
	hash, err := gl.calculateConfigHash(config)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate hash: %w", err)
	}

	// Use singleflight to prevent multiple goroutines from compiling
	// the same graph simultaneously.
	v, err, _ := gl.sf.Do(hash, func() (any, error) {
		// Check cache inside singleflight to handle race between cache check
		// and singleflight group execution.
		if graph, ok := gl.getCachedGraph(hash); ok {
			return graph, nil
		}

		// Config is already parsed and validated outside singleflight.
		if err := gl.validateConfig(config); err != nil {
			return nil, fmt.Errorf("validation failed: %w", err)
		}

		graph, err := gl.buildGraph(ctx, config)
		if err != nil {
			return nil, fmt.Errorf("failed to build graph: %w", err)
		}

		gl.cacheGraph(hash, graph)

		return graph, nil
	})

	if err != nil {
		return nil, err
	}

	return v.(*Graph), nil
}

// LoadFromFile loads and compiles an evaluation graph from a YAML file,
// utilizing SHA256-based caching to avoid recompilation of identical files.
// LoadFromFile performs comprehensive validation including struct validation,
// semantic validation, and unit parameter validation.
// WARNING: The returned graph is a pointer to a cached instance. Callers
// MUST NOT mutate the graph by calling AddNode or AddEdge methods.
// LoadFromFile returns an error if file reading, parsing, validation,
// or graph compilation fails.
func (gl *GraphLoader) LoadFromFile(ctx context.Context, path string) (*Graph, error) {
	// Clean the path to prevent directory traversal attacks.
	cleanPath := filepath.Clean(path)

	data, err := os.ReadFile(cleanPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	return gl.load(ctx, data)
}

// LoadFromReader loads and compiles an evaluation graph from an io.Reader,
// supporting any source that implements the Reader interface.
// LoadFromReader reads all data into memory, applies SHA256-based caching,
// and performs the same validation as LoadFromFile.
// WARNING: The returned graph is a pointer to a cached instance. Callers
// MUST NOT mutate the graph by calling AddNode or AddEdge methods.
// LoadFromReader returns an error if reading, parsing, validation,
// or graph compilation fails.
func (gl *GraphLoader) LoadFromReader(ctx context.Context, r io.Reader) (*Graph, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("failed to read data: %w", err)
	}

	return gl.load(ctx, data)
}

// parseYAML unmarshals YAML byte data into a structured GraphConfig,
// handling nested configuration elements and preserving parameter flexibility.
// parseYAML uses strict decoding to detect unknown fields, preventing
// configuration typos from being silently ignored.
// parseYAML returns an error if YAML syntax is invalid, if unknown fields
// are present, or if the structure doesn't match the expected GraphConfig schema.
func (gl *GraphLoader) parseYAML(data []byte) (*GraphConfig, error) {
	var config GraphConfig
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true) // Strict mode - fail on unknown fields.

	if err := decoder.Decode(&config); err != nil {
		return nil, fmt.Errorf("YAML decode failed: %w", err)
	}
	return &config, nil
}

// validateConfig performs comprehensive validation on a parsed graph
// configuration, including both struct field validation and semantic
// validation of relationships between configuration elements.
// validateConfig returns an error if any validation rule fails.
func (gl *GraphLoader) validateConfig(config *GraphConfig) error {
	if err := gl.validator.Struct(config); err != nil {
		return fmt.Errorf("struct validation failed: %w", err)
	}

	if err := gl.validateSemantics(config); err != nil {
		return fmt.Errorf("semantic validation failed: %w", err)
	}

	return nil
}

// validateSemantics performs domain-specific validation rules that
// cannot be expressed through struct tags, including uniqueness
// constraints, reference integrity, and parameter validation.
// validateSemantics ensures all node IDs are globally unique across
// units, pipelines, and layers to prevent ambiguous edge references.
func (gl *GraphLoader) validateSemantics(config *GraphConfig) error {
	// Track all node IDs globally to ensure uniqueness across categories.
	allNodeIDs := make(map[string]string) // ID -> node type for better error messages.
	unitIDs := make(map[string]struct{})

	// Check unit IDs for global uniqueness.
	for _, unit := range config.Units {
		if nodeType, exists := allNodeIDs[unit.ID]; exists {
			return fmt.Errorf("duplicate ID %q: already used by %s", unit.ID, nodeType)
		}
		allNodeIDs[unit.ID] = "unit"
		unitIDs[unit.ID] = struct{}{}

		if err := ValidateUnitParameters(unit.Type, unit.Parameters); err != nil {
			return fmt.Errorf("unit %s parameter validation failed: %w", unit.ID, err)
		}
	}

	// Check pipeline IDs for global uniqueness.
	for _, pipeline := range config.Graph.Pipelines {
		if nodeType, exists := allNodeIDs[pipeline.ID]; exists {
			return fmt.Errorf("duplicate ID %q: already used by %s", pipeline.ID, nodeType)
		}
		allNodeIDs[pipeline.ID] = "pipeline"

		for _, unitID := range pipeline.Units {
			if _, exists := unitIDs[unitID]; !exists {
				return fmt.Errorf("pipeline %s references non-existent unit: %s", pipeline.ID, unitID)
			}
		}
	}

	// Check layer IDs for global uniqueness.
	for _, layer := range config.Graph.Layers {
		if nodeType, exists := allNodeIDs[layer.ID]; exists {
			return fmt.Errorf("duplicate ID %q: already used by %s", layer.ID, nodeType)
		}
		allNodeIDs[layer.ID] = "layer"

		for _, unitID := range layer.Units {
			if _, exists := unitIDs[unitID]; !exists {
				return fmt.Errorf("layer %s references non-existent unit: %s", layer.ID, unitID)
			}
		}
	}

	for _, edge := range config.Graph.Edges {
		if _, exists := allNodeIDs[edge.From]; !exists {
			return fmt.Errorf("edge references non-existent source node: %s", edge.From)
		}
		if _, exists := allNodeIDs[edge.To]; !exists {
			return fmt.Errorf("edge references non-existent target node: %s", edge.To)
		}

		// TODO: Edge conditions are validated but not yet implemented in graph execution.
		// This validation ensures the configuration is correct for future implementation.
		// Remove this validation if conditions won't be implemented, or implement
		// condition support in the graph execution engine.
		for i, condition := range edge.Conditions {
			if err := ValidateConditionParameters(condition.Type, condition.Parameters); err != nil {
				return fmt.Errorf("edge %s->%s condition %d validation failed: %w",
					edge.From, edge.To, i, err)
			}
		}
	}

	return nil
}

// buildGraph constructs an executable graph from a validated configuration,
// creating units, pipelines, layers, and their dependency relationships.
// buildGraph instantiates units through the unit registry, wraps them
// in adapters, and establishes edges while ensuring no cycles are created.
// buildGraph returns an error if unit creation, graph construction,
// or cycle detection fails.
func (gl *GraphLoader) buildGraph(ctx context.Context, config *GraphConfig) (*Graph, error) {
	graph := NewGraph()

	units := make(map[string]ports.Unit)
	for _, unitConfig := range config.Units {
		unit, err := gl.createUnit(unitConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create unit %s: %w", unitConfig.ID, err)
		}
		units[unitConfig.ID] = unit
	}

	pipelines := make(map[string]ports.Pipeline)
	for _, pipelineConfig := range config.Graph.Pipelines {
		pipeline := NewPipeline(pipelineConfig.ID)

		for _, unitID := range pipelineConfig.Units {
			unit, ok := units[unitID]
			if !ok {
				return nil, fmt.Errorf("unit %s not found for pipeline %s", unitID, pipelineConfig.ID)
			}
			executable := NewUnitAdapter(unit, unitID)
			if err := pipeline.Add(executable); err != nil {
				return nil, fmt.Errorf("failed to add unit to pipeline: %w", err)
			}
		}

		pipelines[pipelineConfig.ID] = pipeline
		if err := graph.AddNode(pipeline); err != nil {
			return nil, fmt.Errorf("failed to add pipeline to graph: %w", err)
		}
	}

	layers := make(map[string]ports.Layer)
	for _, layerConfig := range config.Graph.Layers {
		layer := NewLayer(layerConfig.ID)

		for _, unitID := range layerConfig.Units {
			unit, ok := units[unitID]
			if !ok {
				return nil, fmt.Errorf("unit %s not found for layer %s", unitID, layerConfig.ID)
			}
			// Wrap unit in adapter to implement Executable.
			executable := NewUnitAdapter(unit, unitID)
			if err := layer.Add(executable); err != nil {
				return nil, fmt.Errorf("failed to add unit to layer: %w", err)
			}
		}

		layers[layerConfig.ID] = layer
		if err := graph.AddNode(layer); err != nil {
			return nil, fmt.Errorf("failed to add layer to graph: %w", err)
		}
	}

	// Track which units are placed in pipelines or layers for O(1) lookup.
	placedUnits := make(map[string]struct{})

	// Record units placed in pipelines.
	for _, pipeline := range config.Graph.Pipelines {
		for _, unitID := range pipeline.Units {
			placedUnits[unitID] = struct{}{}
		}
	}

	// Record units placed in layers.
	for _, layer := range config.Graph.Layers {
		for _, unitID := range layer.Units {
			placedUnits[unitID] = struct{}{}
		}
	}

	// Add standalone units to graph (not part of any pipeline or layer).
	for id, unit := range units {
		if _, isPlaced := placedUnits[id]; !isPlaced {
			// Wrap unit in adapter to implement Executable.
			executable := NewUnitAdapter(unit, id)
			if err := graph.AddNode(executable); err != nil {
				return nil, fmt.Errorf("failed to add unit to graph: %w", err)
			}
		}
	}

	// Add edges to establish execution dependencies between graph nodes.
	for _, edge := range config.Graph.Edges {
		if err := graph.AddEdge(edge.From, edge.To); err != nil {
			return nil, fmt.Errorf("failed to add edge: %w", err)
		}
	}

	// Verify no cycles exist to ensure graph can execute without deadlock.
	if graph.HasCycle() {
		return nil, fmt.Errorf("graph contains cycles")
	}

	return graph, nil
}

// createUnit instantiates an evaluation unit from its configuration,
// merging YAML parameters with budget, retry, and timeout settings.
// createUnit delegates to the unit registry for type-specific creation
// while handling parameter decoding and configuration merging.
// createUnit returns an error if parameter decoding or unit creation fails.
func (gl *GraphLoader) createUnit(config UnitConfig) (ports.Unit, error) {
	// Convert yaml.Node parameters to map[string]any.
	var params map[string]any
	if err := config.Parameters.Decode(&params); err != nil {
		return nil, fmt.Errorf("failed to decode parameters: %w", err)
	}

	// Merge parameters with other configuration.
	unitConfig := map[string]any{
		"budget":  config.Budget,
		"retry":   config.Retry,
		"timeout": config.Timeout,
	}

	// Add decoded parameters to merge YAML config with system settings.
	for k, v := range params {
		unitConfig[k] = v
	}

	// Use the unit registry to create the unit.
	unit, err := gl.unitRegistry.CreateUnit(config.Type, config.ID, unitConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create unit: %w", err)
	}

	return unit, nil
}

// calculateConfigHash computes the SHA256 hash of a normalized GraphConfig
// for cache indexing, ensuring semantically identical configurations produce
// the same hash regardless of whitespace or key ordering differences.
// calculateConfigHash returns a hexadecimal string representation of the hash.
func (gl *GraphLoader) calculateConfigHash(config *GraphConfig) (string, error) {
	// Normalize the config by re-encoding it with consistent formatting.
	var buf bytes.Buffer
	encoder := yaml.NewEncoder(&buf)
	encoder.SetIndent(2) // Use consistent 2-space indentation.

	if err := encoder.Encode(config); err != nil {
		return "", fmt.Errorf("failed to encode config for hashing: %w", err)
	}

	hash := sha256.Sum256(buf.Bytes())
	return hex.EncodeToString(hash[:]), nil
}

// getCachedGraph attempts to retrieve a previously compiled graph
// from the cache using its SHA256 hash as the lookup key.
// getCachedGraph returns the cached graph and true if found,
// or nil and false if no cached version exists.
// getCachedGraph is safe for concurrent use.
func (gl *GraphLoader) getCachedGraph(hash string) (*Graph, bool) {
	gl.cacheMu.RLock()
	defer gl.cacheMu.RUnlock()

	graph, ok := gl.cache[hash]
	return graph, ok
}

// cacheGraph stores a compiled graph in the cache indexed by its
// source YAML's SHA256 hash for future retrieval.
// cacheGraph is safe for concurrent use and will overwrite
// any existing entry with the same hash.
func (gl *GraphLoader) cacheGraph(hash string, graph *Graph) {
	gl.cacheMu.Lock()
	defer gl.cacheMu.Unlock()

	gl.cache[hash] = graph
}

// ClearCache removes all cached graphs and reinitializes the cache map,
// forcing subsequent loads to recompile from source.
// ClearCache is safe for concurrent use and is useful for development
// or when memory management is needed.
func (gl *GraphLoader) ClearCache() {
	gl.cacheMu.Lock()
	defer gl.cacheMu.Unlock()

	gl.cache = make(map[string]*Graph)
}

// registerCustomValidators registers domain-specific validation functions
// with the validator instance, including semantic version validation
// and graph-specific validation rules.
// registerCustomValidators returns an error if any validator registration fails.
func registerCustomValidators(v *validator.Validate) error {
	// Register semver validator.
	if err := v.RegisterValidation("semver", validateSemver); err != nil {
		return fmt.Errorf("failed to register semver validator: %w", err)
	}

	// Register graph-specific validators.
	if err := RegisterGraphValidators(v); err != nil {
		return fmt.Errorf("failed to register graph validators: %w", err)
	}

	return nil
}

// validateSemver validates that a string follows semantic versioning
// format (X.Y.Z where X, Y, Z are non-negative integers).
// validateSemver is a validator.Func that can be registered with
// the validator instance for use in struct tags.
func validateSemver(fl validator.FieldLevel) bool {
	// Simple semver validation - could be enhanced with a proper semver library.
	value := fl.Field().String()
	// Basic pattern: X.Y.Z where X, Y, Z are numbers.
	var major, minor, patch int
	n, err := fmt.Sscanf(value, "%d.%d.%d", &major, &minor, &patch)
	return err == nil && n == 3
}
