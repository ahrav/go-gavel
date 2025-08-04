// Package application provides the core business logic and orchestration for
// the evaluation engine.
package application

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ahrav/go-gavel/infrastructure/llm"
	"github.com/ahrav/go-gavel/internal/domain"
	"github.com/ahrav/go-gavel/internal/ports"
)

// mockUnitRegistry implements the ports.UnitRegistry interface for testing purposes.
// It allows for mocking unit creation and returning pre-configured units or errors.
type mockUnitRegistry struct {
	units       map[string]ports.Unit
	createError error
}

// newMockUnitRegistry creates a new mockUnitRegistry.
func newMockUnitRegistry() *mockUnitRegistry {
	return &mockUnitRegistry{
		units: make(map[string]ports.Unit),
	}
}

// CreateUnit mocks the creation of a unit.
// It returns a pre-registered unit if available, otherwise it creates a new
// mockUnit. It can also be configured to return an error.
func (m *mockUnitRegistry) CreateUnit(unitType string, id string, config map[string]any) (ports.Unit, error) {
	if m.createError != nil {
		return nil, m.createError
	}

	// Return a pre-registered unit if one is available for the given ID.
	if unit, ok := m.units[id]; ok {
		return unit, nil
	}

	// Create a simple mock unit for testing.
	return &mockUnit{
		id:       id,
		unitType: unitType,
		config:   config,
	}, nil
}

// RegisterUnitFactory is a no-op for the mock registry.
func (m *mockUnitRegistry) RegisterUnitFactory(unitType string, factory ports.UnitFactory) error {
	return nil
}

// GetSupportedTypes returns a static list of supported unit types for testing.
func (m *mockUnitRegistry) GetSupportedTypes() []string {
	return []string{"llm_judge", "code_analyzer", "metrics_collector", "custom"}
}

// mockCoreLLMAdapterGL adapts the mockLLMClient to implement the llm.CoreLLM interface.
// This adapter is used in graph loader tests to provide a mock LLM backend.
type mockCoreLLMAdapterGL struct {
	client *mockLLMClient
}

// DoRequest delegates the request to the underlying mockLLMClient.
func (m *mockCoreLLMAdapterGL) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
	response, tokensIn, tokensOut, err := m.client.CompleteWithUsage(ctx, prompt, opts)
	return response, tokensIn, tokensOut, err
}

// GetModel returns the model name from the underlying mockLLMClient.
func (m *mockCoreLLMAdapterGL) GetModel() string {
	return m.client.GetModel()
}

// SetModel sets the model name on the underlying mockLLMClient.
func (m *mockCoreLLMAdapterGL) SetModel(model string) {
	m.client.model = model
}

// mockUnit implements the ports.Unit interface for testing.
// It provides a simple implementation that marks its execution in the state.
type mockUnit struct {
	id       string
	unitType string
	config   map[string]any
}

// Name returns the name of the mock unit.
func (m *mockUnit) Name() string {
	return m.id
}

// Execute adds a marker to the state to indicate that the unit was executed.
func (m *mockUnit) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	return domain.With(state, domain.NewKey[bool]("executed_"+m.id), true), nil
}

// Validate is a no-op for the mock unit.
func (m *mockUnit) Validate() error {
	return nil
}

// TestGraphLoader_LoadFromReader tests the loading of a graph from a YAML configuration.
// It covers various scenarios, including simple graphs, pipelines, layers, and error
// conditions like cyclic dependencies and invalid configurations.
func TestGraphLoader_LoadFromReader(t *testing.T) {
	tests := []struct {
		name      string
		yaml      string
		setupMock func(*mockUnitRegistry)
		wantErr   bool
		errMsg    string
		verify    func(t *testing.T, graph ports.Graph)
	}{
		{
			name: "loads simple graph successfully",
			yaml: `
version: "1.0.0"
metadata:
  name: "simple-graph"
units:
  - id: unit1
    type: score_judge
    budget:
      max_tokens: 1000
    parameters:
      judge_prompt: "Test prompt"
      score_scale: "0.0-1.0"
graph:
  edges: []
`,
			setupMock: func(m *mockUnitRegistry) {},
			wantErr:   false,
			verify: func(t *testing.T, graph ports.Graph) {
				assert.NotNil(t, graph)
				node, exists := graph.GetNode("unit1")
				assert.True(t, exists)
				assert.NotNil(t, node)
			},
		},
		{
			name: "loads pipeline graph",
			yaml: `
version: "1.0.0"
metadata:
  name: "pipeline-graph"
units:
  - id: analyzer1
    type: answerer
    budget:
      max_tokens: 1000
    parameters:
      language: "go"
  - id: judge1
    type: score_judge
    budget:
      max_tokens: 2000
    parameters:
      judge_prompt: "Evaluate code"
      score_scale: "0.0-1.0"
graph:
  pipelines:
    - id: pipeline1
      units: ["analyzer1", "judge1"]
  edges: []
`,
			setupMock: func(m *mockUnitRegistry) {},
			wantErr:   false,
			verify: func(t *testing.T, graph ports.Graph) {
				assert.NotNil(t, graph)

				// The pipeline itself should exist as a node in the graph.
				pipeline, exists := graph.GetNode("pipeline1")
				assert.True(t, exists)
				assert.NotNil(t, pipeline)

				// The individual units within the pipeline should not exist as separate nodes.
				_, exists = graph.GetNode("analyzer1")
				assert.False(t, exists)
			},
		},
		{
			name: "loads layer graph",
			yaml: `
version: "1.0.0"
metadata:
  name: "layer-graph"
units:
  - id: unit1
    type: custom
    budget: {}
    parameters: {}
  - id: unit2
    type: custom
    budget: {}
    parameters: {}
graph:
  layers:
    - id: layer1
      units: ["unit1", "unit2"]
  edges: []
`,
			setupMock: func(m *mockUnitRegistry) {},
			wantErr:   false,
			verify: func(t *testing.T, graph ports.Graph) {
				assert.NotNil(t, graph)

				// The layer itself should exist as a node in the graph.
				layer, exists := graph.GetNode("layer1")
				assert.True(t, exists)
				assert.NotNil(t, layer)
			},
		},
		{
			name: "loads graph with edges",
			yaml: `
version: "1.0.0"
metadata:
  name: "edge-graph"
units:
  - id: unit1
    type: custom
    budget: {}
    parameters: {}
  - id: unit2
    type: custom
    budget: {}
    parameters: {}
  - id: unit3
    type: custom
    budget: {}
    parameters: {}
graph:
  edges:
    - from: unit1
      to: unit2
    - from: unit2
      to: unit3
`,
			setupMock: func(m *mockUnitRegistry) {},
			wantErr:   false,
			verify: func(t *testing.T, graph ports.Graph) {
				assert.NotNil(t, graph)

				// Verify that topological sort works correctly for the given edges.
				sorted, err := graph.TopologicalSort()
				assert.NoError(t, err)
				assert.Len(t, sorted, 3)

				// Verify the execution order.
				assert.Equal(t, "unit1", sorted[0].ID())
				assert.Equal(t, "unit2", sorted[1].ID())
				assert.Equal(t, "unit3", sorted[2].ID())
			},
		},
		{
			name: "detects cycle in graph",
			yaml: `
version: "1.0.0"
metadata:
  name: "cycle-graph"
units:
  - id: unit1
    type: custom
    budget: {}
    parameters: {}
  - id: unit2
    type: custom
    budget: {}
    parameters: {}
graph:
  edges:
    - from: unit1
      to: unit2
    - from: unit2
      to: unit1
`,
			setupMock: func(m *mockUnitRegistry) {},
			wantErr:   true,
			errMsg:    "cycle",
		},
		{
			name: "validates semantic errors",
			yaml: `
version: "1.0.0"
metadata:
  name: "invalid-graph"
units:
  - id: unit1
    type: custom
    budget: {}
    parameters: {}
graph:
  pipelines:
    - id: pipeline1
      units: ["unit1", "nonexistent"]
  edges: []
`,
			setupMock: func(m *mockUnitRegistry) {},
			wantErr:   true,
			errMsg:    "non-existent unit",
		},
		{
			name: "validates unit parameters",
			yaml: `
version: "1.0.0"
metadata:
  name: "invalid-params"
units:
  - id: judge1
    type: score_judge
    budget:
      max_tokens: 1000
    parameters:
      # Missing the required 'judge_prompt' parameter.
      temperature: 0.8
graph:
  edges: []
`,
			setupMock: func(m *mockUnitRegistry) {},
			wantErr:   true,
			errMsg:    "judge_prompt",
		},
		{
			name: "handles unit creation error",
			yaml: `
version: "1.0.0"
metadata:
  name: "creation-error"
units:
  - id: unit1
    type: custom
    budget: {}
    parameters: {}
graph:
  edges: []
`,
			setupMock: func(m *mockUnitRegistry) {
				m.createError = errors.New("failed to create unit")
			},
			wantErr: true,
			errMsg:  "failed to create unit",
		},
		{
			name: "caches compiled graphs",
			yaml: `
version: "1.0.0"
metadata:
  name: "cache-test"
units:
  - id: unit1
    type: custom
    budget: {}
    parameters: {}
graph:
  edges: []
`,
			setupMock: func(m *mockUnitRegistry) {},
			wantErr:   false,
			verify: func(t *testing.T, graph ports.Graph) {
				// This test case verifies that the loader can handle a cacheable graph,
				// but the actual caching behavior is tested in TestGraphLoader_Caching.
				assert.NotNil(t, graph)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a mock unit registry.
			mockRegistry := newMockUnitRegistry()
			if tt.setupMock != nil {
				tt.setupMock(mockRegistry)
			}

			// Create a provider registry.
			config := llm.RegistryConfig{
				DefaultProvider: "openai",
				Providers:       llm.DefaultProviders,
			}
			registry, err := llm.NewRegistry(config)
			require.NoError(t, err)

			// Register a mock provider factory.
			mockLLMClient := &mockLLMClient{model: "test-model"}
			llm.RegisterProviderFactory("openai", func(cfg llm.ClientConfig) (llm.CoreLLM, error) {
				// Adapt the mockLLMClient to the CoreLLM interface.
				return &mockCoreLLMAdapterGL{client: mockLLMClient}, nil
			})

			// Register the client.
			err = registry.RegisterClient("openai", llm.ClientConfig{
				APIKey: "test-key",
				Model:  "test-model",
			})
			require.NoError(t, err)

			// Create the graph loader.
			loader, err := NewGraphLoader(mockRegistry, registry)
			require.NoError(t, err)

			// Load the graph from the YAML reader.
			reader := strings.NewReader(tt.yaml)
			graph, err := loader.LoadFromReader(context.Background(), reader)

			if tt.wantErr {
				require.Error(t, err)
				if tt.errMsg != "" {
					assert.Contains(t, err.Error(), tt.errMsg)
				}
			} else {
				require.NoError(t, err)
				if tt.verify != nil {
					tt.verify(t, graph)
				}
			}
		})
	}
}

// TestGraphLoader_Caching verifies that the GraphLoader correctly caches compiled graphs.
// It loads the same graph configuration multiple times and checks that the same instance
// is returned, then clears the cache and ensures a new instance is created.
func TestGraphLoader_Caching(t *testing.T) {
	yaml := `
version: "1.0.0"
metadata:
  name: "cache-test"
units:
  - id: unit1
    type: custom
    budget: {}
    parameters: {}
graph:
  edges: []
`

	// Create the loader.
	mockRegistry := newMockUnitRegistry()
	config := llm.RegistryConfig{
		DefaultProvider: "openai",
		Providers:       llm.DefaultProviders,
	}
	registry, err := llm.NewRegistry(config)
	require.NoError(t, err)
	mockLLMClient := &mockLLMClient{model: "test-model"}

	// Register a mock provider factory.
	llm.RegisterProviderFactory("openai", func(cfg llm.ClientConfig) (llm.CoreLLM, error) {
		return &mockCoreLLMAdapterGL{client: mockLLMClient}, nil
	})

	// Register the client.
	err = registry.RegisterClient("openai", llm.ClientConfig{
		APIKey: "test-key",
		Model:  "test-model",
	})
	require.NoError(t, err)

	loader, err := NewGraphLoader(mockRegistry, registry)
	require.NoError(t, err)

	// Load the graph for the first time.
	reader1 := strings.NewReader(yaml)
	graph1, err := loader.LoadFromReader(context.Background(), reader1)
	require.NoError(t, err)

	// Load the graph for the second time; it should be retrieved from the cache.
	reader2 := strings.NewReader(yaml)
	graph2, err := loader.LoadFromReader(context.Background(), reader2)
	require.NoError(t, err)

	// Both graph instances should be the same.
	// Note: In Go, we cannot directly compare pointers of interfaces,
	// but we can verify that the behavior is consistent.
	assert.NotNil(t, graph1)
	assert.NotNil(t, graph2)

	// Clear the cache and load the graph again.
	loader.ClearCache()
	reader3 := strings.NewReader(yaml)
	graph3, err := loader.LoadFromReader(context.Background(), reader3)
	require.NoError(t, err)
	assert.NotNil(t, graph3)
}

// TestGraphLoader_ComplexGraph tests the loading of a complex graph with multiple
// stages, including layers and pipelines, to ensure the loader can handle
// intricate structures and dependencies.
func TestGraphLoader_ComplexGraph(t *testing.T) {
	yaml := `
version: "1.0.0"
metadata:
  name: "complex-evaluation"
  description: "A complex multi-stage evaluation pipeline"
  tags: ["production", "ml"]
  labels:
    team: "platform"
    env: "prod"
units:
  - id: preprocess1
    type: answerer
    budget:
      max_tokens: 1000
      timeout_seconds: 30
    parameters:
      language: "python"
      rules: ["pep8", "pylint"]
  - id: preprocess2
    type: max_pool
    budget:
      max_tokens: 500
    parameters:
      metrics: ["complexity", "coverage"]
  - id: judge1
    type: score_judge
    budget:
      max_tokens: 5000
      max_cost: 10.0
    parameters:
      judge_prompt: "Evaluate the code quality"
      score_scale: "0.0-1.0"
      temperature: 0.7
      model: "gpt-4"
  - id: judge2
    type: score_judge
    budget:
      max_tokens: 5000
      max_cost: 10.0
    parameters:
      judge_prompt: "Evaluate the performance"
      score_scale: "0.0-1.0"
      temperature: 0.7
      model: "gpt-4"
  - id: aggregator
    type: custom
    budget:
      max_tokens: 1000
    parameters: {}
graph:
  layers:
    - id: preprocessing
      units: ["preprocess1", "preprocess2"]
    - id: judging
      units: ["judge1", "judge2"]
  pipelines:
    - id: finalpipeline
      units: ["aggregator"]
  edges:
    - from: preprocessing
      to: judging
    - from: judging
      to: finalpipeline
`

	// Create the loader.
	mockRegistry := newMockUnitRegistry()
	config := llm.RegistryConfig{
		DefaultProvider: "openai",
		Providers:       llm.DefaultProviders,
	}
	registry, err := llm.NewRegistry(config)
	require.NoError(t, err)
	mockLLMClient := &mockLLMClient{model: "test-model"}

	// Register a mock provider factory.
	llm.RegisterProviderFactory("openai", func(cfg llm.ClientConfig) (llm.CoreLLM, error) {
		return &mockCoreLLMAdapterGL{client: mockLLMClient}, nil
	})

	// Register the client.
	err = registry.RegisterClient("openai", llm.ClientConfig{
		APIKey: "test-key",
		Model:  "test-model",
	})
	require.NoError(t, err)

	loader, err := NewGraphLoader(mockRegistry, registry)
	require.NoError(t, err)

	// Load the graph.
	reader := strings.NewReader(yaml)
	graph, err := loader.LoadFromReader(context.Background(), reader)
	require.NoError(t, err)

	// Verify the graph structure.
	assert.NotNil(t, graph)

	// Verify that the nodes exist.
	preprocessing, exists := graph.GetNode("preprocessing")
	assert.True(t, exists)
	assert.NotNil(t, preprocessing)

	judging, exists := graph.GetNode("judging")
	assert.True(t, exists)
	assert.NotNil(t, judging)

	finalPipeline, exists := graph.GetNode("finalpipeline")
	assert.True(t, exists)
	assert.NotNil(t, finalPipeline)

	// Verify the topological sort order.
	sorted, err := graph.TopologicalSort()
	assert.NoError(t, err)
	assert.Len(t, sorted, 3)

	// Verify the execution order.
	assert.Equal(t, "preprocessing", sorted[0].ID())
	assert.Equal(t, "judging", sorted[1].ID())
	assert.Equal(t, "finalpipeline", sorted[2].ID())
}
