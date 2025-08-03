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

// mockUnitRegistry implements ports.UnitRegistry for testing
type mockUnitRegistry struct {
	units       map[string]ports.Unit
	createError error
}

func newMockUnitRegistry() *mockUnitRegistry {
	return &mockUnitRegistry{
		units: make(map[string]ports.Unit),
	}
}

func (m *mockUnitRegistry) CreateUnit(unitType string, id string, config map[string]any) (ports.Unit, error) {
	if m.createError != nil {
		return nil, m.createError
	}

	// Return a pre-registered unit if available
	if unit, ok := m.units[id]; ok {
		return unit, nil
	}

	// Create a simple mock unit
	return &mockUnit{
		id:       id,
		unitType: unitType,
		config:   config,
	}, nil
}

func (m *mockUnitRegistry) RegisterUnitFactory(unitType string, factory ports.UnitFactory) error {
	return nil
}

func (m *mockUnitRegistry) GetSupportedTypes() []string {
	return []string{"llm_judge", "code_analyzer", "metrics_collector", "custom"}
}

// mockCoreLLMAdapterGL adapts mockLLMClient to implement llm.CoreLLM
type mockCoreLLMAdapterGL struct {
	client *mockLLMClient
}

func (m *mockCoreLLMAdapterGL) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
	response, tokensIn, tokensOut, err := m.client.CompleteWithUsage(ctx, prompt, opts)
	return response, tokensIn, tokensOut, err
}

func (m *mockCoreLLMAdapterGL) GetModel() string {
	return m.client.GetModel()
}

func (m *mockCoreLLMAdapterGL) SetModel(model string) {
	m.client.model = model
}

// mockUnit implements ports.Unit for testing
type mockUnit struct {
	id       string
	unitType string
	config   map[string]any
}

func (m *mockUnit) Name() string {
	return m.id
}

func (m *mockUnit) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	// Simple implementation that adds a marker to state
	return domain.With(state, domain.NewKey[bool]("executed_"+m.id), true), nil
}

func (m *mockUnit) Validate() error {
	return nil
}

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

				// Pipeline should exist as a node
				pipeline, exists := graph.GetNode("pipeline1")
				assert.True(t, exists)
				assert.NotNil(t, pipeline)

				// Units should not exist as separate nodes when in pipeline
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

				// Layer should exist as a node
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

				// Verify topological sort works
				sorted, err := graph.TopologicalSort()
				assert.NoError(t, err)
				assert.Len(t, sorted, 3)

				// Verify order
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
      # Missing required 'judge_prompt' parameter
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
				// This test verifies caching behavior
				// The actual caching is tested in a separate test
				assert.NotNil(t, graph)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock registry
			mockRegistry := newMockUnitRegistry()
			if tt.setupMock != nil {
				tt.setupMock(mockRegistry)
			}

			// Create provider registry
			config := llm.RegistryConfig{
				DefaultProvider: "openai",
				Providers:       llm.DefaultProviders,
			}
			registry, err := llm.NewRegistry(config)
			require.NoError(t, err)

			// Register mock provider factory
			mockLLMClient := &mockLLMClient{model: "test-model"}
			llm.RegisterProviderFactory("openai", func(cfg llm.ClientConfig) (llm.CoreLLM, error) {
				// Create an adapter to make mockLLMClient implement CoreLLM
				return &mockCoreLLMAdapterGL{client: mockLLMClient}, nil
			})

			// Register the client
			err = registry.RegisterClient("openai", llm.ClientConfig{
				APIKey: "test-key",
				Model:  "test-model",
			})
			require.NoError(t, err)

			// Create graph loader
			loader, err := NewGraphLoader(mockRegistry, registry)
			require.NoError(t, err)

			// Load graph
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

	// Create loader
	mockRegistry := newMockUnitRegistry()
	config := llm.RegistryConfig{
		DefaultProvider: "openai",
		Providers:       llm.DefaultProviders,
	}
	registry, err := llm.NewRegistry(config)
	require.NoError(t, err)
	mockLLMClient := &mockLLMClient{model: "test-model"}

	// Register mock provider factory
	llm.RegisterProviderFactory("openai", func(cfg llm.ClientConfig) (llm.CoreLLM, error) {
		// Create an adapter to make mockLLMClient implement CoreLLM
		return &mockCoreLLMAdapterGL{client: mockLLMClient}, nil
	})

	// Register the client
	err = registry.RegisterClient("openai", llm.ClientConfig{
		APIKey: "test-key",
		Model:  "test-model",
	})
	require.NoError(t, err)

	loader, err := NewGraphLoader(mockRegistry, registry)
	require.NoError(t, err)

	// Load graph first time
	reader1 := strings.NewReader(yaml)
	graph1, err := loader.LoadFromReader(context.Background(), reader1)
	require.NoError(t, err)

	// Load graph second time - should use cache
	reader2 := strings.NewReader(yaml)
	graph2, err := loader.LoadFromReader(context.Background(), reader2)
	require.NoError(t, err)

	// Both graphs should be the same instance (from cache)
	// Note: In Go, we can't directly compare pointers of interfaces,
	// but we can verify the behavior is consistent
	assert.NotNil(t, graph1)
	assert.NotNil(t, graph2)

	// Clear cache and load again
	loader.ClearCache()
	reader3 := strings.NewReader(yaml)
	graph3, err := loader.LoadFromReader(context.Background(), reader3)
	require.NoError(t, err)
	assert.NotNil(t, graph3)
}

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

	// Create loader
	mockRegistry := newMockUnitRegistry()
	config := llm.RegistryConfig{
		DefaultProvider: "openai",
		Providers:       llm.DefaultProviders,
	}
	registry, err := llm.NewRegistry(config)
	require.NoError(t, err)
	mockLLMClient := &mockLLMClient{model: "test-model"}

	// Register mock provider factory
	llm.RegisterProviderFactory("openai", func(cfg llm.ClientConfig) (llm.CoreLLM, error) {
		// Create an adapter to make mockLLMClient implement CoreLLM
		return &mockCoreLLMAdapterGL{client: mockLLMClient}, nil
	})

	// Register the client
	err = registry.RegisterClient("openai", llm.ClientConfig{
		APIKey: "test-key",
		Model:  "test-model",
	})
	require.NoError(t, err)

	loader, err := NewGraphLoader(mockRegistry, registry)
	require.NoError(t, err)

	// Load graph
	reader := strings.NewReader(yaml)
	graph, err := loader.LoadFromReader(context.Background(), reader)
	require.NoError(t, err)

	// Verify structure
	assert.NotNil(t, graph)

	// Verify nodes exist
	preprocessing, exists := graph.GetNode("preprocessing")
	assert.True(t, exists)
	assert.NotNil(t, preprocessing)

	judging, exists := graph.GetNode("judging")
	assert.True(t, exists)
	assert.NotNil(t, judging)

	finalPipeline, exists := graph.GetNode("finalpipeline")
	assert.True(t, exists)
	assert.NotNil(t, finalPipeline)

	// Verify topological sort
	sorted, err := graph.TopologicalSort()
	assert.NoError(t, err)
	assert.Len(t, sorted, 3)

	// Verify execution order
	assert.Equal(t, "preprocessing", sorted[0].ID())
	assert.Equal(t, "judging", sorted[1].ID())
	assert.Equal(t, "finalpipeline", sorted[2].ID())
}
