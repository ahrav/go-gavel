//go:build go1.18
// +build go1.18

package application

import (
	"context"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/ahrav/go-gavel/infrastructure/llm"
)

// mockCoreLLMAdapter adapts mockLLMClient to implement llm.CoreLLM
type mockCoreLLMAdapter struct {
	client *mockLLMClient
}

func (m *mockCoreLLMAdapter) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
	response, tokensIn, tokensOut, err := m.client.CompleteWithUsage(ctx, prompt, opts)
	return response, tokensIn, tokensOut, err
}

func (m *mockCoreLLMAdapter) GetModel() string {
	return m.client.GetModel()
}

func (m *mockCoreLLMAdapter) SetModel(model string) {
	m.client.model = model
}

// FuzzGraphLoader_ParseYAML tests the YAML parsing with random inputs
func FuzzGraphLoader_ParseYAML(f *testing.F) {
	// Add seed corpus with valid and invalid YAML
	testcases := []string{
		// Valid minimal YAML
		`version: "1.0.0"
metadata:
  name: "test"
units:
  - id: unit1
    type: custom
    budget: {}
    parameters: {}
graph:
  edges: []`,

		// Invalid YAML syntax
		`version: "1.0.0
metadata:
  name: test"
units:
  - id: unit1`,

		// Missing required fields
		`metadata:
  name: "test"
units: []
graph:
  edges: []`,

		// Invalid structure
		`version: 1
metadata: "invalid"
units: "should be array"
graph: null`,

		// Malformed YAML
		`version: "1.0.0"
metadata:
  name: [[[[[
units:
  - id: !!!
    type: @#$%^&*
    budget: {{{{{`,

		// Deeply nested structure
		`version: "1.0.0"
metadata:
  name: "nested"
  labels:
    a:
      b:
        c:
          d:
            e: "deep"
units:
  - id: unit1
    type: custom
    budget: {}
    parameters:
      nested:
        deeply:
          very:
            much:
              so: "value"
graph:
  edges: []`,

		// Unicode and special characters
		`version: "1.0.0"
metadata:
  name: "测试 🚀 тест"
  description: "Multi-line\nstring with\ttabs"
units:
  - id: unit1
    type: custom
    budget: {}
    parameters: {}
graph:
  edges: []`,

		// Large numbers and edge cases
		`version: "999999999.0.0"
metadata:
  name: "x"
units:
  - id: unit1
    type: custom
    budget:
      max_tokens: 99999999999999999999
      max_cost: 1.7976931348623157e+308
      timeout_seconds: -1
    parameters: {}
graph:
  edges: []`,
	}

	for _, tc := range testcases {
		f.Add(tc)
	}

	mockRegistry := newMockUnitRegistry()
	config := llm.RegistryConfig{
		DefaultProvider: "openai",
		Providers:       llm.DefaultProviders,
	}
	registry, err := llm.NewRegistry(config)
	if err != nil {
		f.Fatal(err)
	}

	// Register mock provider factory
	mockLLMClient := &mockLLMClient{model: "test-model"}
	llm.RegisterProviderFactory("openai", func(cfg llm.ClientConfig) (llm.CoreLLM, error) {
		// Create an adapter to make mockLLMClient implement CoreLLM
		return &mockCoreLLMAdapter{client: mockLLMClient}, nil
	})

	// Register the client
	if err := registry.RegisterClient("openai", llm.ClientConfig{
		APIKey: "test-key",
		Model:  "test-model",
	}); err != nil {
		f.Fatal(err)
	}
	loader, err := NewGraphLoader(mockRegistry, registry)
	if err != nil {
		f.Fatal(err)
	}

	f.Fuzz(func(t *testing.T, yamlInput string) {
		// Test YAML parsing - should not panic
		reader := strings.NewReader(yamlInput)
		graph, err := loader.LoadFromReader(context.Background(), reader)

		// If parsing succeeded, validate the graph
		if err == nil && graph != nil {
			// Verify graph operations don't panic
			_ = graph.HasCycle()
			_, _ = graph.TopologicalSort()
		}

		// Clear cache periodically to avoid memory issues
		loader.ClearCache()
	})
}

// FuzzGraphLoader_Validation tests validation with various inputs
func FuzzGraphLoader_Validation(f *testing.F) {
	// Add seed corpus focusing on edge cases for validation
	testcases := []string{
		// Duplicate unit IDs
		`version: "1.0.0"
metadata:
  name: "duplicate"
units:
  - id: unit1
    type: custom
    budget: {}
    parameters: {}
  - id: unit1
    type: custom
    budget: {}
    parameters: {}
graph:
  edges: []`,

		// Invalid unit references
		`version: "1.0.0"
metadata:
  name: "invalid-ref"
units:
  - id: unit1
    type: custom
    budget: {}
    parameters: {}
graph:
  pipelines:
    - id: pipeline1
      units: ["unit1", "nonexistent"]
  edges: []`,

		// Cyclic dependencies
		`version: "1.0.0"
metadata:
  name: "cycle"
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
      to: unit1`,

		// Invalid unit types
		`version: "1.0.0"
metadata:
  name: "invalid-type"
units:
  - id: unit1
    type: "unknown_type_!@#$%"
    budget: {}
    parameters: {}
graph:
  edges: []`,

		// Invalid parameter types
		`version: "1.0.0"
metadata:
  name: "invalid-params"
units:
  - id: judge1
    type: llm_judge
    budget:
      max_tokens: "not a number"
    parameters:
      prompt: 123
      temperature: "high"
graph:
  edges: []`,
	}

	for _, tc := range testcases {
		f.Add(tc)
	}

	mockRegistry := newMockUnitRegistry()
	config := llm.RegistryConfig{
		DefaultProvider: "openai",
		Providers:       llm.DefaultProviders,
	}
	registry, err := llm.NewRegistry(config)
	if err != nil {
		f.Fatal(err)
	}

	// Register mock provider factory
	mockLLMClient := &mockLLMClient{model: "test-model"}
	llm.RegisterProviderFactory("openai", func(cfg llm.ClientConfig) (llm.CoreLLM, error) {
		// Create an adapter to make mockLLMClient implement CoreLLM
		return &mockCoreLLMAdapter{client: mockLLMClient}, nil
	})

	// Register the client
	if err := registry.RegisterClient("openai", llm.ClientConfig{
		APIKey: "test-key",
		Model:  "test-model",
	}); err != nil {
		f.Fatal(err)
	}
	loader, err := NewGraphLoader(mockRegistry, registry)
	if err != nil {
		f.Fatal(err)
	}

	f.Fuzz(func(t *testing.T, yamlInput string) {
		// Test validation - should not panic
		reader := strings.NewReader(yamlInput)
		_, _ = loader.LoadFromReader(context.Background(), reader)

		// Clear cache periodically
		loader.ClearCache()
	})
}

// FuzzValidateUnitParameters tests unit parameter validation
func FuzzValidateUnitParameters(f *testing.F) {
	// Seed with various parameter combinations
	testcases := []struct {
		unitType string
		params   string
	}{
		{"llm_judge", `{"prompt": "test", "temperature": 0.7, "model": "gpt-4"}`},
		{"llm_judge", `{"temperature": 3.0}`},
		{"llm_judge", `{"prompt": ""}`},
		{"llm_judge", `{"prompt": null}`},
		{"llm_judge", `{}`},
		{"code_analyzer", `{"language": "go", "rules": ["fmt", "vet"]}`},
		{"code_analyzer", `{"language": "unknown"}`},
		{"code_analyzer", `{"language": 123}`},
		{"code_analyzer", `{"rules": "not-array"}`},
		{"metrics_collector", `{"metrics": ["complexity", "coverage"]}`},
		{"metrics_collector", `{"metrics": []}`},
		{"metrics_collector", `{"metrics": ["unknown"]}`},
		{"custom", `{"any": "value", "nested": {"deep": true}}`},
		{"unknown_type", `{"some": "params"}`},
	}

	for _, tc := range testcases {
		f.Add(tc.unitType, tc.params)
	}

	f.Fuzz(func(t *testing.T, unitType string, paramsJSON string) {
		// Parse JSON params into yaml.Node
		var params map[string]interface{}
		err := yaml.Unmarshal([]byte(paramsJSON), &params)
		if err != nil {
			// If JSON is invalid, skip this iteration
			return
		}

		// Convert to yaml.Node
		yamlBytes, err := yaml.Marshal(params)
		if err != nil {
			return
		}

		var node yaml.Node
		err = yaml.Unmarshal(yamlBytes, &node)
		if err != nil {
			return
		}

		// Test validation - should not panic
		_ = ValidateUnitParameters(unitType, node)
	})
}

// FuzzDAGOperations tests DAG operations with random graphs
func FuzzDAGOperations(f *testing.F) {
	// Seed with various graph structures
	testcases := []string{
		// Simple chain
		`unit1,unit2;unit2,unit3`,
		// Diamond
		`A,B;A,C;B,D;C,D`,
		// Disconnected
		`A,B;C,D`,
		// Self-loop
		`A,A`,
		// Complex
		`A,B;B,C;C,D;A,D;B,D;A,C`,
		// Large chain
		`A,B;B,C;C,D;D,E;E,F;F,G;G,H;H,I;I,J`,
	}

	for _, tc := range testcases {
		f.Add(tc)
	}

	f.Fuzz(func(t *testing.T, graphSpec string) {
		// Parse graph specification
		graph := NewGraph()

		// Add all nodes first
		nodes := make(map[string]bool)
		edges := strings.Split(graphSpec, ";")
		for _, edge := range edges {
			parts := strings.Split(edge, ",")
			if len(parts) == 2 {
				nodes[strings.TrimSpace(parts[0])] = true
				nodes[strings.TrimSpace(parts[1])] = true
			}
		}

		// Add nodes to graph
		for node := range nodes {
			if node != "" {
				exec := &mockExecutable{id: node}
				_ = graph.AddNode(exec)
			}
		}

		// Add edges
		for _, edge := range edges {
			parts := strings.Split(edge, ",")
			if len(parts) == 2 {
				from := strings.TrimSpace(parts[0])
				to := strings.TrimSpace(parts[1])
				if from != "" && to != "" {
					_ = graph.AddEdge(from, to)
				}
			}
		}

		// Test operations - should not panic
		_ = graph.HasCycle()
		_, _ = graph.TopologicalSort()

		// Test node retrieval
		for node := range nodes {
			_, _ = graph.GetNode(node)
		}
	})
}
