//go:build go1.18
// +build go1.18

// Package application provides the core business logic and orchestration for
// the evaluation engine.
package application

import (
	"context"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/ahrav/go-gavel/infrastructure/llm"
)

// mockCoreLLMAdapter adapts the mockLLMClient to implement the llm.CoreLLM interface.
// This adapter is used in fuzz tests to provide a mock LLM backend for the graph loader.
type mockCoreLLMAdapter struct {
	client *mockLLMClient
}

// DoRequest delegates the request to the underlying mockLLMClient.
func (m *mockCoreLLMAdapter) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
	response, tokensIn, tokensOut, err := m.client.CompleteWithUsage(ctx, prompt, opts)
	return response, tokensIn, tokensOut, err
}

// GetModel returns the model name from the underlying mockLLMClient.
func (m *mockCoreLLMAdapter) GetModel() string {
	return m.client.GetModel()
}

// SetModel sets the model name on the underlying mockLLMClient.
func (m *mockCoreLLMAdapter) SetModel(model string) {
	m.client.model = model
}

// FuzzGraphLoader_ParseYAML tests the YAML parsing logic of the GraphLoader with random inputs.
// It aims to uncover panics, crashes, or unexpected behavior when parsing a wide variety of
// potentially malformed or complex YAML strings.
func FuzzGraphLoader_ParseYAML(f *testing.F) {
	// Add a seed corpus with both valid and invalid YAML to guide the fuzzer.
	testcases := []string{
		// Valid minimal YAML.
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

  // Invalid YAML syntax.
  `version: "1.0.0
metadata:
  name: test"
units:
  - id: unit1`,

  // Missing required fields.
  `metadata:
  name: "test"
units: []
graph:
  edges: []`,

  // Invalid structure.
  `version: 1
metadata: "invalid"
units: "should be array"
graph: null`,

  // Malformed YAML.
  `version: "1.0.0"
metadata:
  name: [[[[[
units:
  - id: !!!
    type: @#$%^&*
    budget: {{{{{`,

  // Deeply nested structure.
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

 // Unicode and special characters.
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

 // Large numbers and other edge cases.
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

	// Register a mock provider factory to handle LLM client creation.
	mockLLMClient := &mockLLMClient{model: "test-model"}
	llm.RegisterProviderFactory("openai", func(cfg llm.ClientConfig) (llm.CoreLLM, error) {
		// Adapt the mockLLMClient to the CoreLLM interface.
		return &mockCoreLLMAdapter{client: mockLLMClient}, nil
	})

	// Register the client with the provider registry.
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
		// Test that YAML parsing does not panic.
		reader := strings.NewReader(yamlInput)
		graph, err := loader.LoadFromReader(context.Background(), reader)

		// If parsing succeeded, validate that graph operations do not panic.
		if err == nil && graph != nil {
			_ = graph.HasCycle()
			_, _ = graph.TopologicalSort()
		}

		// Clear the cache periodically to avoid memory issues during fuzzing.
		loader.ClearCache()
	})
}

// FuzzGraphLoader_Validation tests the semantic validation logic of the GraphLoader.
// It uses a corpus of YAML strings with common semantic errors, such as duplicate IDs,
// cyclic dependencies, and invalid references, to ensure the validator is robust.
func FuzzGraphLoader_Validation(f *testing.F) {
	// Add a seed corpus focusing on edge cases for validation.
	testcases := []string{
		// Duplicate unit IDs.
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

  // Invalid unit references in a pipeline.
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

  // Cyclic dependencies in the graph.
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

  // Invalid unit types.
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

  // Invalid parameter types.
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

	// Register a mock provider factory.
	mockLLMClient := &mockLLMClient{model: "test-model"}
	llm.RegisterProviderFactory("openai", func(cfg llm.ClientConfig) (llm.CoreLLM, error) {
		return &mockCoreLLMAdapter{client: mockLLMClient}, nil
	})

	// Register the client.
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
		// Test that validation does not panic.
		reader := strings.NewReader(yamlInput)
		_, _ = loader.LoadFromReader(context.Background(), reader)

		// Clear the cache periodically.
		loader.ClearCache()
	})
}

// FuzzValidateUnitParameters tests the validation of unit parameters.
// It fuzzes the unit type and a JSON string representing the parameters to ensure
// that the validation logic can handle a wide range of inputs without panicking.
func FuzzValidateUnitParameters(f *testing.F) {
	// Seed with various parameter combinations.
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
		// Parse the JSON parameters into a yaml.Node.
		var params map[string]interface{}
		err := yaml.Unmarshal([]byte(paramsJSON), &params)
		if err != nil {
			// If JSON is invalid, skip this iteration.
			return
		}

		// Convert the parameters to a yaml.Node.
		yamlBytes, err := yaml.Marshal(params)
		if err != nil {
			return
		}

		var node yaml.Node
		err = yaml.Unmarshal(yamlBytes, &node)
		if err != nil {
			return
		}

		// Test that validation does not panic.
		_ = ValidateUnitParameters(unitType, node)
	})
}

// FuzzDAGOperations tests the core operations of the DAG, such as cycle detection
// and topological sorting, with randomly generated graph structures. This ensures
// the robustness of the graph algorithms against various edge cases.
func FuzzDAGOperations(f *testing.F) {
	// Seed with various graph structures.
	testcases := []string{
		// Simple chain.
		`unit1,unit2;unit2,unit3`,
		// Diamond pattern.
		`A,B;A,C;B,D;C,D`,
		// Disconnected components.
		`A,B;C,D`,
		// Self-loop.
		`A,A`,
		// Complex graph.
		`A,B;B,C;C,D;A,D;B,D;A,C`,
		// Large chain.
		`A,B;B,C;C,D;D,E;E,F;F,G;G,H;H,I;I,J`,
	}

	for _, tc := range testcases {
		f.Add(tc)
	}

	f.Fuzz(func(t *testing.T, graphSpec string) {
		// Parse the graph specification to build a graph.
		graph := NewGraph()

		// Add all nodes first to ensure they exist before adding edges.
		nodes := make(map[string]bool)
		edges := strings.Split(graphSpec, ";")
		for _, edge := range edges {
			parts := strings.Split(edge, ",")
			if len(parts) == 2 {
				nodes[strings.TrimSpace(parts[0])] = true
				nodes[strings.TrimSpace(parts[1])] = true
			}
		}

		// Add nodes to the graph.
		for node := range nodes {
			if node != "" {
				exec := &mockExecutable{id: node}
				_ = graph.AddNode(exec)
			}
		}

		// Add edges to the graph.
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

		// Test that graph operations do not panic.
		_ = graph.HasCycle()
		_, _ = graph.TopologicalSort()

		// Test node retrieval.
		for node := range nodes {
			_, _ = graph.GetNode(node)
		}
	})
}
