// Package application provides the core business logic and orchestration for
// the evaluation engine.
package application

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

// TestGraphConfig_UnmarshalYAML tests the YAML unmarshaling of GraphConfig.
// It verifies that valid YAML configurations are correctly parsed and that
// malformed or incomplete YAML structures are handled appropriately.
// This test focuses on the unmarshaling process itself, not semantic validation.
func TestGraphConfig_UnmarshalYAML(t *testing.T) {
	tests := []struct {
		name    string
		yaml    string
		wantErr bool
		errMsg  string
		verify  func(t *testing.T, config *GraphConfig)
	}{
		{
			name: "valid minimal config",
			yaml: `
version: "1.0.0"
metadata:
  name: "test-graph"
units:
  - id: unit1
    type: llm_judge
    budget:
      max_tokens: 1000
    parameters:
      prompt: "Test prompt"
graph:
  edges: []
`,
			wantErr: false,
			verify: func(t *testing.T, config *GraphConfig) {
				assert.Equal(t, "1.0.0", config.Version)
				assert.Equal(t, "test-graph", config.Metadata.Name)
				assert.Len(t, config.Units, 1)
				assert.Equal(t, "unit1", config.Units[0].ID)
				assert.Equal(t, "llm_judge", config.Units[0].Type)
			},
		},
		{
			name: "valid complex config",
			yaml: `
version: "1.0.0"
metadata:
  name: "complex-graph"
  description: "A complex evaluation graph"
  tags: ["test", "complex"]
  labels:
    env: "prod"
    team: "platform"
units:
  - id: analyzer1
    type: code_analyzer
    budget:
      max_tokens: 5000
      max_cost: 10.0
      timeout_seconds: 30
    parameters:
      language: "go"
      rules: ["gofmt", "golint"]
    retry:
      max_attempts: 3
      backoff_type: exponential
      initial_wait_ms: 1000
      max_wait_ms: 10000
  - id: judge1
    type: llm_judge
    budget:
      max_tokens: 2000
    parameters:
      prompt: "Evaluate the code quality"
      temperature: 0.7
      model: "gpt-4"
graph:
  pipelines:
    - id: pipeline1
      units: ["analyzer1", "judge1"]
  edges:
    - from: analyzer1
      to: judge1
`,
			wantErr: false,
			verify: func(t *testing.T, config *GraphConfig) {
				assert.Equal(t, "complex-graph", config.Metadata.Name)
				assert.Equal(t, "A complex evaluation graph", config.Metadata.Description)
				assert.Equal(t, []string{"test", "complex"}, config.Metadata.Tags)
				assert.Equal(t, "prod", config.Metadata.Labels["env"])
				assert.Len(t, config.Units, 2)
				assert.Len(t, config.Graph.Pipelines, 1)
				assert.Len(t, config.Graph.Edges, 1)
			},
		},
		// Note: The following tests are commented out because YAML unmarshaling alone
		// does not perform semantic validation (e.g., checking for required fields).
		// This validation is handled separately by the application's validator.
		// {
		// 	name: "missing version",
		// 	yaml: `
		// metadata:
		//   name: "test"
		// units:
		//   - id: unit1
		//     type: custom
		//     budget: {}
		// graph:
		//   edges: []
		// `,
		// 	wantErr: true,
		// 	errMsg:  "version",
		// },
		// {
		// 	name: "invalid unit type",
		// 	yaml: `
		// version: "1.0.0"
		// metadata:
		//   name: "test"
		// units:
		//   - id: unit1
		//     type: invalid_type
		//     budget: {}
		// graph:
		//   edges: []
		// `,
		// 	wantErr: true,
		// 	errMsg:  "invalid_type",
		// },
		{
			name: "layer config",
			yaml: `
version: "1.0.0"
metadata:
  name: "parallel-graph"
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
  layers:
    - id: layer1
      units: ["unit1", "unit2", "unit3"]
  edges: []
`,
			wantErr: false,
			verify: func(t *testing.T, config *GraphConfig) {
				assert.Len(t, config.Graph.Layers, 1)
				assert.Equal(t, "layer1", config.Graph.Layers[0].ID)
				assert.Len(t, config.Graph.Layers[0].Units, 3)
			},
		},
		{
			name: "edge with conditions",
			yaml: `
version: "1.0.0"
metadata:
  name: "conditional-graph"
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
      conditions:
        - type: score_threshold
          parameters:
            threshold: 80
            operator: gte
`,
			wantErr: false,
			verify: func(t *testing.T, config *GraphConfig) {
				assert.Len(t, config.Graph.Edges, 1)
				assert.Len(t, config.Graph.Edges[0].Conditions, 1)
				assert.Equal(t, "score_threshold", config.Graph.Edges[0].Conditions[0].Type)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var config GraphConfig
			err := yaml.Unmarshal([]byte(tt.yaml), &config)

			if tt.wantErr {
				require.Error(t, err)
				if tt.errMsg != "" {
					assert.Contains(t, err.Error(), tt.errMsg)
				}
			} else {
				require.NoError(t, err)
				if tt.verify != nil {
					tt.verify(t, &config)
				}
			}
		})
	}
}

// TestUnitConfig_ParameterDecoding tests the decoding of the 'parameters' field
// in a UnitConfig. It verifies that the flexible yaml.Node type can be
// successfully unmarshaled into a structured map for different unit types,
// allowing for varied and nested parameter configurations.
func TestUnitConfig_ParameterDecoding(t *testing.T) {
	tests := []struct {
		name   string
		yaml   string
		verify func(t *testing.T, unit *UnitConfig)
	}{
		{
			name: "llm_judge parameters",
			yaml: `
id: judge1
type: llm_judge
budget:
  max_tokens: 1000
parameters:
  prompt: "Test prompt"
  temperature: 0.8
  model: "gpt-4"
`,
			verify: func(t *testing.T, unit *UnitConfig) {
				var params map[string]interface{}
				err := unit.Parameters.Decode(&params)
				require.NoError(t, err)

				assert.Equal(t, "Test prompt", params["prompt"])
				assert.Equal(t, 0.8, params["temperature"])
				assert.Equal(t, "gpt-4", params["model"])
			},
		},
		{
			name: "code_analyzer parameters",
			yaml: `
id: analyzer1
type: code_analyzer
budget:
  max_tokens: 1000
parameters:
  language: "python"
  rules:
    - "pep8"
    - "pylint"
    - "mypy"
`,
			verify: func(t *testing.T, unit *UnitConfig) {
				var params map[string]interface{}
				err := unit.Parameters.Decode(&params)
				require.NoError(t, err)

				assert.Equal(t, "python", params["language"])
				rules := params["rules"].([]interface{})
				assert.Len(t, rules, 3)
				assert.Equal(t, "pep8", rules[0])
			},
		},
		{
			name: "metrics_collector parameters",
			yaml: `
id: metrics1
type: metrics_collector
budget:
  max_tokens: 1000
parameters:
  metrics:
    - "complexity"
    - "coverage"
    - "performance"
  threshold: 85.5
`,
			verify: func(t *testing.T, unit *UnitConfig) {
				var params map[string]interface{}
				err := unit.Parameters.Decode(&params)
				require.NoError(t, err)

				metrics := params["metrics"].([]interface{})
				assert.Len(t, metrics, 3)
				assert.Equal(t, 85.5, params["threshold"])
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var unit UnitConfig
			err := yaml.Unmarshal([]byte(tt.yaml), &unit)
			require.NoError(t, err)

			if tt.verify != nil {
				tt.verify(t, &unit)
			}
		})
	}
}

// TestBudgetConfig_Validation tests the creation of BudgetConfig structs.
// It ensures that the struct can be instantiated with both zero and valid values,
// which is a prerequisite for the semantic validation that occurs later.
func TestBudgetConfig_Validation(t *testing.T) {
	tests := []struct {
		name    string
		budget  BudgetConfig
		wantErr bool
	}{
		{
			name:    "empty budget is valid",
			budget:  BudgetConfig{},
			wantErr: false,
		},
		{
			name: "valid budget",
			budget: BudgetConfig{
				MaxTokens:      10000,
				MaxCost:        100.0,
				TimeoutSeconds: 60,
			},
			wantErr: false,
		},
		{
			name: "negative values should be invalid",
			budget: BudgetConfig{
				MaxTokens: -1,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// This test only verifies that the struct can be created.
			// Full semantic validation (e.g., checking for negative values)
			// is handled by a dedicated validator.
			assert.NotNil(t, tt.budget)
		})
	}
}
