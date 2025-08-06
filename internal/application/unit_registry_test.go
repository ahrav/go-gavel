// Package application provides the core business logic and orchestration for
// the evaluation engine.
package application

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ahrav/go-gavel/internal/domain"
	"github.com/ahrav/go-gavel/internal/ports"
)

// mockLLMClient implements the ports.LLMClient interface for testing.
type mockLLMClient struct {
	model string
}

// Complete returns a mock response for the LLM completion.
func (m *mockLLMClient) Complete(ctx context.Context, prompt string, options map[string]any) (string, error) {
	return "mock response", nil
}

// CompleteWithUsage returns a mock response and token usage for the LLM completion.
func (m *mockLLMClient) CompleteWithUsage(ctx context.Context, prompt string, options map[string]any) (output string, tokensIn, tokensOut int, err error) {
	tokensIn, _ = m.EstimateTokens(prompt)
	output = "mock response"
	tokensOut, _ = m.EstimateTokens(output)
	return output, tokensIn, tokensOut, nil
}

// EstimateTokens provides a mock estimation of token count.
func (m *mockLLMClient) EstimateTokens(text string) (int, error) {
	return len(text), nil
}

// GetModel returns the model name of the mock client.
func (m *mockLLMClient) GetModel() string {
	return m.model
}

// testMockUnit implements the ports.Unit interface for testing custom factory registration.
type testMockUnit struct {
	name string
}

func (t *testMockUnit) Name() string { return t.name }

func (t *testMockUnit) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	return state, nil
}

func (t *testMockUnit) Validate() error { return nil }

func TestNewRegistry(t *testing.T) {
	t.Run("creates empty registry with LLM client", func(t *testing.T) {
		mockClient := &mockLLMClient{model: "test-model"}
		registry := NewRegistry(mockClient)

		assert.NotNil(t, registry)

		// Registry starts empty - no units registered by default
		supportedTypes := registry.GetSupportedTypes()
		assert.Empty(t, supportedTypes)
	})

	t.Run("creates empty registry with nil LLM client", func(t *testing.T) {
		registry := NewRegistry(nil)

		assert.NotNil(t, registry)

		// Registry starts empty - explicit registration required
		supportedTypes := registry.GetSupportedTypes()
		assert.Empty(t, supportedTypes)
	})
}

func TestRegistry_RegisterBuiltinUnits(t *testing.T) {
	t.Run("registers all builtin units explicitly", func(t *testing.T) {
		mockClient := &mockLLMClient{model: "test-model"}
		registry := NewRegistry(mockClient)

		// Initially empty
		assert.Empty(t, registry.GetSupportedTypes())

		// Register builtin units
		registry.RegisterBuiltinUnits()

		// All 8 core units should now be registered
		supportedTypes := registry.GetSupportedTypes()
		assert.Len(t, supportedTypes, 8)
		assert.Contains(t, supportedTypes, "score_judge")
		assert.Contains(t, supportedTypes, "answerer")
		assert.Contains(t, supportedTypes, "verification")
		assert.Contains(t, supportedTypes, "exact_match")
		assert.Contains(t, supportedTypes, "fuzzy_match")
		assert.Contains(t, supportedTypes, "arithmetic_mean")
		assert.Contains(t, supportedTypes, "max_pool")
		assert.Contains(t, supportedTypes, "median_pool")
	})
}

func TestRegistry_Register(t *testing.T) {
	t.Run("registers custom factory", func(t *testing.T) {
		registry := NewRegistry(nil)

		customFactory := func(id string, config map[string]any, llm ports.LLMClient) (ports.Unit, error) {
			return &testMockUnit{name: id}, nil
		}

		registry.Register("custom", customFactory)

		supportedTypes := registry.GetSupportedTypes()
		assert.Contains(t, supportedTypes, "custom")
	})

	t.Run("panics on duplicate registration", func(t *testing.T) {
		registry := NewRegistry(nil)

		customFactory := func(id string, config map[string]any, llm ports.LLMClient) (ports.Unit, error) {
			return &testMockUnit{name: id}, nil
		}

		registry.Register("custom", customFactory)

		assert.Panics(t, func() {
			registry.Register("custom", customFactory)
		})
	})
}

func TestRegistry_CreateUnit(t *testing.T) {
	mockClient := &mockLLMClient{model: "test-model"}

	t.Run("creates unit with registered factory", func(t *testing.T) {
		registry := NewRegistry(mockClient)

		customFactory := func(id string, config map[string]any, llm ports.LLMClient) (ports.Unit, error) {
			assert.Equal(t, mockClient, llm)
			return &testMockUnit{name: id}, nil
		}

		registry.Register("custom", customFactory)

		unit, err := registry.CreateUnit("custom", "test-unit", map[string]any{})
		assert.NoError(t, err)
		assert.NotNil(t, unit)
		assert.Equal(t, "test-unit", unit.Name())
	})

	t.Run("returns error for unknown unit type", func(t *testing.T) {
		registry := NewRegistry(mockClient)

		unit, err := registry.CreateUnit("unknown", "test-unit", map[string]any{})
		assert.Error(t, err)
		assert.Nil(t, unit)
		assert.Contains(t, err.Error(), "unknown unit type")
	})

	t.Run("returns error for empty ID", func(t *testing.T) {
		registry := NewRegistry(mockClient)

		customFactory := func(id string, config map[string]any, llm ports.LLMClient) (ports.Unit, error) {
			return &testMockUnit{name: id}, nil
		}

		registry.Register("custom", customFactory)

		unit, err := registry.CreateUnit("custom", "", map[string]any{})
		assert.Error(t, err)
		assert.Nil(t, unit)
		assert.Contains(t, err.Error(), "unit ID cannot be empty")
	})

	t.Run("passes configuration to factory", func(t *testing.T) {
		registry := NewRegistry(mockClient)

		expectedConfig := map[string]any{
			"key1": "value1",
			"key2": 42,
		}

		customFactory := func(id string, config map[string]any, llm ports.LLMClient) (ports.Unit, error) {
			assert.Equal(t, expectedConfig, config)
			return &testMockUnit{name: id}, nil
		}

		registry.Register("custom", customFactory)

		unit, err := registry.CreateUnit("custom", "test-unit", expectedConfig)
		assert.NoError(t, err)
		assert.NotNil(t, unit)
	})

	t.Run("factory error is propagated", func(t *testing.T) {
		registry := NewRegistry(mockClient)

		expectedErr := fmt.Errorf("factory error")
		customFactory := func(id string, config map[string]any, llm ports.LLMClient) (ports.Unit, error) {
			return nil, expectedErr
		}

		registry.Register("custom", customFactory)

		unit, err := registry.CreateUnit("custom", "test-unit", map[string]any{})
		assert.Error(t, err)
		assert.Nil(t, unit)
		assert.Equal(t, expectedErr, err)
	})
}

func TestRegistry_GetSupportedTypes(t *testing.T) {
	t.Run("returns all registered types", func(t *testing.T) {
		registry := NewRegistry(nil)

		// Register multiple factories
		for i := 0; i < 5; i++ {
			unitType := fmt.Sprintf("type%d", i)
			registry.Register(unitType, func(id string, config map[string]any, llm ports.LLMClient) (ports.Unit, error) {
				return &testMockUnit{name: id}, nil
			})
		}

		supportedTypes := registry.GetSupportedTypes()
		assert.Len(t, supportedTypes, 5)

		// Check all types are present (order not guaranteed)
		for i := 0; i < 5; i++ {
			assert.Contains(t, supportedTypes, fmt.Sprintf("type%d", i))
		}
	})

	t.Run("returns empty for no registered types", func(t *testing.T) {
		registry := NewRegistry(nil)
		supportedTypes := registry.GetSupportedTypes()
		assert.Empty(t, supportedTypes)
	})
}

func TestRegistry_ThreadSafety(t *testing.T) {
	t.Run("concurrent registration and creation", func(t *testing.T) {
		registry := NewRegistry(&mockLLMClient{model: "test"})

		var wg sync.WaitGroup
		numGoroutines := 10

		// Register different unit types concurrently
		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()

				unitType := fmt.Sprintf("type%d", idx)
				factory := func(id string, config map[string]any, llm ports.LLMClient) (ports.Unit, error) {
					return &testMockUnit{name: id}, nil
				}

				// This should not panic for unique types
				require.NotPanics(t, func() {
					registry.Register(unitType, factory)
				})
			}(i)
		}

		wg.Wait()

		// Create units concurrently
		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()

				unitType := fmt.Sprintf("type%d", idx)
				unitID := fmt.Sprintf("unit%d", idx)

				unit, err := registry.CreateUnit(unitType, unitID, map[string]any{})
				assert.NoError(t, err)
				assert.NotNil(t, unit)
				assert.Equal(t, unitID, unit.Name())
			}(i)
		}

		wg.Wait()

		// Verify all types are registered
		supportedTypes := registry.GetSupportedTypes()
		assert.Len(t, supportedTypes, numGoroutines)
	})
}
