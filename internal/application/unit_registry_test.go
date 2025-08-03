package application

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ahrav/go-gavel/internal/domain"
	"github.com/ahrav/go-gavel/internal/ports"
)

// mockLLMClient implements ports.LLMClient for testing.
type mockLLMClient struct {
	model string
}

func (m *mockLLMClient) Complete(ctx context.Context, prompt string, options map[string]any) (string, error) {
	return "mock response", nil
}

func (m *mockLLMClient) CompleteWithUsage(ctx context.Context, prompt string, options map[string]any) (output string, tokensIn, tokensOut int, err error) {
	tokensIn, _ = m.EstimateTokens(prompt)
	output = "mock response"
	tokensOut, _ = m.EstimateTokens(output)
	return output, tokensIn, tokensOut, nil
}

func (m *mockLLMClient) EstimateTokens(text string) (int, error) {
	return len(text), nil
}

func (m *mockLLMClient) GetModel() string {
	return m.model
}

// testMockUnit implements ports.Unit for testing custom factory registration.
type testMockUnit struct {
	name string
}

func (m *testMockUnit) Name() string {
	return m.name
}

func (m *testMockUnit) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	return state, nil
}

func (m *testMockUnit) Validate() error {
	return nil
}

func TestNewDefaultUnitRegistry(t *testing.T) {
	t.Run("creates registry with LLM client", func(t *testing.T) {
		mockClient := &mockLLMClient{model: "test-model"}
		registry := NewDefaultUnitRegistry(mockClient)

		assert.NotNil(t, registry)
		assert.NotNil(t, registry.factories)
		assert.Equal(t, mockClient, registry.llmClient)

		// Verify built-in factories are registered.
		supportedTypes := registry.GetSupportedTypes()
		assert.Contains(t, supportedTypes, "answerer")
		assert.Contains(t, supportedTypes, "score_judge")
		assert.Contains(t, supportedTypes, "max_pool")
		assert.Contains(t, supportedTypes, "mean_pool")
		assert.Contains(t, supportedTypes, "median_pool")
	})

	t.Run("creates registry with nil LLM client", func(t *testing.T) {
		registry := NewDefaultUnitRegistry(nil)

		assert.NotNil(t, registry)
		assert.NotNil(t, registry.factories)
		assert.Nil(t, registry.llmClient)

		// Built-in factories should still be registered.
		supportedTypes := registry.GetSupportedTypes()
		assert.Len(t, supportedTypes, 6) // answerer, score_judge, verification, max_pool, mean_pool, median_pool
	})
}

func TestCreateUnit_Success(t *testing.T) {
	mockClient := &mockLLMClient{model: "test-model"}
	registry := NewDefaultUnitRegistry(mockClient)

	tests := []struct {
		name     string
		unitType string
		unitID   string
		config   map[string]any
	}{
		{
			name:     "creates answerer unit",
			unitType: "answerer",
			unitID:   "test_answerer",
			config: map[string]any{
				"num_answers":     2,
				"prompt":          "Answer this: {{.Question}}",
				"temperature":     0.7,
				"max_tokens":      100,
				"timeout":         "30s",
				"max_concurrency": 5,
			},
		},
		{
			name:     "creates answerer unit with minimal config",
			unitType: "answerer",
			unitID:   "minimal_answerer",
			config: map[string]any{
				"num_answers":     1,
				"prompt":          "Simple prompt: {{.Question}}",
				"max_tokens":      50,
				"timeout":         "10s",
				"max_concurrency": 1,
			},
		},
		{
			name:     "creates unit with nil config",
			unitType: "max_pool",
			unitID:   "test_pool",
			config:   nil,
		},
		{
			name:     "creates unit with empty config",
			unitType: "max_pool",
			unitID:   "empty_pool",
			config:   map[string]any{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			unit, err := registry.CreateUnit(tt.unitType, tt.unitID, tt.config)
			require.NoError(t, err)
			assert.NotNil(t, unit)
			assert.Equal(t, tt.unitID, unit.Name())
		})
	}
}

func TestCreateUnit_Errors(t *testing.T) {
	mockClient := &mockLLMClient{model: "test-model"}
	registry := NewDefaultUnitRegistry(mockClient)

	tests := []struct {
		name          string
		unitType      string
		unitID        string
		config        map[string]any
		expectedError string
	}{
		{
			name:          "fails with unsupported unit type",
			unitType:      "unsupported",
			unitID:        "test_id",
			config:        map[string]any{},
			expectedError: "unsupported unit type",
		},
		{
			name:          "fails with empty unit ID",
			unitType:      "answerer",
			unitID:        "",
			config:        map[string]any{},
			expectedError: "unit ID cannot be empty",
		},
		{
			name:     "fails with invalid answerer config",
			unitType: "answerer",
			unitID:   "bad_answerer",
			config: map[string]any{
				"num_answers": -1, // Invalid
				"prompt":      "test",
			},
			expectedError: "failed to create unit",
		},
		{
			name:     "fails with missing required fields",
			unitType: "answerer",
			unitID:   "incomplete_answerer",
			config: map[string]any{
				"num_answers": 1,
				// Missing required fields: prompt, max_tokens, timeout, max_concurrency
			},
			expectedError: "failed to create unit",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			unit, err := registry.CreateUnit(tt.unitType, tt.unitID, tt.config)
			require.Error(t, err)
			assert.Nil(t, unit)
			assert.Contains(t, err.Error(), tt.expectedError)
		})
	}
}

func TestCreateUnit_YAMLConversionErrors(t *testing.T) {
	mockClient := &mockLLMClient{model: "test-model"}
	registry := NewDefaultUnitRegistry(mockClient)

	tests := []struct {
		name          string
		config        map[string]any
		expectedError string
	}{
		{
			name: "handles invalid duration format",
			config: map[string]any{
				"num_answers":     1,
				"prompt":          "Test prompt: {{.Question}}",
				"max_tokens":      100,
				"timeout":         "invalid-duration", // Should fail
				"max_concurrency": 1,
			},
			expectedError: "failed to unmarshal answerer config",
		},
		{
			name: "handles type mismatch in temperature",
			config: map[string]any{
				"num_answers":     1,
				"prompt":          "Test prompt: {{.Question}}",
				"temperature":     "not-a-number", // Should fail
				"max_tokens":      100,
				"timeout":         "30s",
				"max_concurrency": 1,
			},
			expectedError: "failed to unmarshal answerer config",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			unit, err := registry.CreateUnit("answerer", "test_id", tt.config)
			require.Error(t, err)
			assert.Nil(t, unit)
			assert.Contains(t, err.Error(), tt.expectedError)
		})
	}
}

func TestRegisterUnitFactory(t *testing.T) {
	mockClient := &mockLLMClient{model: "test-model"}
	registry := NewDefaultUnitRegistry(mockClient)

	t.Run("registers new factory successfully", func(t *testing.T) {
		customFactory := func(id string, config map[string]any) (ports.Unit, error) {
			return &testMockUnit{name: id}, nil
		}

		err := registry.RegisterUnitFactory("custom", customFactory)
		require.NoError(t, err)

		// Verify factory is registered.
		supportedTypes := registry.GetSupportedTypes()
		assert.Contains(t, supportedTypes, "custom")

		// Create unit with custom factory.
		unit, err := registry.CreateUnit("custom", "test_custom", nil)
		require.NoError(t, err)
		assert.Equal(t, "test_custom", unit.Name())
	})

	t.Run("overrides existing factory", func(t *testing.T) {
		// Register initial factory.
		factory1 := func(id string, config map[string]any) (ports.Unit, error) {
			return &testMockUnit{name: "factory1_" + id}, nil
		}
		err := registry.RegisterUnitFactory("override_test", factory1)
		require.NoError(t, err)

		// Create unit with first factory.
		unit1, err := registry.CreateUnit("override_test", "unit", nil)
		require.NoError(t, err)
		assert.Equal(t, "factory1_unit", unit1.Name())

		// Override with new factory.
		factory2 := func(id string, config map[string]any) (ports.Unit, error) {
			return &testMockUnit{name: "factory2_" + id}, nil
		}
		err = registry.RegisterUnitFactory("override_test", factory2)
		require.NoError(t, err)

		// Create unit with overridden factory.
		unit2, err := registry.CreateUnit("override_test", "unit", nil)
		require.NoError(t, err)
		assert.Equal(t, "factory2_unit", unit2.Name())
	})

	t.Run("fails with empty unit type", func(t *testing.T) {
		customFactory := func(id string, config map[string]any) (ports.Unit, error) {
			return &testMockUnit{name: id}, nil
		}

		err := registry.RegisterUnitFactory("", customFactory)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "unit type cannot be empty")
	})

	t.Run("fails with nil factory", func(t *testing.T) {
		err := registry.RegisterUnitFactory("nil_factory", nil)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "factory function cannot be nil")
	})
}

func TestGetSupportedTypes(t *testing.T) {
	mockClient := &mockLLMClient{model: "test-model"}
	registry := NewDefaultUnitRegistry(mockClient)

	t.Run("returns built-in types", func(t *testing.T) {
		types := registry.GetSupportedTypes()
		sort.Strings(types) // For consistent comparison

		expected := []string{"answerer", "max_pool", "mean_pool", "median_pool", "score_judge", "verification"}
		sort.Strings(expected)

		assert.Equal(t, expected, types)
	})

	t.Run("includes custom registered types", func(t *testing.T) {
		// Register custom type.
		customFactory := func(id string, config map[string]any) (ports.Unit, error) {
			return &testMockUnit{name: id}, nil
		}
		err := registry.RegisterUnitFactory("custom_type", customFactory)
		require.NoError(t, err)

		types := registry.GetSupportedTypes()
		assert.Contains(t, types, "custom_type")
		assert.Len(t, types, 7) // 6 built-in + 1 custom
	})
}

func TestSetLLMClient(t *testing.T) {
	initialClient := &mockLLMClient{model: "initial-model"}
	registry := NewDefaultUnitRegistry(initialClient)

	t.Run("updates LLM client", func(t *testing.T) {
		// Verify initial client.
		assert.Equal(t, initialClient, registry.GetLLMClient())

		// Update to new client.
		newClient := &mockLLMClient{model: "new-model"}
		registry.SetLLMClient(newClient)

		// Verify client was updated.
		assert.Equal(t, newClient, registry.GetLLMClient())
	})

	t.Run("re-registers built-in factories", func(t *testing.T) {
		// Register a custom factory.
		customFactory := func(id string, config map[string]any) (ports.Unit, error) {
			return &testMockUnit{name: id}, nil
		}
		err := registry.RegisterUnitFactory("custom", customFactory)
		require.NoError(t, err)

		// Update LLM client.
		newClient := &mockLLMClient{model: "updated-model"}
		registry.SetLLMClient(newClient)

		// Verify custom factory is still registered.
		types := registry.GetSupportedTypes()
		assert.Contains(t, types, "custom")

		// Verify built-in factories still work with new client.
		unit, err := registry.CreateUnit("answerer", "test_answerer", map[string]any{
			"num_answers":     1,
			"prompt":          "Test: {{.Question}}",
			"max_tokens":      100,
			"timeout":         "30s",
			"max_concurrency": 1,
		})
		require.NoError(t, err)
		assert.NotNil(t, unit)
	})

	t.Run("handles nil client", func(t *testing.T) {
		registry.SetLLMClient(nil)
		assert.Nil(t, registry.GetLLMClient())

		// Registry should still function.
		types := registry.GetSupportedTypes()
		assert.NotEmpty(t, types)
	})
}

func TestThreadSafety_CreateUnit(t *testing.T) {
	mockClient := &mockLLMClient{model: "test-model"}
	registry := NewDefaultUnitRegistry(mockClient)

	t.Run("concurrent CreateUnit calls", func(t *testing.T) {
		const numGoroutines = 10
		var wg sync.WaitGroup
		wg.Add(numGoroutines)

		errors := make(chan error, numGoroutines)

		for i := range numGoroutines {
			go func(id int) {
				defer wg.Done()

				unit, err := registry.CreateUnit("max_pool", fmt.Sprintf("unit_%d", id), map[string]any{
					"tie_breaker": "first",
				})

				if err != nil {
					errors <- err
					return
				}

				if unit.Name() != fmt.Sprintf("unit_%d", id) {
					errors <- fmt.Errorf("unexpected unit name: %s", unit.Name())
				}
			}(i)
		}

		wg.Wait()
		close(errors)

		// Check for errors.
		for err := range errors {
			t.Errorf("Concurrent error: %v", err)
		}
	})
}

func TestThreadSafety_RegisterAndCreate(t *testing.T) {
	mockClient := &mockLLMClient{model: "test-model"}
	registry := NewDefaultUnitRegistry(mockClient)

	t.Run("concurrent register and create", func(t *testing.T) {
		const numOperations = 20
		var wg sync.WaitGroup
		wg.Add(numOperations)

		errors := make(chan error, numOperations)

		for i := range numOperations {
			go func(id int) {
				defer wg.Done()

				if id%2 == 0 {
					// Register new factory.
					factory := func(unitID string, config map[string]any) (ports.Unit, error) {
						return &testMockUnit{name: unitID}, nil
					}
					err := registry.RegisterUnitFactory(fmt.Sprintf("type_%d", id), factory)
					if err != nil {
						errors <- err
					}
				} else {
					// Create unit.
					unitType := "max_pool"
					if id > 10 {
						unitType = fmt.Sprintf("type_%d", id-1) // Use previously registered type
					}

					_, err := registry.CreateUnit(unitType, fmt.Sprintf("unit_%d", id), nil)
					if err != nil && !isExpectedError(err) {
						errors <- err
					}
				}
			}(i)
		}

		wg.Wait()
		close(errors)

		// Check for unexpected errors.
		for err := range errors {
			t.Errorf("Concurrent error: %v", err)
		}
	})
}

func TestThreadSafety_SetLLMClient(t *testing.T) {
	mockClient := &mockLLMClient{model: "test-model"}
	registry := NewDefaultUnitRegistry(mockClient)

	t.Run("concurrent SetLLMClient and CreateUnit", func(t *testing.T) {
		const numOperations = 20
		var wg sync.WaitGroup
		wg.Add(numOperations)

		errs := make(chan error, numOperations)

		for i := range numOperations {
			go func(id int) {
				defer wg.Done()

				switch id % 3 {
				case 0:
					// Update LLM client.
					newClient := &mockLLMClient{model: fmt.Sprintf("model-%d", id)}
					registry.SetLLMClient(newClient)
				case 1:
					// Get LLM client.
					client := registry.GetLLMClient()
					if client == nil {
						errs <- errors.New("GetLLMClient returned nil")
					}
				default:
					// Create unit.
					unit, err := registry.CreateUnit("answerer", fmt.Sprintf("unit_%d", id), map[string]any{
						"num_answers":     1,
						"prompt":          "Test: {{.Question}}",
						"max_tokens":      100,
						"timeout":         "30s",
						"max_concurrency": 1,
					})
					if err != nil {
						errs <- err
						return
					}
					if unit.Name() != fmt.Sprintf("unit_%d", id) {
						errs <- fmt.Errorf("unexpected unit name: %s", unit.Name())
					}
				}
			}(i)
		}

		wg.Wait()
		close(errs)

		// Check for errors.
		for err := range errs {
			t.Errorf("Concurrent error: %v", err)
		}
	})
}

// TestRaceConditions should be run with -race flag.
func TestRaceConditions(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping race condition test in short mode")
	}

	mockClient := &mockLLMClient{model: "test-model"}
	registry := NewDefaultUnitRegistry(mockClient)

	t.Run("stress test all operations", func(t *testing.T) {
		const numGoroutines = 50
		done := make(chan bool)

		// Start multiple goroutines performing different operations.
		for i := range numGoroutines {
			go func(id int) {
				defer func() { done <- true }()

				operation := id % 5
				switch operation {
				case 0:
					// Register factory.
					factory := func(unitID string, config map[string]any) (ports.Unit, error) {
						return &testMockUnit{name: unitID}, nil
					}
					_ = registry.RegisterUnitFactory(fmt.Sprintf("stress_%d", id), factory)

				case 1:
					// Create unit.
					_, _ = registry.CreateUnit("max_pool", fmt.Sprintf("stress_unit_%d", id), nil)

				case 2:
					// Get supported types.
					_ = registry.GetSupportedTypes()

				case 3:
					// Set LLM client.
					newClient := &mockLLMClient{model: fmt.Sprintf("stress_model_%d", id)}
					registry.SetLLMClient(newClient)

				case 4:
					// Get LLM client.
					_ = registry.GetLLMClient()
				}

				// Add some randomness.
				time.Sleep(time.Microsecond * time.Duration(id))
			}(i)
		}

		// Wait for all goroutines to complete.
		for range numGoroutines {
			<-done
		}
	})
}

// Helper function to determine if an error is expected in concurrent scenarios.
func isExpectedError(err error) bool {
	// In concurrent scenarios, "unsupported unit type" is expected
	// if we try to use a type before it's registered.
	if err == nil {
		return false
	}
	errStr := err.Error()
	return strings.Contains(errStr, "unsupported unit type") ||
		strings.Contains(errStr, "failed to create unit")
}
