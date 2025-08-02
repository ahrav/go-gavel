package application

import (
	"fmt"
	"sync"

	"github.com/ahrav/go-gavel/infrastructure/units"
	"github.com/ahrav/go-gavel/internal/ports"
)

// Verify interface compliance at compile time.
var _ ports.UnitRegistry = (*DefaultUnitRegistry)(nil)

// DefaultUnitRegistry implements the UnitRegistry interface providing
// a factory for creating evaluation units based on type and configuration.
// It supports dynamic registration of unit factories and manages dependencies
// like LLM clients for units that require them.
type DefaultUnitRegistry struct {
	// factories maps unit type strings to their factory functions.
	factories map[string]ports.UnitFactory
	// mu protects concurrent access to the factories map.
	mu sync.RWMutex
	// llmClient is the default LLM client injected into units that need it.
	llmClient ports.LLMClient
}

// NewDefaultUnitRegistry creates a new unit registry with standard unit types
// pre-registered and a default LLM client for units that require it.
// The registry comes with built-in support for answerer, score_judge, and max_pool units.
func NewDefaultUnitRegistry(llmClient ports.LLMClient) *DefaultUnitRegistry {
	registry := &DefaultUnitRegistry{
		factories: make(map[string]ports.UnitFactory),
		llmClient: llmClient,
	}

	// Register built-in unit types.
	registry.registerBuiltinFactories()

	return registry
}

// registerBuiltinFactories registers the standard unit types provided
// by the evaluation framework.
// This includes answerer, score_judge, max_pool, and verification units.
func (r *DefaultUnitRegistry) registerBuiltinFactories() {
	// Capture the current LLM client to avoid data races.
	client := r.llmClient

	// Register AnswererUnit factory.
	r.factories["answerer"] = func(id string, config map[string]any) (ports.Unit, error) {
		// Inject LLM client into config.
		config["llm_client"] = client
		unit, err := units.CreateAnswererUnit(id, config)
		if err != nil {
			return nil, err
		}
		return unit, nil
	}

	// Register ScoreJudgeUnit factory.
	r.factories["score_judge"] = func(id string, config map[string]any) (ports.Unit, error) {
		// Inject LLM client into config.
		config["llm_client"] = client
		unit, err := units.CreateScoreJudgeUnit(id, config)
		if err != nil {
			return nil, err
		}
		return unit, nil
	}

	// Register VerificationUnit factory.
	r.factories["verification"] = func(id string, config map[string]any) (ports.Unit, error) {
		// Inject LLM client into config.
		config["llm_client"] = client
		unit, err := units.CreateVerificationUnit(id, config)
		if err != nil {
			return nil, err
		}
		return unit, nil
	}

	// Register MaxPoolUnit factory.
	maxPoolFactory := func(id string, config map[string]any) (ports.Unit, error) {
		unit, err := units.CreateMaxPoolUnit(id, config)
		if err != nil {
			return nil, err
		}
		return unit, nil
	}

	// Register both "max_pool" and "mean_pool" (for backwards compatibility)
	r.factories["max_pool"] = maxPoolFactory
	r.factories["mean_pool"] = maxPoolFactory // Alias for backwards compatibility

}

// CreateUnit creates a new unit instance based on the provided type,
// identifier, and configuration.
// It looks up the appropriate factory function and delegates unit creation,
// injecting any required dependencies like LLM clients.
func (r *DefaultUnitRegistry) CreateUnit(
	unitType string,
	id string,
	config map[string]any,
) (ports.Unit, error) {
	r.mu.RLock()
	factory, exists := r.factories[unitType]
	r.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("unsupported unit type: %s", unitType)
	}

	if id == "" {
		return nil, fmt.Errorf("unit ID cannot be empty")
	}

	if config == nil {
		config = make(map[string]any)
	}

	unit, err := factory(id, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create unit %s of type %s: %w", id, unitType, err)
	}

	return unit, nil
}

// RegisterUnitFactory registers a new factory function for a specific unit type.
// This allows extending the registry with custom unit types at runtime.
// The factory function is responsible for creating and configuring unit instances.
func (r *DefaultUnitRegistry) RegisterUnitFactory(
	unitType string,
	factory ports.UnitFactory,
) error {
	if unitType == "" {
		return fmt.Errorf("unit type cannot be empty")
	}

	if factory == nil {
		return fmt.Errorf("factory function cannot be nil")
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	r.factories[unitType] = factory
	return nil
}

// GetSupportedTypes returns a list of all registered unit types.
// This is useful for validation, documentation, and introspection purposes.
func (r *DefaultUnitRegistry) GetSupportedTypes() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	types := make([]string, 0, len(r.factories))
	for unitType := range r.factories {
		types = append(types, unitType)
	}

	return types
}

// SetLLMClient updates the default LLM client used by units that require it.
// This is useful for switching between different LLM providers or configurations.
func (r *DefaultUnitRegistry) SetLLMClient(client ports.LLMClient) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.llmClient = client

	// Re-register built-in factories with the new client.
	r.registerBuiltinFactories()
}

// GetLLMClient returns the current default LLM client.
// This is useful for debugging and testing purposes.
func (r *DefaultUnitRegistry) GetLLMClient() ports.LLMClient {
	r.mu.RLock()
	defer r.mu.RUnlock()

	return r.llmClient
}
