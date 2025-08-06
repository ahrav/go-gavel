// Package application provides the core business logic and orchestration for
// the evaluation engine.
package application

import (
	"fmt"
	"sync"

	"github.com/ahrav/go-gavel/infrastructure/units"
	"github.com/ahrav/go-gavel/internal/ports"
)

// FactoryFunc creates a unit from configuration and dependencies.
// The LLM client may be nil for units that don't require language model capabilities.
// Factories should validate their configuration and return descriptive errors
// for invalid inputs.
type FactoryFunc func(id string, config map[string]any, llm ports.LLMClient) (ports.Unit, error)

// Registry manages unit factories and dependencies.
// It provides thread-safe registration and creation of evaluation units,
// implementing the ports.UnitRegistry interface for the GraphLoader.
// The zero value is not usable; use NewRegistry to create instances.
type Registry struct {
	mu        sync.RWMutex
	factories map[string]FactoryFunc
	llmClient ports.LLMClient
}

// NewRegistry creates a registry with optional LLM client.
// Pass nil for llmClient if only non-LLM units will be used.
// The registry starts empty; call RegisterBuiltinUnits to add core units
// or use Register to add custom factories.
func NewRegistry(llmClient ports.LLMClient) *Registry {
	return &Registry{
		factories: make(map[string]FactoryFunc),
		llmClient: llmClient,
	}
}

// Register adds a factory for a unit type.
// Panics if unitType is already registered.
// This panic is intentional - duplicate registrations indicate
// a programming error that should fail fast during initialization.
func (r *Registry) Register(unitType string, factory FactoryFunc) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.factories[unitType]; exists {
		panic(fmt.Sprintf("unit type %q already registered", unitType))
	}

	r.factories[unitType] = factory
}

// CreateUnit creates a unit instance using the registered factory.
// Returns an error if the unit type is unknown or the ID is empty.
// The factory receives the registry's LLM client, which may be nil.
// Configuration validation is delegated to the factory implementation.
func (r *Registry) CreateUnit(unitType string, id string, config map[string]any) (ports.Unit, error) {
	if id == "" {
		return nil, fmt.Errorf("unit ID cannot be empty")
	}

	r.mu.RLock()
	factory, exists := r.factories[unitType]
	llm := r.llmClient
	r.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("unknown unit type: %s", unitType)
	}

	return factory(id, config, llm)
}

// GetSupportedTypes returns all registered unit types.
// The returned slice is a copy and can be safely modified.
func (r *Registry) GetSupportedTypes() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	types := make([]string, 0, len(r.factories))
	for unitType := range r.factories {
		types = append(types, unitType)
	}
	return types
}

// RegisterBuiltinUnits registers all built-in evaluation units.
// Registers: answerer, score_judge, verification, exact_match,
// fuzzy_match, arithmetic_mean, max_pool, and median_pool.
// Call this once during initialization to enable core functionality.
func (r *Registry) RegisterBuiltinUnits() {
	r.Register("answerer", units.NewAnswererFromConfig)
	r.Register("score_judge", units.NewScoreJudgeFromConfig)
	r.Register("verification", units.NewVerificationFromConfig)
	r.Register("exact_match", units.NewExactMatchFromConfig)
	r.Register("fuzzy_match", units.NewFuzzyMatchFromConfig)
	r.Register("arithmetic_mean", units.NewArithmeticMeanFromConfig)
	r.Register("max_pool", units.NewMaxPoolFromConfig)
	r.Register("median_pool", units.NewMedianPoolFromConfig)
}
