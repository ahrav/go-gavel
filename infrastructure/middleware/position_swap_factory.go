package middleware

import (
	"fmt"

	"github.com/ahrav/go-gavel/internal/ports"
)

// NewPositionSwapFromConfig creates a PositionSwapMiddleware from configuration.
// This follows the same pattern as other units in the new simplified registry.
// The middleware wraps another unit that must be created first.
func NewPositionSwapFromConfig(id string, config map[string]any, llm ports.LLMClient) (ports.Unit, error) {
	// Note: Since middleware wraps other units, the wrapped unit must be passed
	// as an already-created Unit instance in the config, not as a configuration.
	// This is typically handled by the graph loader which creates units in dependency order.
	wrappedUnit, ok := config["wrapped_unit"].(ports.Unit)
	if !ok {
		return nil, fmt.Errorf("position_swap_wrapper requires 'wrapped_unit' as a Unit instance")
	}

	return NewPositionSwapMiddleware(wrappedUnit, id), nil
}
