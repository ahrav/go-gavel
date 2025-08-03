package middleware

import (
	"fmt"

	"github.com/ahrav/go-gavel/internal/ports"
)

// CreatePositionSwapMiddleware creates a PositionSwapMiddleware instance from
// configuration parameters. This factory is designed for use by the
// UnitRegistry to prevent import cycles.
func CreatePositionSwapMiddleware(
	id string,
	config map[string]any,
	unitFactory func(unitType, unitID string, unitConfig map[string]any) (ports.Unit, error),
) (ports.Unit, error) {
	wrappedUnitConfig, ok := config["wrapped_unit"]
	if !ok {
		return nil, fmt.Errorf("position_swap_wrapper requires 'wrapped_unit' configuration")
	}

	wrappedUnitMap, ok := wrappedUnitConfig.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("wrapped_unit must be a configuration object")
	}

	wrappedUnitType, ok := wrappedUnitMap["type"].(string)
	if !ok {
		return nil, fmt.Errorf("wrapped_unit must have a 'type' field")
	}

	wrappedUnitID, ok := wrappedUnitMap["id"].(string)
	if !ok {
		return nil, fmt.Errorf("wrapped_unit must have an 'id' field")
	}

	wrappedUnitParams, _ := wrappedUnitMap["params"].(map[string]any)
	if wrappedUnitParams == nil {
		wrappedUnitParams = make(map[string]any)
	}

	wrappedUnit, err := unitFactory(wrappedUnitType, wrappedUnitID, wrappedUnitParams)
	if err != nil {
		return nil, fmt.Errorf("failed to create wrapped unit: %w", err)
	}

	return NewPositionSwapMiddleware(wrappedUnit, id), nil
}

// RegisterPositionSwapMiddleware registers the position_swap_wrapper factory
// with a unit registry. This function should be called during application
// initialization to make the middleware available for YAML configuration.
func RegisterPositionSwapMiddleware(registry interface {
	RegisterMiddlewareFactory(
		unitType string,
		middlewareFactory func(
			id string,
			config map[string]any,
			unitFactory func(string, string, map[string]any) (ports.Unit, error),
		) (ports.Unit, error),
	) error
}) error {
	return registry.RegisterMiddlewareFactory("position_swap_wrapper", CreatePositionSwapMiddleware)
}
