package application

import (
	"context"

	"github.com/ahrav/go-gavel/internal/domain"
	"github.com/ahrav/go-gavel/internal/ports"
)

// UnitAdapter is an adapter that wraps a ports.Unit to implement the
// ports.Executable interface, enabling units to participate in graph
// execution workflows.
// Use UnitAdapter when you need to integrate evaluation units into
// pipelines, layers, or graphs that expect the Executable interface.
type UnitAdapter struct {
	// unit is the underlying evaluation unit that performs the actual
	// work when Execute is called.
	unit ports.Unit
	// id is the unique identifier for this adapter within the graph
	// scope, used for referencing and error reporting.
	id string
}

// NewUnitAdapter creates a new adapter that wraps a ports.Unit to
// implement the ports.Executable interface, enabling the unit to
// participate in graph-based execution workflows.
// NewUnitAdapter preserves the unit's functionality while providing
// the interface expected by pipelines, layers, and graphs.
func NewUnitAdapter(unit ports.Unit, id string) *UnitAdapter {
	return &UnitAdapter{
		unit: unit,
		id:   id,
	}
}

// Execute delegates to the underlying unit's Execute method,
// providing transparent pass-through of context, state, and results.
// Execute maintains the same semantics as the wrapped unit,
// including error handling and context cancellation support.
func (ua *UnitAdapter) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	return ua.unit.Execute(ctx, state)
}

// ID returns the unique string identifier for this adapter.
// The ID is used for referencing in graph topologies, error reporting,
// and debugging, and remains constant throughout the adapter's lifetime.
func (ua *UnitAdapter) ID() string { return ua.id }
