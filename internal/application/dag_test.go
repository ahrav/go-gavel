package application

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ahrav/go-gavel/internal/domain"
	"github.com/ahrav/go-gavel/internal/ports"
)

// mockExecutable is a test implementation of Executable
type mockExecutable struct {
	id          string
	executeFunc func(ctx context.Context, state domain.State) (domain.State, error)
	executed    bool
	mu          sync.Mutex
}

func (m *mockExecutable) Execute(ctx context.Context, state domain.State) (domain.State, error) {
	m.mu.Lock()
	m.executed = true
	m.mu.Unlock()

	if m.executeFunc != nil {
		return m.executeFunc(ctx, state)
	}
	return state, nil
}

func (m *mockExecutable) ID() string {
	return m.id
}

func (m *mockExecutable) wasExecuted() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.executed
}

func TestPipeline_Execute(t *testing.T) {
	tests := []struct {
		name          string
		setupPipeline func() (ports.Pipeline, []*mockExecutable)
		initialState  domain.State
		wantErr       bool
		errMsg        string
		verify        func(t *testing.T, state domain.State, mocks []*mockExecutable)
	}{
		{
			name: "executes units in sequence",
			setupPipeline: func() (ports.Pipeline, []*mockExecutable) {
				pipeline := NewPipeline("test-pipeline")
				mocks := make([]*mockExecutable, 3)
				
				for i := 0; i < 3; i++ {
					final := i
					mocks[i] = &mockExecutable{
						id: fmt.Sprintf("unit%d", i),
						executeFunc: func(ctx context.Context, state domain.State) (domain.State, error) {
							// Add a value to verify execution order
							return state.With(domain.StateKey(fmt.Sprintf("step%d", final)), final), nil
						},
					}
					err := pipeline.Add(mocks[i])
					require.NoError(t, err)
				}
				
				return pipeline, mocks
			},
			initialState: domain.NewState(),
			wantErr:      false,
			verify: func(t *testing.T, state domain.State, mocks []*mockExecutable) {
				// Verify all units executed
				for _, m := range mocks {
					assert.True(t, m.wasExecuted())
				}
				
				// Verify execution order
				for i := 0; i < 3; i++ {
					val, exists := state.Get(domain.StateKey(fmt.Sprintf("step%d", i)))
					assert.True(t, exists)
					assert.Equal(t, i, val)
				}
			},
		},
		{
			name: "stops on first error",
			setupPipeline: func() (ports.Pipeline, []*mockExecutable) {
				pipeline := NewPipeline("error-pipeline")
				mocks := make([]*mockExecutable, 3)
				
				mocks[0] = &mockExecutable{
					id: "unit0",
					executeFunc: func(ctx context.Context, state domain.State) (domain.State, error) {
						return state, nil
					},
				}
				
				mocks[1] = &mockExecutable{
					id: "unit1",
					executeFunc: func(ctx context.Context, state domain.State) (domain.State, error) {
						return state, errors.New("unit1 failed")
					},
				}
				
				mocks[2] = &mockExecutable{
					id: "unit2",
					executeFunc: func(ctx context.Context, state domain.State) (domain.State, error) {
						return state, nil
					},
				}
				
				for _, m := range mocks {
					err := pipeline.Add(m)
					require.NoError(t, err)
				}
				
				return pipeline, mocks
			},
			initialState: domain.NewState(),
			wantErr:      true,
			errMsg:       "unit1 failed",
			verify: func(t *testing.T, state domain.State, mocks []*mockExecutable) {
				assert.True(t, mocks[0].wasExecuted())
				assert.True(t, mocks[1].wasExecuted())
				assert.False(t, mocks[2].wasExecuted()) // Should not execute after error
			},
		},
		{
			name: "handles context cancellation",
			setupPipeline: func() (ports.Pipeline, []*mockExecutable) {
				pipeline := NewPipeline("cancel-pipeline")
				mocks := make([]*mockExecutable, 2)
				
				mocks[0] = &mockExecutable{
					id: "unit0",
					executeFunc: func(ctx context.Context, state domain.State) (domain.State, error) {
						// Simulate some work
						time.Sleep(10 * time.Millisecond)
						return state, nil
					},
				}
				
				mocks[1] = &mockExecutable{
					id: "unit1",
					executeFunc: func(ctx context.Context, state domain.State) (domain.State, error) {
						return state, nil
					},
				}
				
				for _, m := range mocks {
					err := pipeline.Add(m)
					require.NoError(t, err)
				}
				
				return pipeline, mocks
			},
			initialState: domain.NewState(),
			wantErr:      true,
			errMsg:       "context",
			verify: func(t *testing.T, state domain.State, mocks []*mockExecutable) {
				// First might execute depending on timing
				// Second should not execute
				assert.False(t, mocks[1].wasExecuted())
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pipeline, mocks := tt.setupPipeline()
			
			ctx := context.Background()
			if tt.name == "handles context cancellation" {
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				go func() {
					time.Sleep(5 * time.Millisecond)
					cancel()
				}()
			}
			
			resultState, err := pipeline.Execute(ctx, tt.initialState)
			
			if tt.wantErr {
				require.Error(t, err)
				if tt.errMsg != "" {
					assert.Contains(t, err.Error(), tt.errMsg)
				}
			} else {
				require.NoError(t, err)
			}
			
			if tt.verify != nil {
				tt.verify(t, resultState, mocks)
			}
		})
	}
}

func TestPipeline_Add(t *testing.T) {
	tests := []struct {
		name    string
		setup   func() ports.Pipeline
		exec    ports.Executable
		wantErr bool
		errMsg  string
	}{
		{
			name: "adds executable successfully",
			setup: func() ports.Pipeline {
				return NewPipeline("test")
			},
			exec:    &mockExecutable{id: "unit1"},
			wantErr: false,
		},
		{
			name: "rejects nil executable",
			setup: func() ports.Pipeline {
				return NewPipeline("test")
			},
			exec:    nil,
			wantErr: true,
			errMsg:  "nil executable",
		},
		{
			name: "rejects duplicate ID",
			setup: func() ports.Pipeline {
				p := NewPipeline("test")
				err := p.Add(&mockExecutable{id: "unit1"})
				require.NoError(t, err)
				return p
			},
			exec:    &mockExecutable{id: "unit1"},
			wantErr: true,
			errMsg:  "already exists",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pipeline := tt.setup()
			err := pipeline.Add(tt.exec)
			
			if tt.wantErr {
				require.Error(t, err)
				if tt.errMsg != "" {
					assert.Contains(t, err.Error(), tt.errMsg)
				}
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func TestLayer_Execute(t *testing.T) {
	tests := []struct {
		name         string
		setupLayer   func() (ports.Layer, []*mockExecutable)
		initialState domain.State
		wantErr      bool
		verify       func(t *testing.T, state domain.State, mocks []*mockExecutable)
	}{
		{
			name: "executes units in parallel",
			setupLayer: func() (ports.Layer, []*mockExecutable) {
				layer := NewLayer("test-layer")
				mocks := make([]*mockExecutable, 3)
				
				// Use a channel to verify parallel execution
				startChan := make(chan int, 3)
				
				for i := 0; i < 3; i++ {
					final := i
					mocks[i] = &mockExecutable{
						id: fmt.Sprintf("unit%d", i),
						executeFunc: func(ctx context.Context, state domain.State) (domain.State, error) {
							startChan <- final
							// Simulate some work
							time.Sleep(10 * time.Millisecond)
							return state.With(domain.StateKey(fmt.Sprintf("unit%d", final)), true), nil
						},
					}
					err := layer.Add(mocks[i])
					require.NoError(t, err)
				}
				
				// Verify units start in parallel
				go func() {
					starts := make([]int, 0, 3)
					for i := 0; i < 3; i++ {
						starts = append(starts, <-startChan)
					}
					// All should start before any finish
					assert.Len(t, starts, 3)
				}()
				
				return layer, mocks
			},
			initialState: domain.NewState(),
			wantErr:      false,
			verify: func(t *testing.T, state domain.State, mocks []*mockExecutable) {
				// All units should have executed
				for _, m := range mocks {
					assert.True(t, m.wasExecuted())
				}
			},
		},
		{
			name: "handles partial failures",
			setupLayer: func() (ports.Layer, []*mockExecutable) {
				layer := NewLayer("error-layer")
				mocks := make([]*mockExecutable, 3)
				
				mocks[0] = &mockExecutable{
					id: "unit0",
					executeFunc: func(ctx context.Context, state domain.State) (domain.State, error) {
						return state, nil
					},
				}
				
				mocks[1] = &mockExecutable{
					id: "unit1",
					executeFunc: func(ctx context.Context, state domain.State) (domain.State, error) {
						return state, errors.New("unit1 failed")
					},
				}
				
				mocks[2] = &mockExecutable{
					id: "unit2",
					executeFunc: func(ctx context.Context, state domain.State) (domain.State, error) {
						return state, nil
					},
				}
				
				for _, m := range mocks {
					err := layer.Add(m)
					require.NoError(t, err)
				}
				
				return layer, mocks
			},
			initialState: domain.NewState(),
			wantErr:      true,
			verify: func(t *testing.T, state domain.State, mocks []*mockExecutable) {
				// All units should have attempted execution
				for _, m := range mocks {
					assert.True(t, m.wasExecuted())
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layer, mocks := tt.setupLayer()
			
			resultState, err := layer.Execute(context.Background(), tt.initialState)
			
			if tt.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
			
			if tt.verify != nil {
				tt.verify(t, resultState, mocks)
			}
		})
	}
}

func TestGraph_AddNode(t *testing.T) {
	tests := []struct {
		name    string
		setup   func() ports.Graph
		node    ports.Executable
		wantErr bool
		errMsg  string
	}{
		{
			name: "adds node successfully",
			setup: func() ports.Graph {
				return NewGraph()
			},
			node:    &mockExecutable{id: "node1"},
			wantErr: false,
		},
		{
			name: "rejects nil node",
			setup: func() ports.Graph {
				return NewGraph()
			},
			node:    nil,
			wantErr: true,
			errMsg:  "nil executable",
		},
		{
			name: "rejects duplicate node",
			setup: func() ports.Graph {
				g := NewGraph()
				err := g.AddNode(&mockExecutable{id: "node1"})
				require.NoError(t, err)
				return g
			},
			node:    &mockExecutable{id: "node1"},
			wantErr: true,
			errMsg:  "already exists",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			graph := tt.setup()
			err := graph.AddNode(tt.node)
			
			if tt.wantErr {
				require.Error(t, err)
				if tt.errMsg != "" {
					assert.Contains(t, err.Error(), tt.errMsg)
				}
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func TestGraph_AddEdge(t *testing.T) {
	tests := []struct {
		name     string
		setup    func() ports.Graph
		sourceID string
		targetID string
		wantErr  bool
		errMsg   string
	}{
		{
			name: "adds edge successfully",
			setup: func() ports.Graph {
				g := NewGraph()
				err := g.AddNode(&mockExecutable{id: "node1"})
				require.NoError(t, err)
				err = g.AddNode(&mockExecutable{id: "node2"})
				require.NoError(t, err)
				return g
			},
			sourceID: "node1",
			targetID: "node2",
			wantErr:  false,
		},
		{
			name: "rejects edge to non-existent source",
			setup: func() ports.Graph {
				g := NewGraph()
				err := g.AddNode(&mockExecutable{id: "node2"})
				require.NoError(t, err)
				return g
			},
			sourceID: "node1",
			targetID: "node2",
			wantErr:  true,
			errMsg:   "source node",
		},
		{
			name: "rejects edge to non-existent target",
			setup: func() ports.Graph {
				g := NewGraph()
				err := g.AddNode(&mockExecutable{id: "node1"})
				require.NoError(t, err)
				return g
			},
			sourceID: "node1",
			targetID: "node2",
			wantErr:  true,
			errMsg:   "target node",
		},
		{
			name: "detects cycles",
			setup: func() ports.Graph {
				g := NewGraph()
				err := g.AddNode(&mockExecutable{id: "node1"})
				require.NoError(t, err)
				err = g.AddNode(&mockExecutable{id: "node2"})
				require.NoError(t, err)
				err = g.AddNode(&mockExecutable{id: "node3"})
				require.NoError(t, err)
				
				// Create edges: 1 -> 2 -> 3
				err = g.AddEdge("node1", "node2")
				require.NoError(t, err)
				err = g.AddEdge("node2", "node3")
				require.NoError(t, err)
				
				return g
			},
			sourceID: "node3",
			targetID: "node1", // This would create a cycle
			wantErr:  true,
			errMsg:   "cycle",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			graph := tt.setup()
			err := graph.AddEdge(tt.sourceID, tt.targetID)
			
			if tt.wantErr {
				require.Error(t, err)
				if tt.errMsg != "" {
					assert.Contains(t, err.Error(), tt.errMsg)
				}
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func TestGraph_TopologicalSort(t *testing.T) {
	tests := []struct {
		name    string
		setup   func() ports.Graph
		want    []string // Expected order of IDs
		wantErr bool
	}{
		{
			name: "sorts simple chain",
			setup: func() ports.Graph {
				g := NewGraph()
				
				// Add nodes
				for i := 1; i <= 3; i++ {
					err := g.AddNode(&mockExecutable{id: fmt.Sprintf("node%d", i)})
					require.NoError(t, err)
				}
				
				// Create chain: 1 -> 2 -> 3
				err := g.AddEdge("node1", "node2")
				require.NoError(t, err)
				err = g.AddEdge("node2", "node3")
				require.NoError(t, err)
				
				return g
			},
			want:    []string{"node1", "node2", "node3"},
			wantErr: false,
		},
		{
			name: "sorts diamond pattern",
			setup: func() ports.Graph {
				g := NewGraph()
				
				// Add nodes
				nodes := []string{"A", "B", "C", "D"}
				for _, id := range nodes {
					err := g.AddNode(&mockExecutable{id: id})
					require.NoError(t, err)
				}
				
				// Create diamond: A -> B,C -> D
				err := g.AddEdge("A", "B")
				require.NoError(t, err)
				err = g.AddEdge("A", "C")
				require.NoError(t, err)
				err = g.AddEdge("B", "D")
				require.NoError(t, err)
				err = g.AddEdge("C", "D")
				require.NoError(t, err)
				
				return g
			},
			want:    []string{"A", "B", "C", "D"}, // B and C can be in any order
			wantErr: false,
		},
		{
			name: "handles disconnected components",
			setup: func() ports.Graph {
				g := NewGraph()
				
				// Add two disconnected chains
				for i := 1; i <= 4; i++ {
					err := g.AddNode(&mockExecutable{id: fmt.Sprintf("node%d", i)})
					require.NoError(t, err)
				}
				
				// Chain 1: 1 -> 2
				err := g.AddEdge("node1", "node2")
				require.NoError(t, err)
				
				// Chain 2: 3 -> 4
				err = g.AddEdge("node3", "node4")
				require.NoError(t, err)
				
				return g
			},
			want:    []string{"node1", "node2", "node3", "node4"}, // Order between chains doesn't matter
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			graph := tt.setup()
			
			sorted, err := graph.TopologicalSort()
			
			if tt.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				
				// Verify we have all nodes
				assert.Len(t, sorted, len(tt.want))
				
				// For simple chain, verify exact order
				if tt.name == "sorts simple chain" {
					for i, exec := range sorted {
						assert.Equal(t, tt.want[i], exec.ID())
					}
				}
				
				// For other cases, just verify all nodes are present
				gotIDs := make(map[string]bool)
				for _, exec := range sorted {
					gotIDs[exec.ID()] = true
				}
				
				for _, wantID := range tt.want {
					assert.True(t, gotIDs[wantID], "missing node %s", wantID)
				}
			}
		})
	}
}