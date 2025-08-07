package units

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"

	"github.com/ahrav/go-gavel/internal/domain"
)

func TestNewExactMatchUnit(t *testing.T) {
	tests := []struct {
		name      string
		unitName  string
		config    ExactMatchConfig
		wantError bool
		errorMsg  string
	}{
		{
			name:     "valid configuration",
			unitName: "test-exact-match",
			config: ExactMatchConfig{
				CaseSensitive:  true,
				TrimWhitespace: true,
			},
			wantError: false,
		},
		{
			name:      "default configuration",
			unitName:  "test-exact-match",
			config:    DefaultExactMatchConfig(),
			wantError: false,
		},
		{
			name:      "empty unit name",
			unitName:  "",
			config:    DefaultExactMatchConfig(),
			wantError: true,
			errorMsg:  "unit name cannot be empty",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			unit, err := NewExactMatchUnit(tt.unitName, tt.config)
			if tt.wantError {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errorMsg)
				assert.Nil(t, unit)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, unit)
				assert.Equal(t, tt.unitName, unit.Name())
				assert.Equal(t, tt.config, unit.config)
			}
		})
	}
}

func TestExactMatchUnit_Execute(t *testing.T) {
	tests := []struct {
		name            string
		config          ExactMatchConfig
		answers         []domain.Answer
		referenceAnswer string
		expectedScores  []float64
		expectedError   bool
		errorMsg        string
	}{
		{
			name:   "exact match with default config",
			config: DefaultExactMatchConfig(),
			answers: []domain.Answer{
				{ID: "1", Content: "Hello World"},
				{ID: "2", Content: "hello world"},
				{ID: "3", Content: "  Hello World  "},
			},
			referenceAnswer: "hello world",
			expectedScores:  []float64{1.0, 1.0, 1.0},
			expectedError:   false,
		},
		{
			name: "case sensitive matching",
			config: ExactMatchConfig{
				CaseSensitive:  true,
				TrimWhitespace: true,
			},
			answers: []domain.Answer{
				{ID: "1", Content: "Hello World"},
				{ID: "2", Content: "hello world"},
				{ID: "3", Content: "HELLO WORLD"},
			},
			referenceAnswer: "hello world",
			expectedScores:  []float64{0.0, 1.0, 0.0},
			expectedError:   false,
		},
		{
			name: "whitespace not trimmed",
			config: ExactMatchConfig{
				CaseSensitive:  false,
				TrimWhitespace: false,
			},
			answers: []domain.Answer{
				{ID: "1", Content: "hello world"},
				{ID: "2", Content: "  hello world  "},
				{ID: "3", Content: "hello world "},
			},
			referenceAnswer: "hello world",
			expectedScores:  []float64{1.0, 0.0, 0.0},
			expectedError:   false,
		},
		{
			name:   "no matches",
			config: DefaultExactMatchConfig(),
			answers: []domain.Answer{
				{ID: "1", Content: "foo"},
				{ID: "2", Content: "bar"},
				{ID: "3", Content: "baz"},
			},
			referenceAnswer: "hello world",
			expectedScores:  []float64{0.0, 0.0, 0.0},
			expectedError:   false,
		},
		{
			name:            "missing answers",
			config:          DefaultExactMatchConfig(),
			answers:         nil,
			referenceAnswer: "hello world",
			expectedError:   true,
			errorMsg:        "answers not found in state",
		},
		{
			name:            "empty answers",
			config:          DefaultExactMatchConfig(),
			answers:         []domain.Answer{},
			referenceAnswer: "hello world",
			expectedError:   true,
			errorMsg:        "no answers provided for exact match evaluation",
		},
		{
			name:   "missing reference answer",
			config: DefaultExactMatchConfig(),
			answers: []domain.Answer{
				{ID: "1", Content: "hello"},
			},
			referenceAnswer: "",
			expectedError:   true,
			errorMsg:        "reference_answer required for deterministic evaluation",
		},
		{
			name:   "too many answers",
			config: DefaultExactMatchConfig(),
			answers: func() []domain.Answer {
				// Create MaxAnswers + 1 answers
				answers := make([]domain.Answer, MaxAnswers+1)
				for i := 0; i < len(answers); i++ {
					answers[i] = domain.Answer{
						ID:      fmt.Sprintf("%d", i),
						Content: "test",
					}
				}
				return answers
			}(),
			referenceAnswer: "test",
			expectedError:   true,
			errorMsg:        fmt.Sprintf("too many answers: %d exceeds limit of %d", MaxAnswers+1, MaxAnswers),
		},
		{
			name:   "reference answer too long",
			config: DefaultExactMatchConfig(),
			answers: []domain.Answer{
				{ID: "1", Content: "hello"},
			},
			referenceAnswer: strings.Repeat("a", MaxStringLength+1),
			expectedError:   true,
			errorMsg:        fmt.Sprintf("reference answer too long: %d bytes exceeds limit of %d", MaxStringLength+1, MaxStringLength),
		},
		{
			name:   "answer content too long",
			config: DefaultExactMatchConfig(),
			answers: []domain.Answer{
				{ID: "1", Content: strings.Repeat("a", MaxStringLength+1)},
			},
			referenceAnswer: "test",
			expectedError:   true,
			errorMsg:        fmt.Sprintf("answer 0 too long: %d bytes exceeds limit of %d", MaxStringLength+1, MaxStringLength),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			unit, err := NewExactMatchUnit("test-unit", tt.config)
			require.NoError(t, err)

			ctx := context.Background()
			state := domain.NewState()

			// Add answers if provided.
			if tt.answers != nil {
				state = domain.With(state, domain.KeyAnswers, tt.answers)
			}

			// Add reference answer if not testing missing reference.
			if !tt.expectedError || tt.errorMsg != "reference_answer required for deterministic evaluation" {
				state = domain.With(state, domain.KeyReferenceAnswer, tt.referenceAnswer)
			}

			newState, err := unit.Execute(ctx, state)

			if tt.expectedError {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errorMsg)
			} else {
				assert.NoError(t, err)

				// Verify judge scores were added to state.
				scores, ok := domain.Get(newState, domain.KeyJudgeScores)
				assert.True(t, ok)
				assert.Len(t, scores, len(tt.answers))

				// Check each score.
				for i, expectedScore := range tt.expectedScores {
					assert.Equal(t, expectedScore, scores[i].Score, "Score mismatch for answer %d", i)
					assert.Equal(t, 1.0, scores[i].Confidence, "Confidence should always be 1.0")
					if expectedScore == 1.0 {
						assert.Equal(t, "Exact match found", scores[i].Reasoning)
					} else {
						assert.Equal(t, "No exact match", scores[i].Reasoning)
					}
				}
			}
		})
	}
}

func TestExactMatchUnit_Determinism(t *testing.T) {
	// Test that the unit produces identical results for identical inputs.
	unit, err := NewExactMatchUnit("determinism-test", DefaultExactMatchConfig())
	require.NoError(t, err)

	ctx := context.Background()
	state := domain.NewState()
	state = domain.With(state, domain.KeyAnswers, []domain.Answer{
		{ID: "1", Content: "Hello World"},
		{ID: "2", Content: "hello world"},
		{ID: "3", Content: "  HELLO WORLD  "},
	})
	state = domain.With(state, domain.KeyReferenceAnswer, "hello world")

	// Run the same evaluation multiple times.
	const iterations = 10
	var results [][]domain.JudgeSummary

	for range iterations {
		newState, err := unit.Execute(ctx, state)
		require.NoError(t, err)

		scores, ok := domain.Get(newState, domain.KeyJudgeScores)
		require.True(t, ok)
		results = append(results, scores)
	}

	// Verify all results are identical.
	firstResult := results[0]
	for i := 1; i < iterations; i++ {
		assert.Equal(t, firstResult, results[i], "Result %d differs from first result", i)
	}
}

func TestExactMatchUnit_UnmarshalParameters(t *testing.T) {
	tests := []struct {
		name      string
		yaml      string
		expected  ExactMatchConfig
		wantError bool
		errorMsg  string
	}{
		{
			name: "valid parameters",
			yaml: `case_sensitive: true
trim_whitespace: false`,
			expected: ExactMatchConfig{
				CaseSensitive:  true,
				TrimWhitespace: false,
			},
			wantError: false,
		},
		{
			name: "default values",
			yaml: ``,
			expected: ExactMatchConfig{
				CaseSensitive:  false,
				TrimWhitespace: false,
			},
			wantError: false,
		},
		// Note: yaml.v3 Node.Decode doesn't enforce strict mode like json.Unmarshal,
		// so unknown fields are silently ignored. This is consistent with other units.
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			unit, err := NewExactMatchUnit("test", DefaultExactMatchConfig())
			require.NoError(t, err)

			var node yaml.Node
			err = yaml.Unmarshal([]byte(tt.yaml), &node)
			require.NoError(t, err)

			// Handle empty YAML content.
			if len(node.Content) == 0 {
				// Create an empty node for empty YAML.
				emptyNode := yaml.Node{Kind: yaml.MappingNode}
				err = unit.UnmarshalParameters(emptyNode)
			} else {
				err = unit.UnmarshalParameters(*node.Content[0])
			}

			if tt.wantError {
				assert.Error(t, err)
				if tt.errorMsg != "" {
					assert.Contains(t, err.Error(), tt.errorMsg)
				}
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected, unit.config)
			}
		})
	}
}

func TestExactMatchUnit_Validate(t *testing.T) {
	unit, err := NewExactMatchUnit("test", DefaultExactMatchConfig())
	require.NoError(t, err)

	err = unit.Validate()
	assert.NoError(t, err)
}

func TestNewExactMatchFromConfig(t *testing.T) {
	tests := []struct {
		name      string
		id        string
		config    map[string]any
		wantError bool
		errorMsg  string
		expected  ExactMatchConfig
	}{
		{
			name: "valid config map",
			id:   "test-unit",
			config: map[string]any{
				"case_sensitive":  true,
				"trim_whitespace": false,
			},
			wantError: false,
			expected: ExactMatchConfig{
				CaseSensitive:  true,
				TrimWhitespace: false,
			},
		},
		{
			name:      "empty config uses defaults",
			id:        "test-unit",
			config:    map[string]any{},
			wantError: false,
			expected:  DefaultExactMatchConfig(),
		},
		{
			name:      "nil config uses defaults",
			id:        "test-unit",
			config:    nil,
			wantError: false,
			expected:  DefaultExactMatchConfig(),
		},
		{
			name:      "empty id",
			id:        "",
			config:    map[string]any{},
			wantError: true,
			errorMsg:  "unit name cannot be empty",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			unitPort, err := NewExactMatchFromConfig(tt.id, tt.config, nil)
			if tt.wantError {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errorMsg)
				assert.Nil(t, unitPort)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, unitPort)
				assert.Equal(t, tt.id, unitPort.Name())
				// Type assert to access internal config
				unit, ok := unitPort.(*ExactMatchUnit)
				require.True(t, ok, "unit should be *ExactMatchUnit")
				assert.Equal(t, tt.expected, unit.config)
			}
		})
	}
}

func TestExactMatchUnit_ThreadSafety(t *testing.T) {
	// Test concurrent execution to ensure thread safety.
	unit, err := NewExactMatchUnit("thread-safety-test", DefaultExactMatchConfig())
	require.NoError(t, err)

	ctx := context.Background()
	state := domain.NewState()
	state = domain.With(state, domain.KeyAnswers, []domain.Answer{
		{ID: "1", Content: "Test Answer"},
	})
	state = domain.With(state, domain.KeyReferenceAnswer, "test answer")

	// Run concurrent executions.
	const goroutines = 100
	errors := make(chan error, goroutines)

	for range goroutines {
		go func() {
			_, err := unit.Execute(ctx, state)
			errors <- err
		}()
	}

	// Collect results.
	for range goroutines {
		err := <-errors
		assert.NoError(t, err)
	}
}

// BenchmarkExactMatchUnit_Execute benchmarks the execution performance.
// The unit must achieve p95 latency ≤ 50µs as per AC#8.
func BenchmarkExactMatchUnit_Execute(b *testing.B) {
	unit, err := NewExactMatchUnit("benchmark", ExactMatchConfig{
		CaseSensitive:  true,
		TrimWhitespace: true,
	})
	require.NoError(b, err)

	ctx := context.Background()
	state := domain.NewState()
	state = domain.With(state, domain.KeyAnswers, []domain.Answer{
		{ID: "1", Content: "This is a test answer"},
		{ID: "2", Content: "Another test answer"},
		{ID: "3", Content: "Yet another test answer"},
	})
	state = domain.With(state, domain.KeyReferenceAnswer, "This is a test answer")

	// Measure latencies for p95 calculation.
	latencies := make([]time.Duration, 0, b.N)

	for b.Loop() {
		start := time.Now()
		_, err := unit.Execute(ctx, state)
		latency := time.Since(start)
		require.NoError(b, err)
		latencies = append(latencies, latency)
	}

	// Calculate p95 latency.
	if len(latencies) > 0 {
		p95Index := int(float64(len(latencies)) * 0.95)
		if p95Index >= len(latencies) {
			p95Index = len(latencies) - 1
		}
		p95Latency := latencies[p95Index]
		b.Logf("p95 latency: %v (target: ≤50µs)", p95Latency)

		// Assert that p95 latency meets the requirement.
		if p95Latency > 50*time.Microsecond {
			b.Errorf("p95 latency %v exceeds 50µs requirement", p95Latency)
		}
	}
}
