package units

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"

	"github.com/ahrav/go-gavel/internal/domain"
)

func TestNewFuzzyMatchUnit(t *testing.T) {
	tests := []struct {
		name      string
		unitName  string
		config    FuzzyMatchConfig
		wantError bool
		errorMsg  string
	}{
		{
			name:     "valid configuration",
			unitName: "test-fuzzy-match",
			config: FuzzyMatchConfig{
				Algorithm:     "levenshtein",
				Threshold:     0.8,
				CaseSensitive: true,
			},
			wantError: false,
		},
		{
			name:     "default configuration",
			unitName: "test-fuzzy-match",
			config:   DefaultFuzzyMatchConfig(),
			wantError: false,
		},
		{
			name:      "empty unit name",
			unitName:  "",
			config:    DefaultFuzzyMatchConfig(),
			wantError: true,
			errorMsg:  "unit name cannot be empty",
		},
		{
			name:     "invalid algorithm",
			unitName: "test-fuzzy-match",
			config: FuzzyMatchConfig{
				Algorithm:     "invalid",
				Threshold:     0.8,
				CaseSensitive: false,
			},
			wantError: true,
			errorMsg:  "oneof",
		},
		{
			name:     "threshold below minimum",
			unitName: "test-fuzzy-match",
			config: FuzzyMatchConfig{
				Algorithm:     "levenshtein",
				Threshold:     -0.1,
				CaseSensitive: false,
			},
			wantError: true,
			errorMsg:  "min",
		},
		{
			name:     "threshold above maximum",
			unitName: "test-fuzzy-match",
			config: FuzzyMatchConfig{
				Algorithm:     "levenshtein",
				Threshold:     1.1,
				CaseSensitive: false,
			},
			wantError: true,
			errorMsg:  "max",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			unit, err := NewFuzzyMatchUnit(tt.unitName, tt.config)
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

func TestFuzzyMatchUnit_Execute(t *testing.T) {
	tests := []struct {
		name            string
		config          FuzzyMatchConfig
		answers         []domain.Answer
		referenceAnswer string
		expectedMinScore []float64 // Minimum expected scores
		expectedMaxScore []float64 // Maximum expected scores
		expectedError   bool
		errorMsg        string
	}{
		{
			name: "high similarity matches",
			config: FuzzyMatchConfig{
				Algorithm:     "levenshtein",
				Threshold:     0.8,
				CaseSensitive: false,
			},
			answers: []domain.Answer{
				{ID: "1", Content: "Hello World"},
				{ID: "2", Content: "Hello world"},
				{ID: "3", Content: "Hello Wrld"}, // One character missing
			},
			referenceAnswer: "hello world",
			expectedMinScore: []float64{1.0, 1.0, 0.85},
			expectedMaxScore: []float64{1.0, 1.0, 0.95},
			expectedError:   false,
		},
		{
			name: "below threshold matches",
			config: FuzzyMatchConfig{
				Algorithm:     "levenshtein",
				Threshold:     0.9,
				CaseSensitive: false,
			},
			answers: []domain.Answer{
				{ID: "1", Content: "Hello World"},
				{ID: "2", Content: "Hi World"},     // Too different
				{ID: "3", Content: "Goodbye World"}, // Very different
			},
			referenceAnswer: "hello world",
			expectedMinScore: []float64{1.0, 0.0, 0.0},
			expectedMaxScore: []float64{1.0, 0.0, 0.0},
			expectedError:   false,
		},
		{
			name: "case sensitive matching",
			config: FuzzyMatchConfig{
				Algorithm:     "levenshtein",
				Threshold:     0.7,
				CaseSensitive: true,
			},
			answers: []domain.Answer{
				{ID: "1", Content: "Hello World"}, // 2 case differences
				{ID: "2", Content: "hello world"}, // Exact match
				{ID: "3", Content: "HELLO WORLD"}, // All case differences
			},
			referenceAnswer: "hello world",
			expectedMinScore: []float64{0.8, 1.0, 0.0}, // Hello World is >70% similar even with case differences
			expectedMaxScore: []float64{0.85, 1.0, 0.0},
			expectedError:   false,
		},
		{
			name: "partial matches",
			config: FuzzyMatchConfig{
				Algorithm:     "levenshtein",
				Threshold:     0.5,
				CaseSensitive: false,
			},
			answers: []domain.Answer{
				{ID: "1", Content: "hello"},
				{ID: "2", Content: "world"},
				{ID: "3", Content: "helo word"}, // Typos
			},
			referenceAnswer: "hello world",
			expectedMinScore: []float64{0.0, 0.0, 0.7},
			expectedMaxScore: []float64{0.5, 0.5, 0.85},
			expectedError:   false,
		},
		{
			name:            "missing answers",
			config:          DefaultFuzzyMatchConfig(),
			answers:         nil,
			referenceAnswer: "hello world",
			expectedError:   true,
			errorMsg:        "answers not found in state",
		},
		{
			name:            "empty answers",
			config:          DefaultFuzzyMatchConfig(),
			answers:         []domain.Answer{},
			referenceAnswer: "hello world",
			expectedError:   true,
			errorMsg:        "no answers provided for fuzzy match evaluation",
		},
		{
			name:   "missing reference answer",
			config: DefaultFuzzyMatchConfig(),
			answers: []domain.Answer{
				{ID: "1", Content: "hello"},
			},
			referenceAnswer: "",
			expectedError:   true,
			errorMsg:        "reference_answer required for deterministic evaluation",
		},
		{
			name:   "too many answers",
			config: DefaultFuzzyMatchConfig(),
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
			config: DefaultFuzzyMatchConfig(),
			answers: []domain.Answer{
				{ID: "1", Content: "hello"},
			},
			referenceAnswer: strings.Repeat("a", MaxStringLength+1),
			expectedError:   true,
			errorMsg:        fmt.Sprintf("reference answer too long: %d bytes exceeds limit of %d", MaxStringLength+1, MaxStringLength),
		},
		{
			name:   "answer content too long",
			config: DefaultFuzzyMatchConfig(),
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
			unit, err := NewFuzzyMatchUnit("test-unit", tt.config)
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

				// Check each score is within expected range.
				for i := range tt.expectedMinScore {
					assert.GreaterOrEqual(t, scores[i].Score, tt.expectedMinScore[i],
						"Score for answer %d below minimum", i)
					assert.LessOrEqual(t, scores[i].Score, tt.expectedMaxScore[i],
						"Score for answer %d above maximum", i)
					assert.Equal(t, 1.0, scores[i].Confidence, "Confidence should always be 1.0")
				}
			}
		})
	}
}

func TestFuzzyMatchUnit_CalculateSimilarity(t *testing.T) {
	unit, err := NewFuzzyMatchUnit("test", DefaultFuzzyMatchConfig())
	require.NoError(t, err)

	tests := []struct {
		name         string
		s1           string
		s2           string
		expectedMin  float64
		expectedMax  float64
	}{
		{
			name:        "identical strings",
			s1:          "hello world",
			s2:          "hello world",
			expectedMin: 1.0,
			expectedMax: 1.0,
		},
		{
			name:        "completely different",
			s1:          "abc",
			s2:          "xyz",
			expectedMin: 0.0,
			expectedMax: 0.0,
		},
		{
			name:        "one character difference",
			s1:          "hello",
			s2:          "hallo",
			expectedMin: 0.75,
			expectedMax: 0.85,
		},
		{
			name:        "empty strings",
			s1:          "",
			s2:          "",
			expectedMin: 1.0,
			expectedMax: 1.0,
		},
		{
			name:        "one empty string",
			s1:          "hello",
			s2:          "",
			expectedMin: 0.0,
			expectedMax: 0.0,
		},
		{
			name:        "case differences",
			s1:          "Hello",
			s2:          "HELLO",
			expectedMin: 0.0,  // 4 out of 5 characters are different
			expectedMax: 0.25, // Levenshtein distance is 4 for these
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			similarity := unit.calculateSimilarity(tt.s1, tt.s2)
			assert.GreaterOrEqual(t, similarity, tt.expectedMin,
				"Similarity below expected minimum")
			assert.LessOrEqual(t, similarity, tt.expectedMax,
				"Similarity above expected maximum")
			assert.GreaterOrEqual(t, similarity, 0.0, "Similarity must be >= 0")
			assert.LessOrEqual(t, similarity, 1.0, "Similarity must be <= 1")
		})
	}
}

func TestFuzzyMatchUnit_Determinism(t *testing.T) {
	// Test that the unit produces identical results for identical inputs.
	unit, err := NewFuzzyMatchUnit("determinism-test", DefaultFuzzyMatchConfig())
	require.NoError(t, err)

	ctx := context.Background()
	state := domain.NewState()
	state = domain.With(state, domain.KeyAnswers, []domain.Answer{
		{ID: "1", Content: "Hello World"},
		{ID: "2", Content: "hello world"},
		{ID: "3", Content: "Helo Wrld"},
	})
	state = domain.With(state, domain.KeyReferenceAnswer, "hello world")

	// Run the same evaluation multiple times.
	const iterations = 10
	var results [][]domain.JudgeSummary

	for i := 0; i < iterations; i++ {
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

func TestFuzzyMatchUnit_UnmarshalParameters(t *testing.T) {
	tests := []struct {
		name      string
		yaml      string
		expected  FuzzyMatchConfig
		wantError bool
		errorMsg  string
	}{
		{
			name: "valid parameters",
			yaml: `algorithm: levenshtein
threshold: 0.75
case_sensitive: true`,
			expected: FuzzyMatchConfig{
				Algorithm:     "levenshtein",
				Threshold:     0.75,
				CaseSensitive: true,
			},
			wantError: false,
		},
		{
			name: "default values",
			yaml: ``,
			expected: FuzzyMatchConfig{
				Algorithm:     "",
				Threshold:     0.0,
				CaseSensitive: false,
			},
			wantError: true, // Algorithm is required
			errorMsg:  "required",
		},
		{
			name: "unknown field detection",
			yaml: `algorithm: levenshtein
threshold: 0.75
case_sensitiv: true`,  // Typo: case_sensitiv instead of case_sensitive
			wantError: true,
			errorMsg:  "check for typos",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			unit, err := NewFuzzyMatchUnit("test", DefaultFuzzyMatchConfig())
			require.NoError(t, err)

			var node yaml.Node
			err = yaml.Unmarshal([]byte(tt.yaml), &node)
			require.NoError(t, err)

			// Handle empty YAML content.
			var newUnit *FuzzyMatchUnit
			if len(node.Content) == 0 {
				// Create an empty node for empty YAML.
				emptyNode := yaml.Node{Kind: yaml.MappingNode}
				newUnit, err = unit.UnmarshalParameters(emptyNode)
			} else {
				newUnit, err = unit.UnmarshalParameters(*node.Content[0])
			}

			if tt.wantError {
				assert.Error(t, err)
				if tt.errorMsg != "" {
					assert.Contains(t, err.Error(), tt.errorMsg)
				}
				assert.Nil(t, newUnit)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, newUnit)
				assert.Equal(t, tt.expected, newUnit.config)
				// Verify the original unit was not modified
				assert.Equal(t, DefaultFuzzyMatchConfig(), unit.config)
			}
		})
	}
}

func TestFuzzyMatchUnit_Validate(t *testing.T) {
	unit, err := NewFuzzyMatchUnit("test", DefaultFuzzyMatchConfig())
	require.NoError(t, err)

	err = unit.Validate()
	assert.NoError(t, err)
}

func TestCreateFuzzyMatchUnit(t *testing.T) {
	tests := []struct {
		name      string
		id        string
		config    map[string]any
		wantError bool
		errorMsg  string
		expected  FuzzyMatchConfig
	}{
		{
			name: "valid config map",
			id:   "test-unit",
			config: map[string]any{
				"algorithm":      "levenshtein",
				"threshold":      0.75,
				"case_sensitive": true,
			},
			wantError: false,
			expected: FuzzyMatchConfig{
				Algorithm:     "levenshtein",
				Threshold:     0.75,
				CaseSensitive: true,
			},
		},
		{
			name:      "empty config uses defaults",
			id:        "test-unit",
			config:    map[string]any{},
			wantError: false,
			expected:  DefaultFuzzyMatchConfig(),
		},
		{
			name:      "nil config uses defaults",
			id:        "test-unit",
			config:    nil,
			wantError: false,
			expected:  DefaultFuzzyMatchConfig(),
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
			unit, err := CreateFuzzyMatchUnit(tt.id, tt.config)
			if tt.wantError {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errorMsg)
				assert.Nil(t, unit)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, unit)
				assert.Equal(t, tt.id, unit.Name())
				assert.Equal(t, tt.expected, unit.config)
			}
		})
	}
}

func TestFuzzyMatchUnit_ThreadSafety(t *testing.T) {
	// Test concurrent execution to ensure thread safety.
	unit, err := NewFuzzyMatchUnit("thread-safety-test", DefaultFuzzyMatchConfig())
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

	for i := 0; i < goroutines; i++ {
		go func() {
			_, err := unit.Execute(ctx, state)
			errors <- err
		}()
	}

	// Collect results.
	for i := 0; i < goroutines; i++ {
		err := <-errors
		assert.NoError(t, err)
	}
}

// BenchmarkFuzzyMatchUnit_Execute benchmarks the execution performance.
// The unit must achieve p95 latency ≤ 300µs as per AC#8.
func BenchmarkFuzzyMatchUnit_Execute(b *testing.B) {
	unit, err := NewFuzzyMatchUnit("benchmark", FuzzyMatchConfig{
		Algorithm:     "levenshtein",
		Threshold:     0.8,
		CaseSensitive: false,
	})
	require.NoError(b, err)

	ctx := context.Background()
	state := domain.NewState()
	state = domain.With(state, domain.KeyAnswers, []domain.Answer{
		{ID: "1", Content: "This is a test answer with some text"},
		{ID: "2", Content: "Another test answer with different content"},
		{ID: "3", Content: "Yet another test answer to evaluate"},
	})
	state = domain.With(state, domain.KeyReferenceAnswer, "This is a test answer with some text")

	// Measure latencies for p95 calculation.
	latencies := make([]time.Duration, 0, b.N)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
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
		b.Logf("p95 latency: %v (target: ≤300µs)", p95Latency)
		
		// Assert that p95 latency meets the requirement.
		if p95Latency > 300*time.Microsecond {
			b.Errorf("p95 latency %v exceeds 300µs requirement", p95Latency)
		}
	}
}

// TestFuzzyMatchUnit_UnicodeHandling tests that Unicode strings are handled correctly.
func TestFuzzyMatchUnit_UnicodeHandling(t *testing.T) {
	tests := []struct {
		name            string
		answer          string
		reference       string
		caseSensitive   bool
		threshold       float64
		expectedScore   float64
		description     string
	}{
		{
			name:          "Chinese characters exact match",
			answer:        "你好世界",
			reference:     "你好世界",
			caseSensitive: false,
			threshold:     0.8,
			expectedScore: 1.0,
			description:   "Multi-byte Chinese characters should match exactly",
		},
		{
			name:          "Chinese characters one difference",
			answer:        "你好世界",
			reference:     "你好世间",
			caseSensitive: false,
			threshold:     0.5,
			expectedScore: 0.75, // 3 out of 4 characters match
			description:   "Should calculate similarity based on rune count, not byte count",
		},
		{
			name:          "Mixed ASCII and Unicode",
			answer:        "Hello世界",
			reference:     "Hello世间",
			caseSensitive: false,
			threshold:     0.5,
			expectedScore: 0.857, // 6 out of 7 characters match after case folding
			description:   "Mixed character sets should work correctly",
		},
		{
			name:          "Emoji handling",
			answer:        "Hello 👋 World",
			reference:     "Hello 👋 World",
			caseSensitive: false,
			threshold:     0.8,
			expectedScore: 1.0,
			description:   "Emoji should be counted as single runes",
		},
		{
			name:          "Turkish İ case folding",
			answer:        "İstanbul",
			reference:     "istanbul",
			caseSensitive: false,
			threshold:     0.8,
			expectedScore: 0.889, // Actual result with proper Unicode folding
			description:   "Turkish capital İ should fold correctly",
		},
		{
			name:          "German ß case folding",
			answer:        "straße",
			reference:     "STRASSE",
			caseSensitive: false,
			threshold:     0.5,
			expectedScore: 1.0, // ß properly folds to ss, resulting in exact match
			description:   "German ß should be handled properly in case folding",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := FuzzyMatchConfig{
				Algorithm:     "levenshtein",
				Threshold:     tt.threshold,
				CaseSensitive: tt.caseSensitive,
			}

			unit, err := NewFuzzyMatchUnit("unicode-test", config)
			require.NoError(t, err)

			ctx := context.Background()
			state := domain.NewState()
			state = domain.With(state, domain.KeyAnswers, []domain.Answer{
				{ID: "1", Content: tt.answer},
			})
			state = domain.With(state, domain.KeyReferenceAnswer, tt.reference)

			result, err := unit.Execute(ctx, state)
			require.NoError(t, err)

			scores, ok := domain.Get(result, domain.KeyJudgeScores)
			require.True(t, ok)
			require.Len(t, scores, 1)

			// Allow small floating-point tolerance
			assert.InDelta(t, tt.expectedScore, scores[0].Score, 0.01, tt.description)
		})
	}
}

// TestFuzzyMatchUnit_ThreadSafetyWithReconfiguration tests that the unit is thread-safe
// when UnmarshalParameters is called concurrently with Execute.
func TestFuzzyMatchUnit_ThreadSafetyWithReconfiguration(t *testing.T) {
	baseUnit, err := NewFuzzyMatchUnit("thread-safety-reconfig", DefaultFuzzyMatchConfig())
	require.NoError(t, err)

	ctx := context.Background()
	state := domain.NewState()
	state = domain.With(state, domain.KeyAnswers, []domain.Answer{
		{ID: "1", Content: "Test Answer"},
	})
	state = domain.With(state, domain.KeyReferenceAnswer, "test answer")

	// Create different configurations
	configs := []string{
		`algorithm: levenshtein
threshold: 0.7
case_sensitive: false`,
		`algorithm: levenshtein
threshold: 0.8
case_sensitive: true`,
		`algorithm: levenshtein
threshold: 0.9
case_sensitive: false`,
	}

	const numGoroutines = 100
	const numIterations = 50
	
	var wg sync.WaitGroup
	errChan := make(chan error, numGoroutines*numIterations)
	
	// Start goroutines that continuously execute
	for i := 0; i < numGoroutines/2; i++ {
		wg.Add(1)
		go func(unit *FuzzyMatchUnit) {
			defer wg.Done()
			for j := 0; j < numIterations; j++ {
				_, err := unit.Execute(ctx, state)
				if err != nil {
					errChan <- err
				}
			}
		}(baseUnit)
	}

	// Start goroutines that reconfigure the unit
	for i := 0; i < numGoroutines/2; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < numIterations; j++ {
				configYAML := configs[j%len(configs)]
				var node yaml.Node
				if err := yaml.Unmarshal([]byte(configYAML), &node); err != nil {
					errChan <- err
					continue
				}
				// UnmarshalParameters returns a new unit, maintaining thread safety
				if _, err := baseUnit.UnmarshalParameters(*node.Content[0]); err != nil {
					errChan <- err
				}
			}
		}(i)
	}

	wg.Wait()
	close(errChan)

	// Check for any errors
	for err := range errChan {
		t.Errorf("Concurrent operation failed: %v", err)
	}
}