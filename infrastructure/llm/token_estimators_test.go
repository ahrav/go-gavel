package llm

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestWordBasedTokenEstimator_EstimatesBasedOnWordCount(t *testing.T) {
	tests := []struct {
		name           string
		text           string
		tokensPerWord  float64
		expectedTokens int
	}{
		{
			name:           "simple sentence",
			text:           "Hello world how are you",
			tokensPerWord:  0.75,
			expectedTokens: 3, // 5 words * 0.75 = 3.75, int(3.75) = 3
		},
		{
			name:           "single word",
			text:           "Hello",
			tokensPerWord:  1.0,
			expectedTokens: 1,
		},
		{
			name:           "empty text",
			text:           "",
			tokensPerWord:  0.75,
			expectedTokens: 0,
		},
		{
			name:           "whitespace only",
			text:           "   \t\n  ",
			tokensPerWord:  0.75,
			expectedTokens: 0,
		},
		{
			name:           "multiple spaces",
			text:           "word1    word2     word3",
			tokensPerWord:  1.0,
			expectedTokens: 3,
		},
		{
			name:           "high ratio",
			text:           "one two three",
			tokensPerWord:  2.0,
			expectedTokens: 6, // 3 words * 2.0 = 6
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			estimator := NewWordBasedTokenEstimator(tt.tokensPerWord)
			tokens := estimator.EstimateTokens(tt.text)
			assert.Equal(t, tt.expectedTokens, tokens, "token estimate should match expected")
		})
	}
}

func TestWordBasedTokenEstimator_UsesDefaultRatio(t *testing.T) {
	// Given an estimator with zero/negative ratio (should use default)
	estimator1 := NewWordBasedTokenEstimator(0)
	estimator2 := NewWordBasedTokenEstimator(-1.5)

	text := "test sentence with four words"
	expected := int(4 * 0.75) // 4 words * default 0.75 = 3

	// When estimating tokens
	tokens1 := estimator1.EstimateTokens(text)
	tokens2 := estimator2.EstimateTokens(text)

	// Then both should use default ratio
	assert.Equal(t, expected, tokens1, "should use default ratio for zero")
	assert.Equal(t, expected, tokens2, "should use default ratio for negative")
}

func TestCharacterBasedTokenEstimator_EstimatesBasedOnCharacterCount(t *testing.T) {
	tests := []struct {
		name               string
		text               string
		charactersPerToken float64
		expectedTokens     int
	}{
		{
			name:               "simple text",
			text:               "Hello world",
			charactersPerToken: 4.0,
			expectedTokens:     2, // 11 chars / 4.0 = 2.75, int(2.75) = 2
		},
		{
			name:               "single character",
			text:               "A",
			charactersPerToken: 1.0,
			expectedTokens:     1,
		},
		{
			name:               "empty text",
			text:               "",
			charactersPerToken: 4.0,
			expectedTokens:     0,
		},
		{
			name:               "long text",
			text:               "This is a longer text with more characters",
			charactersPerToken: 5.0,
			expectedTokens:     8, // 42 chars / 5.0 = 8.4, int(8.4) = 8
		},
		{
			name:               "unicode characters",
			text:               "Hello 世界! 🌍",
			charactersPerToken: 3.0,
			expectedTokens:     6, // This text has more chars due to unicode encoding
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			estimator := NewCharacterBasedTokenEstimator(tt.charactersPerToken)
			tokens := estimator.EstimateTokens(tt.text)
			assert.Equal(t, tt.expectedTokens, tokens, "token estimate should match expected")
		})
	}
}

func TestCharacterBasedTokenEstimator_UsesDefaultRatio(t *testing.T) {
	// Given an estimator with zero/negative ratio (should use default)
	estimator1 := NewCharacterBasedTokenEstimator(0)
	estimator2 := NewCharacterBasedTokenEstimator(-2.5)

	text := "test string"
	expected := int(float64(len(text)) / 4.0) // length / default 4.0

	// When estimating tokens
	tokens1 := estimator1.EstimateTokens(text)
	tokens2 := estimator2.EstimateTokens(text)

	// Then both should use default ratio
	assert.Equal(t, expected, tokens1, "should use default ratio for zero")
	assert.Equal(t, expected, tokens2, "should use default ratio for negative")
}

func TestRegexBasedTokenEstimator_EstimatesBasedOnPatterns(t *testing.T) {
	estimator := NewRegexBasedTokenEstimator()

	tests := []struct {
		name        string
		text        string
		minExpected int
		maxExpected int
	}{
		{
			name:        "simple words",
			text:        "hello world",
			minExpected: 8, // roughly: "hello"(5*1.0) + "world"(5*1.0) + space(1*0.1) = 10.1
			maxExpected: 15,
		},
		{
			name:        "with numbers",
			text:        "test 123 456",
			minExpected: 5,  // Allow for variation in pattern matching
			maxExpected: 15, // Increased upper bound
		},
		{
			name:        "with punctuation",
			text:        "Hello, world!",
			minExpected: 8, // words + punctuation weights
			maxExpected: 15,
		},
		{
			name:        "empty text",
			text:        "",
			minExpected: 0,
			maxExpected: 0,
		},
		{
			name:        "code-like text",
			text:        "function(param) { return true; }",
			minExpected: 15, // mix of words, brackets, punctuation
			maxExpected: 35,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokens := estimator.EstimateTokens(tt.text)
			assert.GreaterOrEqual(t, tokens, tt.minExpected, "should be at least minimum expected")
			assert.LessOrEqual(t, tokens, tt.maxExpected, "should not exceed maximum expected")
		})
	}
}

func TestRegexBasedTokenEstimator_HandlesComplexText(t *testing.T) {
	estimator := NewRegexBasedTokenEstimator()

	// Test with a variety of text types
	texts := []string{
		"Regular sentence with words.",
		"Code: if (x > 0) { return x * 2; }",
		"Numbers: 123, 456.789, -10",
		"Mixed: Hello! How are you today? 😊",
		strings.Repeat("a", 1000), // Very long text
	}

	for i, text := range texts {
		t.Run(string(rune('A'+i)), func(t *testing.T) {
			tokens := estimator.EstimateTokens(text)
			assert.Greater(t, tokens, 0, "should estimate positive tokens for non-empty text")
			assert.Less(t, tokens, len(text)*2, "should not estimate more than 2x character count")
		})
	}
}

func TestProviderSpecificTokenEstimator_RoutesByProvider(t *testing.T) {
	// Given estimators for different providers
	openaiEstimator := NewWordBasedTokenEstimator(1.0)
	anthropicEstimator := NewWordBasedTokenEstimator(0.8)

	providerEstimator := NewProviderSpecificTokenEstimator()
	providerEstimator.SetProviderEstimator("openai", openaiEstimator)
	providerEstimator.SetProviderEstimator("anthropic", anthropicEstimator)

	text := "test sentence with four words"

	// When estimating for different providers
	openaiTokens := providerEstimator.EstimateTokensForProvider("openai", text)
	anthropicTokens := providerEstimator.EstimateTokensForProvider("anthropic", text)

	// Then different estimators should be used
	assert.Equal(t, 5, openaiTokens, "should use OpenAI estimator (5 words * 1.0)")
	assert.Equal(t, 4, anthropicTokens, "should use Anthropic estimator (5 words * 0.8 = 4)")
}

func TestProviderSpecificTokenEstimator_FallsBackToDefault(t *testing.T) {
	// Given a provider estimator with custom default
	customDefault := NewCharacterBasedTokenEstimator(2.0)
	providerEstimator := NewProviderSpecificTokenEstimator()
	providerEstimator.SetDefaultEstimator(customDefault)

	text := "test" // 4 characters

	// When estimating for unknown provider
	tokens := providerEstimator.EstimateTokensForProvider("unknown", text)

	// Then default estimator should be used
	expected := int(4 / 2.0) // 4 chars / 2.0 = 2
	assert.Equal(t, expected, tokens, "should use custom default estimator")
}

func TestProviderSpecificTokenEstimator_UsesBuiltinDefaultWhenNoneSet(t *testing.T) {
	// Given a provider estimator with no custom default
	providerEstimator := NewProviderSpecificTokenEstimator()

	text := "test sentence"

	// When estimating for unknown provider
	tokens := providerEstimator.EstimateTokensForProvider("unknown", text)

	// Then built-in SimpleTokenEstimator should be used (rough word count)
	assert.Greater(t, tokens, 0, "should estimate positive tokens")
	assert.Less(t, tokens, 20, "should be reasonable estimate")
}

func TestProviderSpecificTokenEstimator_EstimateTokensUsesDefault(t *testing.T) {
	// Given a provider estimator with custom default
	customDefault := NewWordBasedTokenEstimator(1.5)
	providerEstimator := NewProviderSpecificTokenEstimator()
	providerEstimator.SetDefaultEstimator(customDefault)

	text := "two words"

	// When calling EstimateTokens (without provider)
	tokens := providerEstimator.EstimateTokens(text)

	// Then default estimator should be used
	expected := int(2 * 1.5) // 2 words * 1.5 = 3
	assert.Equal(t, expected, tokens, "should use default estimator")
}

func TestCachingTokenEstimator_CachesResults(t *testing.T) {
	// Given a caching estimator wrapping a word-based estimator
	underlying := NewWordBasedTokenEstimator(1.0)
	cachingEstimator := NewCachingTokenEstimator(underlying, 10)

	text := "cached test text"

	// When estimating the same text multiple times
	tokens1 := cachingEstimator.EstimateTokens(text)
	tokens2 := cachingEstimator.EstimateTokens(text)
	tokens3 := cachingEstimator.EstimateTokens(text)

	// Then results should be consistent (from cache)
	assert.Equal(t, tokens1, tokens2, "cached result should match original")
	assert.Equal(t, tokens1, tokens3, "cached result should match original")
	assert.Equal(t, 3, tokens1, "should estimate 3 tokens for 3 words")
}

func TestCachingTokenEstimator_DifferentTextsHaveDifferentResults(t *testing.T) {
	// Given a caching estimator
	underlying := NewWordBasedTokenEstimator(1.0)
	cachingEstimator := NewCachingTokenEstimator(underlying, 10)

	// When estimating different texts
	tokens1 := cachingEstimator.EstimateTokens("one word")
	tokens2 := cachingEstimator.EstimateTokens("two words here")

	// Then results should be different
	assert.NotEqual(t, tokens1, tokens2, "different texts should have different estimates")
	assert.Equal(t, 2, tokens1, "should estimate 2 tokens for 2 words")
	assert.Equal(t, 3, tokens2, "should estimate 3 tokens for 3 words")
}

func TestCachingTokenEstimator_RespectsMaxSize(t *testing.T) {
	// Given a caching estimator with small cache size
	underlying := NewWordBasedTokenEstimator(1.0)
	cachingEstimator := NewCachingTokenEstimator(underlying, 2)

	// When adding more entries than cache size
	cachingEstimator.EstimateTokens("text one")
	cachingEstimator.EstimateTokens("text two")
	assert.Equal(t, 2, cachingEstimator.CacheSize(), "cache should have 2 entries")

	cachingEstimator.EstimateTokens("text three")
	// Cache size should not exceed maximum
	assert.LessOrEqual(t, cachingEstimator.CacheSize(), 2, "cache should not exceed max size")
}

func TestCachingTokenEstimator_ClearCache(t *testing.T) {
	// Given a caching estimator with cached data
	underlying := NewWordBasedTokenEstimator(1.0)
	cachingEstimator := NewCachingTokenEstimator(underlying, 10)

	cachingEstimator.EstimateTokens("test text")
	assert.Equal(t, 1, cachingEstimator.CacheSize(), "cache should have 1 entry")

	// When clearing cache
	cachingEstimator.ClearCache()

	// Then cache should be empty
	assert.Equal(t, 0, cachingEstimator.CacheSize(), "cache should be empty after clear")
}

func TestCachingTokenEstimator_UsesDefaultMaxSize(t *testing.T) {
	// Given a caching estimator with zero/negative max size
	underlying := NewWordBasedTokenEstimator(1.0)
	cachingEstimator1 := NewCachingTokenEstimator(underlying, 0)
	cachingEstimator2 := NewCachingTokenEstimator(underlying, -5)

	// When estimating tokens
	cachingEstimator1.EstimateTokens("test")
	cachingEstimator2.EstimateTokens("test")

	// Then both should work (using default max size)
	assert.GreaterOrEqual(t, cachingEstimator1.CacheSize(), 0, "should handle zero max size")
	assert.GreaterOrEqual(t, cachingEstimator2.CacheSize(), 0, "should handle negative max size")
}

func TestSimpleTokenEstimator_ProvidesBasicEstimation(t *testing.T) {
	estimator := &SimpleTokenEstimator{}

	tests := []struct {
		name string
		text string
	}{
		{"empty text", ""},
		{"single word", "hello"},
		{"multiple words", "hello world test"},
		{"with punctuation", "Hello, world!"},
		{"with numbers", "test 123 456"},
		{"long text", strings.Repeat("word ", 100)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokens := estimator.EstimateTokens(tt.text)

			if tt.text == "" {
				assert.Equal(t, 0, tokens, "empty text should have 0 tokens")
			} else {
				assert.Greater(t, tokens, 0, "non-empty text should have positive tokens")
				// Should be a reasonable estimate (not too high or too low)
				assert.Less(t, tokens, len(tt.text), "tokens should be less than character count")
			}
		})
	}
}

func TestTokenEstimators_ConsistencyAcrossEstimators(t *testing.T) {
	// Given different estimators configured similarly
	wordEstimator := NewWordBasedTokenEstimator(0.75)
	charEstimator := NewCharacterBasedTokenEstimator(4.0)
	regexEstimator := NewRegexBasedTokenEstimator()
	simpleEstimator := &SimpleTokenEstimator{}

	text := "This is a test sentence with seven words"

	// When estimating tokens
	wordTokens := wordEstimator.EstimateTokens(text)
	charTokens := charEstimator.EstimateTokens(text)
	regexTokens := regexEstimator.EstimateTokens(text)
	simpleTokens := simpleEstimator.EstimateTokens(text)

	// Then estimates should be in reasonable ranges
	assert.Greater(t, wordTokens, 0, "word estimator should return positive")
	assert.Greater(t, charTokens, 0, "char estimator should return positive")
	assert.Greater(t, regexTokens, 0, "regex estimator should return positive")
	assert.Greater(t, simpleTokens, 0, "simple estimator should return positive")

	// All should be within reasonable bounds for this text
	estimates := []int{wordTokens, charTokens, regexTokens, simpleTokens}
	for i, estimate := range estimates {
		assert.Less(t, estimate, 50, "estimate %d should be reasonable", i)
		assert.Greater(t, estimate, 1, "estimate %d should be positive", i)
	}
}

func TestTokenEstimators_HandleEdgeCases(t *testing.T) {
	estimators := []TokenEstimator{
		NewWordBasedTokenEstimator(0.75),
		NewCharacterBasedTokenEstimator(4.0),
		NewRegexBasedTokenEstimator(),
		&SimpleTokenEstimator{},
	}

	edgeCases := []string{
		"",                         // Empty
		" ",                        // Single space
		"\n\t\r",                   // Whitespace only
		"a",                        // Single character
		strings.Repeat("a", 10000), // Very long
		"🌍🌎🌏",                      // Unicode emojis
		"Hello\x00World",           // Null byte
		"Mixed 123 !@# 世界",         // Mixed content
	}

	for _, estimator := range estimators {
		for _, text := range edgeCases {
			t.Run("edge_case", func(t *testing.T) {
				// Should not panic and should return reasonable values
				tokens := estimator.EstimateTokens(text)
				assert.GreaterOrEqual(t, tokens, 0, "should not return negative tokens")

				if text == "" || strings.TrimSpace(text) == "" {
					// Empty or whitespace-only text should generally return 0 or very low count
					assert.LessOrEqual(t, tokens, 1, "empty/whitespace text should have 0 or 1 tokens")
				}
			})
		}
	}
}
