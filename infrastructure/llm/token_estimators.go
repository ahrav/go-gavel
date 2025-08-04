// Package llm/token_estimators provides various token estimation strategies for LLM clients.
// This file implements multiple token counting approaches to support cost calculation and
// rate limiting across different LLM providers with varying tokenization methods.
//
// Available Estimator Types:
//   - WordBasedTokenEstimator: Simple word count-based estimation
//   - CharacterBasedTokenEstimator: Character count-based estimation
//   - RegexBasedTokenEstimator: Pattern-based sophisticated estimation
//   - ProviderSpecificTokenEstimator: Provider-aware estimation delegation
//   - CachingTokenEstimator: Cached results for performance optimization
//
// Usage Examples:
//
// Basic Word-Based Estimation:
//
//	estimator := llm.NewWordBasedTokenEstimator(0.75) // ~0.75 tokens per word
//	tokens := estimator.EstimateTokens("Hello world!")
//
// Character-Based Estimation:
//
//	estimator := llm.NewCharacterBasedTokenEstimator(4.0) // ~4 chars per token
//	tokens := estimator.EstimateTokens("Hello world!")
//
// Provider-Specific Estimation:
//
//	estimator := llm.NewProviderSpecificTokenEstimator()
//	estimator.SetProviderEstimator("openai", openaiEstimator)
//	estimator.SetProviderEstimator("anthropic", anthropicEstimator)
//	tokens := estimator.EstimateTokensForProvider("openai", text)
//
// Performance-Optimized Caching:
//
//	base := llm.NewWordBasedTokenEstimator(0.75)
//	cached := llm.NewCachingTokenEstimator(base, 1000) // Cache 1000 results
//	tokens := cached.EstimateTokens(text)
package llm

import (
	"regexp"
	"strings"
)

// WordBasedTokenEstimator estimates tokens based on word count.
// This estimator provides fast, simple estimation using configurable
// tokens-per-word ratios. Best for general-purpose estimation where
// speed is more important than precision.
type WordBasedTokenEstimator struct{ TokensPerWord float64 }

// NewWordBasedTokenEstimator creates a word-based token estimator.
// The tokensPerWord parameter should be tuned based on the target language
// and LLM provider. Typical values: 0.75 for English, 0.6-0.9 for other languages.
func NewWordBasedTokenEstimator(tokensPerWord float64) *WordBasedTokenEstimator {
	if tokensPerWord <= 0 {
		tokensPerWord = 0.75 // Default: ~0.75 tokens per word
	}
	return &WordBasedTokenEstimator{
		TokensPerWord: tokensPerWord,
	}
}

// EstimateTokens calculates token count based on word count.
// This method splits text on whitespace and applies the configured
// tokens-per-word ratio for fast estimation.
func (e *WordBasedTokenEstimator) EstimateTokens(text string) int {
	words := strings.Fields(text)
	return int(float64(len(words)) * e.TokensPerWord)
}

// CharacterBasedTokenEstimator estimates tokens based on character count.
// This estimator provides simple character-to-token ratio estimation.
// More accurate for languages with consistent character density,
// less accurate for code or heavily punctuated text.
type CharacterBasedTokenEstimator struct{ charsPerToken float64 }

// NewCharacterBasedTokenEstimator creates a character-based token estimator.
// The charactersPerToken parameter should be tuned for the target provider.
// Typical values: 4.0 for GPT models, 3.5-4.5 for other providers.
func NewCharacterBasedTokenEstimator(charactersPerToken float64) *CharacterBasedTokenEstimator {
	if charactersPerToken <= 0 {
		charactersPerToken = 4.0 // Default: ~4 characters per token
	}
	return &CharacterBasedTokenEstimator{
		charsPerToken: charactersPerToken,
	}
}

// EstimateTokens calculates token count based on character count.
// This method divides total character count by the configured
// characters-per-token ratio for simple estimation.
func (e *CharacterBasedTokenEstimator) EstimateTokens(text string) int {
	return int(float64(len(text)) / e.charsPerToken)
}

// RegexBasedTokenEstimator provides sophisticated pattern-based estimation.
// This estimator uses regex patterns to identify different text elements
// and applies specific weights to each type. More accurate but slower
// than simple estimators. Best for mixed content with code, punctuation, etc.
type RegexBasedTokenEstimator struct {
	patterns map[*regexp.Regexp]float64
	baseRate float64
}

// NewRegexBasedTokenEstimator creates a pattern-based token estimator.
// This estimator uses predefined regex patterns with weights to estimate
// tokens more accurately for mixed content types.
func NewRegexBasedTokenEstimator() *RegexBasedTokenEstimator {
	patterns := map[*regexp.Regexp]float64{
		regexp.MustCompile(`\b\w+\b`):      1.0, // Regular words
		regexp.MustCompile(`\d+`):          0.5, // Numbers
		regexp.MustCompile(`[.!?;:]`):      0.3, // Punctuation
		regexp.MustCompile(`\s+`):          0.1, // Whitespace
		regexp.MustCompile(`[(){}\[\]<>]`): 0.2, // Brackets
		regexp.MustCompile(`["']`):         0.1, // Quotes
	}

	return &RegexBasedTokenEstimator{
		patterns: patterns,
		baseRate: 0.25, // Base rate for unmatched characters
	}
}

// EstimateTokens calculates tokens using pattern-based analysis.
// This method applies regex patterns to identify text elements and
// calculates weighted token estimates based on content type.
func (e *RegexBasedTokenEstimator) EstimateTokens(text string) int {
	totalTokens := 0.0
	processed := make([]bool, len(text))

	// Apply each pattern with its specific weight
	for pattern, weight := range e.patterns {
		matches := pattern.FindAllStringIndex(text, -1)
		for _, match := range matches {
			start, end := match[0], match[1]
			// Mark characters as processed and add weighted tokens
			for i := start; i < end && i < len(processed); i++ {
				if !processed[i] {
					processed[i] = true
				}
			}
			totalTokens += weight * float64(end-start)
		}
	}

	// Add base rate for unprocessed characters
	unprocessedCount := 0
	for _, isProcessed := range processed {
		if !isProcessed {
			unprocessedCount++
		}
	}
	totalTokens += e.baseRate * float64(unprocessedCount)

	return int(totalTokens)
}

// ProviderSpecificTokenEstimator delegates to provider-aware estimation strategies.
// This estimator maintains separate estimators for different providers
// to account for provider-specific tokenization differences. Useful when
// working with multiple providers that have different tokenization approaches.
type ProviderSpecificTokenEstimator struct {
	providerEstimators map[string]TokenEstimator
	defaultEstimator   TokenEstimator
}

// NewProviderSpecificTokenEstimator creates a provider-aware estimator.
// This estimator can be configured with different estimation strategies
// for each provider to account for tokenization differences.
func NewProviderSpecificTokenEstimator() *ProviderSpecificTokenEstimator {
	return &ProviderSpecificTokenEstimator{
		providerEstimators: make(map[string]TokenEstimator),
		defaultEstimator:   &SimpleTokenEstimator{},
	}
}

// SetProviderEstimator configures a specific estimator for a provider.
// This allows customization of token estimation for providers with
// unique tokenization characteristics.
func (e *ProviderSpecificTokenEstimator) SetProviderEstimator(provider string, estimator TokenEstimator) {
	e.providerEstimators[provider] = estimator
}

// SetDefaultEstimator configures the fallback estimator.
// This estimator is used when no provider-specific estimator is configured.
func (e *ProviderSpecificTokenEstimator) SetDefaultEstimator(estimator TokenEstimator) {
	e.defaultEstimator = estimator
}

// EstimateTokensForProvider estimates tokens using provider-specific logic.
// This method routes to the appropriate estimator based on the provider,
// falling back to the default estimator if no specific one is configured.
func (e *ProviderSpecificTokenEstimator) EstimateTokensForProvider(provider string, text string) int {
	if estimator, exists := e.providerEstimators[provider]; exists {
		return estimator.EstimateTokens(text)
	}
	return e.defaultEstimator.EstimateTokens(text)
}

// EstimateTokens provides default token estimation.
// This method uses the default estimator when no provider context is available.
func (e *ProviderSpecificTokenEstimator) EstimateTokens(text string) int {
	return e.defaultEstimator.EstimateTokens(text)
}

// CachingTokenEstimator wraps another estimator with performance-optimized caching.
// This estimator caches estimation results to avoid repeated calculations
// for the same text. Best for applications with repeated estimation requests
// or expensive underlying estimators. Provides significant performance gains
// for repetitive workloads.
type CachingTokenEstimator struct {
	underlying TokenEstimator
	cache      map[string]int
	maxSize    int
}

// NewCachingTokenEstimator creates a caching wrapper for any TokenEstimator.
// The maxSize parameter controls memory usage vs. performance tradeoff.
// Larger cache sizes provide better hit rates but use more memory.
func NewCachingTokenEstimator(underlying TokenEstimator, maxSize int) *CachingTokenEstimator {
	if maxSize <= 0 {
		maxSize = 1000 // Default cache size
	}
	return &CachingTokenEstimator{
		underlying: underlying,
		cache:      make(map[string]int),
		maxSize:    maxSize,
	}
}

// EstimateTokens provides cached token estimation with fallthrough to underlying estimator.
// This method checks the cache first for O(1) lookup, calculates using the underlying
// estimator on cache misses, and caches results for future use.
func (e *CachingTokenEstimator) EstimateTokens(text string) int {
	if tokens, exists := e.cache[text]; exists {
		return tokens
	}

	tokens := e.underlying.EstimateTokens(text)

	// Add to cache if space available (simple cache eviction)
	if len(e.cache) < e.maxSize {
		e.cache[text] = tokens
	}

	return tokens
}

// ClearCache removes all cached estimation results.
// This method is useful for memory management or when estimation
// parameters change and cached results become invalid.
func (e *CachingTokenEstimator) ClearCache() {
	for k := range e.cache {
		delete(e.cache, k)
	}
}

// CacheSize returns the current number of cached estimation results.
// This method is useful for monitoring cache utilization and performance.
func (e *CachingTokenEstimator) CacheSize() int { return len(e.cache) }
