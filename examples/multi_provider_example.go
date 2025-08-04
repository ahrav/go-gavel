// Package main demonstrates multi-provider LLM evaluation using go-gavel.
// This example shows how to configure and use multiple LLM providers
// (OpenAI, Anthropic, Google) within a single evaluation graph.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/ahrav/go-gavel/infrastructure/llm"
	"github.com/ahrav/go-gavel/infrastructure/middleware"
	"github.com/ahrav/go-gavel/internal/application"
	"github.com/ahrav/go-gavel/internal/domain"
)

func main() {
	// Set up context
	ctx := context.Background()

	// Step 1: Verify environment variables
	if err := verifyEnvironment(); err != nil {
		log.Fatal("Environment setup error:", err)
	}

	// Step 2: Create and configure provider registry
	registry, err := setupProviderRegistry()
	if err != nil {
		log.Fatal("Provider registry setup error:", err)
	}

	// Step 3: Create unit registry with default LLM client
	// Note: Individual units will use their specific providers via the registry
	unitRegistry := application.NewDefaultUnitRegistry(nil)

	// Step 4: Create graph loader with multi-provider support
	graphLoader, err := application.NewGraphLoader(unitRegistry, registry)
	if err != nil {
		log.Fatal("Graph loader creation error:", err)
	}

	// Step 5: Load evaluation configuration
	_, err = graphLoader.LoadFromFile(ctx, "multi-provider-evaluation.yaml")
	if err != nil {
		log.Fatal("Failed to load evaluation graph:", err)
	}

	// Step 6: Prepare evaluation state with a question
	state := domain.NewState()
	state = domain.With(state, domain.KeyQuestion,
		"What are the key differences between machine learning and deep learning?")
	_ = state // State would be used in actual evaluation

	// Step 7: Execute evaluation
	// Note: The graph returned by LoadFromFile represents the evaluation topology.
	// In this simplified example, we'll demonstrate that the graph was loaded successfully.
	// In a real implementation, you would need to:
	// 1. Extract the pipeline or layer structure from the graph
	// 2. Execute the units in the correct order based on the graph topology
	// 3. Or create your own execution logic based on the graph structure
	fmt.Println("Multi-provider evaluation graph loaded successfully!")
	fmt.Printf("Graph contains units configured with multiple providers\n")

	// For demonstration purposes, we'll show the final state with sample data
	// In practice, this would be the result of executing the evaluation pipeline
	fmt.Println("\nSample evaluation results:")
	displaySampleResults()
}

// verifyEnvironment checks that all required environment variables are set
func verifyEnvironment() error {
	required := map[string]string{
		"OPENAI_API_KEY":                 "OpenAI API key",
		"ANTHROPIC_API_KEY":              "Anthropic API key",
		"GOOGLE_APPLICATION_CREDENTIALS": "Google Cloud credentials file path",
	}

	var missing []string
	for env, desc := range required {
		if os.Getenv(env) == "" {
			missing = append(missing, fmt.Sprintf("%s (%s)", env, desc))
		}
	}

	if len(missing) > 0 {
		return fmt.Errorf("missing required environment variables: %v", missing)
	}

	// Verify Google credentials file exists
	googleCreds := os.Getenv("GOOGLE_APPLICATION_CREDENTIALS")
	if _, err := os.Stat(googleCreds); os.IsNotExist(err) {
		return fmt.Errorf("google credentials file not found: %s", googleCreds)
	}

	return nil
}

// setupProviderRegistry creates and configures the multi-provider registry
func setupProviderRegistry() (*llm.Registry, error) {
	// Create registry with OpenAI as default provider
	config := llm.RegistryConfig{
		DefaultProvider: "openai",
		Providers:       llm.DefaultProviders,
	}
	registry, err := llm.NewRegistry(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create provider registry: %w", err)
	}

	// Initialize all providers
	if err := registry.InitializeProviders(); err != nil {
		return nil, fmt.Errorf("failed to initialize providers: %w", err)
	}

	// Optional: Add metrics collection to providers
	if err := addMetricsToProviders(registry); err != nil {
		return nil, fmt.Errorf("failed to add metrics: %w", err)
	}

	fmt.Println("Successfully initialized providers: OpenAI, Anthropic, Google")
	return registry, nil
}

// addMetricsToProviders adds Prometheus metrics collection to each provider
func addMetricsToProviders(registry *llm.Registry) error {
	// Create metrics collector (configured during client creation in the new unified system)
	_ = middleware.NewPrometheusMetrics()

	// Note: In the new unified client system, metrics are configured during client creation
	// via the WithMetrics() option rather than being added post-creation.
	// The registry already handles metrics configuration during initialization.

	// Log successful metrics setup
	providers := []string{"openai", "anthropic", "google"}
	for _, provider := range providers {
		_, err := registry.GetClient(provider)
		if err != nil {
			// Skip if provider not initialized (e.g., missing credentials)
			continue
		}
		fmt.Printf("Metrics configured for provider: %s\n", provider)
	}

	return nil
}

// displaySampleResults shows example evaluation results
func displaySampleResults() {
	// Sample answers that would be generated
	fmt.Printf("\nGenerated 3 answers\n")
	fmt.Printf("\nAnswer 1 (ID: answer-1):\n")
	fmt.Printf("Machine learning is a subset of AI that enables systems to learn from data...\n")
	fmt.Printf("\nAnswer 2 (ID: answer-2):\n")
	fmt.Printf("Deep learning is a specialized form of machine learning that uses neural networks...\n")
	fmt.Printf("\nAnswer 3 (ID: answer-3):\n")
	fmt.Printf("The key differences include: ML uses various algorithms while DL specifically uses neural networks...\n")

	// Sample judge evaluations from different providers
	fmt.Printf("\n\nJudge Evaluations:\n")
	fmt.Printf("\nOpenAI Judge:\n")
	fmt.Printf("  Score: 0.85\n")
	fmt.Printf("  Confidence: 0.90\n")
	fmt.Printf("  Reasoning: Clear and comprehensive explanation of the differences\n")

	fmt.Printf("\nAnthropic Judge:\n")
	fmt.Printf("  Score: 0.82\n")
	fmt.Printf("  Confidence: 0.88\n")
	fmt.Printf("  Reasoning: Good technical accuracy with practical examples\n")

	fmt.Printf("\nGoogle Judge:\n")
	fmt.Printf("  Score: 0.87\n")
	fmt.Printf("  Confidence: 0.92\n")
	fmt.Printf("  Reasoning: Excellent structure and depth of explanation\n")

	// Sample final verdict
	fmt.Printf("\n\nFinal Verdict:\n")
	fmt.Printf("Winner: Answer 3\n")
	fmt.Printf("Aggregate Score: 0.85\n")
	fmt.Printf("\nWinning Answer:\n")
	fmt.Printf("The key differences include: ML uses various algorithms while DL specifically uses neural networks...\n")
}

// Example of programmatic multi-provider configuration (alternative to YAML)
// This function is commented out but provided as a reference for programmatic configuration
/*
func createProgrammaticConfig() application.GraphConfig {
	return application.GraphConfig{
		Version: "1.0.0",
		Metadata: application.Metadata{
			Name:        "multi-provider-evaluation",
			Description: "Programmatic multi-provider configuration",
		},
		Units: []application.UnitConfig{
			{
				ID:    "openai-judge",
				Type:  "score_judge",
				Model: "openai/gpt-4",
				Budget: application.BudgetConfig{
					MaxTokens: 1000,
					MaxCalls:  10,
				},
				// Parameters would be set as YAML nodes in actual usage
			},
			{
				ID:    "anthropic-judge",
				Type:  "score_judge",
				Model: "anthropic/claude-3-sonnet",
				Budget: application.BudgetConfig{
					MaxTokens: 1000,
					MaxCalls:  10,
				},
			},
			{
				ID:    "google-judge",
				Type:  "score_judge",
				Model: "google/gemini-pro",
				Budget: application.BudgetConfig{
					MaxTokens: 1000,
					MaxCalls:  10,
				},
			},
		},
		// Graph topology would be configured here
	}
}
*/
