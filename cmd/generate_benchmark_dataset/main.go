package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/ahrav/go-gavel/internal/testutils"
)

func main() {
	var (
		size       = flag.Int("size", 500, "Number of questions to generate")
		outputPath = flag.String("output", "testdata/benchmark_dataset/sample_benchmark_dataset.json", "Output file path")
	)
	flag.Parse()

	// Generate the dataset.
	dataset := testutils.GenerateSampleBenchmarkDataset(*size, time.Now().UnixNano())

	// Add legal review notice.
	dataset.Metadata.Description = fmt.Sprintf(
		"%s\n\nLEGAL NOTICE: This is a synthetic dataset generated for testing purposes. "+
			"For production benchmarks, a properly licensed dataset must be sourced and undergo legal review.",
		dataset.Metadata.Description,
	)

	if err := testutils.SaveBenchmarkDataset(dataset, *outputPath); err != nil {
		log.Fatalf("Failed to save dataset: %v", err)
	}

	// Compute and display statistics.
	stats := testutils.ComputeDatasetStatistics(dataset)

	fmt.Printf("Generated benchmark dataset:\n")
	fmt.Printf("- Path: %s\n", *outputPath)
	fmt.Printf("- Total questions: %d\n", stats.TotalQuestions)
	fmt.Printf("- Domains: %v\n", stats.DomainsCount)
	fmt.Printf("- Difficulties: %v\n", stats.DifficultyCount)
	fmt.Printf("- Average answers per question: %.2f\n", stats.AvgAnswersPerQuestion)
	fmt.Printf("\nDataset saved successfully!\n")

	// Create a README for the dataset.
	readmePath := filepath.Join(filepath.Dir(*outputPath), "README.md")
	readme := `# Benchmark Dataset

This directory contains benchmark datasets for evaluating ensemble judge performance.

## Sample Dataset

The sample_benchmark_dataset.json file is a synthetic dataset generated for testing purposes only.

### Legal Notice

**IMPORTANT**: This is a synthetic dataset created for development and testing. For production benchmarks:
1. A real dataset with proper licensing must be sourced
2. Legal review must be completed to ensure license compatibility
3. The dataset must contain high-quality, diverse questions with verified ground truth

### Dataset Requirements

Per Story 2.3 requirements:
- Minimum 500 question/answer pairs
- Each question must have multiple candidate answers
- Ground truth answer must be clearly identified
- Compatible license (MIT, Apache 2.0, CC-BY, etc.)

### Recommended Real Datasets

Consider these sources for production benchmarks:
- MMLU (Massive Multitask Language Understanding)
- TruthfulQA
- HellaSwag
- ARC (AI2 Reasoning Challenge)
- OpenBookQA

Each requires legal review before use.
`

	if err := testutils.SaveBenchmarkDataset(dataset, readmePath); err != nil {
		if err := os.WriteFile(readmePath, []byte(readme), 0600); err != nil {
			log.Printf("Warning: Failed to create README: %v", err)
		}
	}
}
