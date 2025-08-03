package testutils

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ahrav/go-gavel/internal/domain"
)

// BenchmarkDataset represents a collection of questions and answers for
// benchmarking judge performance. It includes metadata about the dataset
// source and licensing.
type BenchmarkDataset struct {
	// Questions contains all benchmark questions with their candidate answers.
	Questions []BenchmarkQuestion `json:"questions"`

	// Metadata provides information about the dataset itself.
	Metadata DatasetMetadata `json:"metadata"`
}

// BenchmarkQuestion represents a single question in the benchmark dataset
// with multiple candidate answers and a known correct answer.
type BenchmarkQuestion struct {
	// ID uniquely identifies this question in the dataset.
	ID string `json:"id"`

	// Question is the text of the question being asked.
	Question string `json:"question"`

	// Answers contains all candidate answers for this question.
	Answers []domain.Answer `json:"candidate_answers"`

	// GroundTruthID identifies which answer ID is correct.
	GroundTruthID string `json:"ground_truth_answer_id"`

	// Domain categorizes the question (e.g., "science", "history").
	Domain string `json:"domain,omitempty"`

	// Difficulty indicates the question difficulty level.
	Difficulty string `json:"difficulty,omitempty"`
}

// DatasetMetadata contains information about the benchmark dataset itself,
// including licensing and provenance information.
type DatasetMetadata struct {
	// Name identifies the dataset.
	Name string `json:"name"`

	// Version tracks dataset revisions.
	Version string `json:"version"`

	// License specifies the dataset's license (must be compatible).
	License string `json:"license"`

	// Source indicates where the dataset originated.
	Source string `json:"source"`

	// Description provides details about the dataset contents.
	Description string `json:"description"`

	// Size indicates the total number of questions.
	Size int `json:"question_count"`
}

// LoadBenchmarkDataset loads a benchmark dataset from a JSON file.
// It validates the dataset structure and ensures all required fields are present.
func LoadBenchmarkDataset(path string) (*BenchmarkDataset, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read dataset file: %w", err)
	}

	var dataset BenchmarkDataset
	if err := json.Unmarshal(data, &dataset); err != nil {
		return nil, fmt.Errorf("failed to parse dataset JSON: %w", err)
	}

	if err := ValidateBenchmarkDataset(&dataset); err != nil {
		return nil, fmt.Errorf("dataset validation failed: %w", err)
	}

	return &dataset, nil
}

// LoadBenchmarkDatasetNoValidation loads a benchmark dataset from a JSON file
// without full validation. This is primarily for testing purposes.
func LoadBenchmarkDatasetNoValidation(path string) (*BenchmarkDataset, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read dataset file: %w", err)
	}

	var dataset BenchmarkDataset
	if err := json.Unmarshal(data, &dataset); err != nil {
		return nil, fmt.Errorf("failed to parse dataset JSON: %w", err)
	}

	return &dataset, nil
}

// ValidateBenchmarkDataset ensures a dataset meets all requirements for benchmarking.
// It checks for completeness, consistency, and minimum size requirements.
func ValidateBenchmarkDataset(dataset *BenchmarkDataset) error {
	if dataset == nil {
		return fmt.Errorf("dataset is nil")
	}

	if err := validateMetadata(&dataset.Metadata); err != nil {
		return fmt.Errorf("metadata validation failed: %w", err)
	}

	if len(dataset.Questions) < MinimumDatasetSize {
		return fmt.Errorf("dataset must contain at least %d questions, found %d",
			MinimumDatasetSize, len(dataset.Questions))
	}

	seenIDs := make(map[string]bool)
	for i, q := range dataset.Questions {
		if err := validateQuestion(&q, i); err != nil {
			return fmt.Errorf("question %d validation failed: %w", i, err)
		}

		if seenIDs[q.ID] {
			return fmt.Errorf("duplicate question ID: %s", q.ID)
		}
		seenIDs[q.ID] = true

		// Validate ground truth exists in answers
		found := false
		for _, ans := range q.Answers {
			if ans.ID == q.GroundTruthID {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("question %s: ground truth ID %s not found in answers", q.ID, q.GroundTruthID)
		}

		seenContent := make(map[string]bool)
		for _, ans := range q.Answers {
			if seenContent[ans.Content] {
				return fmt.Errorf("question %s has duplicate answer content: %s", q.ID, ans.Content)
			}
			seenContent[ans.Content] = true
		}
	}

	// Validate metadata size matches actual size
	if dataset.Metadata.Size != len(dataset.Questions) {
		return fmt.Errorf("metadata size (%d) doesn't match actual question count (%d)",
			dataset.Metadata.Size, len(dataset.Questions))
	}

	// Validate answer distribution
	stats := ComputeDatasetStatistics(dataset)
	if stats.MinAnswers < MinimumAnswerCount {
		return fmt.Errorf("all questions must have at least %d answers, found question with %d",
			MinimumAnswerCount, stats.MinAnswers)
	}

	return nil
}

// validateMetadata ensures dataset metadata is complete and valid.
func validateMetadata(meta *DatasetMetadata) error {
	if meta.Name == "" {
		return fmt.Errorf("dataset name is required")
	}
	if meta.Version == "" {
		return fmt.Errorf("dataset version is required")
	}
	if meta.License == "" {
		return fmt.Errorf("dataset license is required")
	}
	if meta.Source == "" {
		return fmt.Errorf("dataset source is required")
	}
	if meta.Size <= 0 {
		return fmt.Errorf("dataset size must be positive")
	}

	// Validate license compatibility
	if !isCompatibleLicense(meta.License) {
		return fmt.Errorf("license %s is not in the list of compatible licenses", meta.License)
	}

	return nil
}

// validateQuestion ensures a single question is properly structured.
func validateQuestion(q *BenchmarkQuestion, index int) error {
	if q.ID == "" {
		return fmt.Errorf("question ID is required")
	}
	if q.Question == "" {
		return fmt.Errorf("question text is required")
	}
	if len(q.Answers) < 2 {
		return fmt.Errorf("question must have at least 2 candidate answers, found %d", len(q.Answers))
	}
	if q.GroundTruthID == "" {
		return fmt.Errorf("ground truth answer ID is required")
	}

	// Validate each answer
	seenAnswerIDs := make(map[string]bool)
	for j, ans := range q.Answers {
		if ans.ID == "" {
			return fmt.Errorf("answer %d: ID is required", j)
		}
		if ans.Content == "" {
			return fmt.Errorf("answer %d: content is required", j)
		}
		if seenAnswerIDs[ans.ID] {
			return fmt.Errorf("duplicate answer ID: %s", ans.ID)
		}
		seenAnswerIDs[ans.ID] = true
	}

	return nil
}

// DatasetStatistics provides summary statistics about a benchmark dataset.
type DatasetStatistics struct {
	// TotalQuestions is the number of questions in the dataset.
	TotalQuestions int

	// DomainsCount maps domain names to question counts.
	DomainsCount map[string]int

	// DifficultyCount maps difficulty levels to question counts.
	DifficultyCount map[string]int

	// AvgAnswersPerQuestion is the average number of candidate answers.
	AvgAnswersPerQuestion float64

	// MinAnswers is the minimum number of answers for any question.
	MinAnswers int

	// MaxAnswers is the maximum number of answers for any question.
	MaxAnswers int
}

// ComputeDatasetStatistics analyzes a benchmark dataset and returns summary statistics.
func ComputeDatasetStatistics(dataset *BenchmarkDataset) *DatasetStatistics {
	stats := &DatasetStatistics{
		TotalQuestions:  len(dataset.Questions),
		DomainsCount:    make(map[string]int),
		DifficultyCount: make(map[string]int),
		MinAnswers:      int(^uint(0) >> 1), // Max int
		MaxAnswers:      0,
	}

	totalAnswers := 0
	for _, q := range dataset.Questions {
		if q.Domain != "" {
			stats.DomainsCount[q.Domain]++
		} else {
			stats.DomainsCount["unspecified"]++
		}

		if q.Difficulty != "" {
			stats.DifficultyCount[q.Difficulty]++
		} else {
			stats.DifficultyCount["unspecified"]++
		}

		ansCount := len(q.Answers)
		totalAnswers += ansCount
		if ansCount < stats.MinAnswers {
			stats.MinAnswers = ansCount
		}
		if ansCount > stats.MaxAnswers {
			stats.MaxAnswers = ansCount
		}
	}

	if stats.TotalQuestions > 0 {
		stats.AvgAnswersPerQuestion = float64(totalAnswers) / float64(stats.TotalQuestions)
	}

	return stats
}

// isCompatibleLicense checks if a license is compatible for benchmark use.
// It performs case-insensitive matching and handles common variations.
func isCompatibleLicense(license string) bool {
	normalized := strings.ToLower(strings.TrimSpace(license))
	return CompatibleLicenses[normalized]
}

// SaveBenchmarkDataset writes a benchmark dataset to a JSON file.
func SaveBenchmarkDataset(dataset *BenchmarkDataset, path string) error {
	// Ensure the directory exists
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Marshal to JSON with indentation
	data, err := json.MarshalIndent(dataset, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal dataset: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write dataset file: %w", err)
	}

	return nil
}
