package testutils

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ahrav/go-gavel/internal/domain"
)

// TestValidateBenchmarkDataset tests the validation logic for benchmark datasets.
// It covers nil datasets, metadata validation, size constraints, and data integrity.
func TestValidateBenchmarkDataset(t *testing.T) {
	tests := []struct {
		name    string
		dataset *BenchmarkDataset
		wantErr string
	}{
		{
			name:    "nil dataset",
			dataset: nil,
			wantErr: "dataset is nil",
		},
		{
			name: "missing metadata name",
			dataset: &BenchmarkDataset{
				Metadata: DatasetMetadata{
					Version: "1.0",
					License: "MIT",
					Source:  "test",
					Size:    500,
				},
				Questions: make([]BenchmarkQuestion, 500),
			},
			wantErr: "metadata validation failed: dataset name is required",
		},
		{
			name: "incompatible license",
			dataset: &BenchmarkDataset{
				Metadata: DatasetMetadata{
					Name:    "Test Dataset",
					Version: "1.0",
					License: "GPL-3.0",
					Source:  "test",
					Size:    500,
				},
				Questions: make([]BenchmarkQuestion, 500),
			},
			wantErr: "metadata validation failed: license GPL-3.0 is not in the list of compatible licenses",
		},
		{
			name: "insufficient questions",
			dataset: &BenchmarkDataset{
				Metadata: DatasetMetadata{
					Name:    "Test Dataset",
					Version: "1.0",
					License: "MIT",
					Source:  "test",
					Size:    100,
				},
				Questions: make([]BenchmarkQuestion, 100),
			},
			wantErr: "dataset must contain at least 500 questions, found 100",
		},
		{
			name: "size mismatch",
			dataset: &BenchmarkDataset{
				Metadata: DatasetMetadata{
					Name:    "Test Dataset",
					Version: "1.0",
					License: "MIT",
					Source:  "test",
					Size:    600,
				},
				Questions: createValidQuestions(500),
			},
			wantErr: "metadata size (600) doesn't match actual question count (500)",
		},
		{
			name: "duplicate question ID",
			dataset: &BenchmarkDataset{
				Metadata: DatasetMetadata{
					Name:    "Test Dataset",
					Version: "1.0",
					License: "MIT",
					Source:  "test",
					Size:    500,
				},
				Questions: createQuestionsWithDuplicateID(500),
			},
			wantErr: "duplicate question ID: q0",
		},
		{
			name: "ground truth not in answers",
			dataset: &BenchmarkDataset{
				Metadata: DatasetMetadata{
					Name:    "Test Dataset",
					Version: "1.0",
					License: "MIT",
					Source:  "test",
					Size:    500,
				},
				Questions: createQuestionsWithInvalidGroundTruth(500),
			},
			wantErr: "question q0: ground truth ID invalid not found in answers",
		},
		{
			name: "valid dataset",
			dataset: &BenchmarkDataset{
				Metadata: DatasetMetadata{
					Name:        "Test Dataset",
					Version:     "1.0",
					License:     "MIT",
					Source:      "test",
					Description: "Test dataset for validation",
					Size:        500,
				},
				Questions: createValidQuestions(500),
			},
			wantErr: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateBenchmarkDataset(tt.dataset)
			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.wantErr)
			} else {
				require.NoError(t, err)
			}
		})
	}
}

// TestComputeDatasetStatistics verifies the calculation of dataset statistics.
// It checks the counts of questions, domains, difficulties, and answer distributions.
func TestComputeDatasetStatistics(t *testing.T) {
	dataset := &BenchmarkDataset{
		Questions: []BenchmarkQuestion{
			{
				ID:         "q1",
				Question:   "Question 1",
				Domain:     "science",
				Difficulty: "easy",
				Answers: []domain.Answer{
					{ID: "a1", Content: "Answer 1"},
					{ID: "a2", Content: "Answer 2"},
				},
			},
			{
				ID:         "q2",
				Question:   "Question 2",
				Domain:     "science",
				Difficulty: "medium",
				Answers: []domain.Answer{
					{ID: "a1", Content: "Answer 1"},
					{ID: "a2", Content: "Answer 2"},
					{ID: "a3", Content: "Answer 3"},
				},
			},
			{
				ID:       "q3",
				Question: "Question 3",
				Domain:   "history",
				Answers: []domain.Answer{
					{ID: "a1", Content: "Answer 1"},
					{ID: "a2", Content: "Answer 2"},
					{ID: "a3", Content: "Answer 3"},
					{ID: "a4", Content: "Answer 4"},
				},
			},
		},
	}

	stats := ComputeDatasetStatistics(dataset)

	assert.Equal(t, 3, stats.TotalQuestions)
	assert.Equal(t, 2, stats.DomainsCount["science"])
	assert.Equal(t, 1, stats.DomainsCount["history"])
	assert.Equal(t, 1, stats.DifficultyCount["easy"])
	assert.Equal(t, 1, stats.DifficultyCount["medium"])
	assert.Equal(t, 1, stats.DifficultyCount["unspecified"])
	assert.Equal(t, 3.0, stats.AvgAnswersPerQuestion)
	assert.Equal(t, 2, stats.MinAnswers)
	assert.Equal(t, 4, stats.MaxAnswers)
}

// TestLoadSaveBenchmarkDataset tests the serialization and deserialization of a benchmark dataset.
// It ensures that a dataset can be saved to a file and loaded back without data loss.
func TestLoadSaveBenchmarkDataset(t *testing.T) {
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.json")

	// Create a small test dataset.
	// Full validation is tested separately.
	originalDataset := &BenchmarkDataset{
		Metadata: DatasetMetadata{
			Name:        "Test Dataset",
			Version:     "1.0",
			License:     "MIT",
			Source:      "unit test",
			Description: "Test dataset for load/save",
			Size:        2,
		},
		Questions: []BenchmarkQuestion{
			{
				ID:            "q1",
				Question:      "What is 2+2?",
				Domain:        "math",
				Difficulty:    "easy",
				GroundTruthID: "a1",
				Answers: []domain.Answer{
					{ID: "a1", Content: "4"},
					{ID: "a2", Content: "5"},
				},
			},
			{
				ID:            "q2",
				Question:      "What is the capital of France?",
				Domain:        "geography",
				Difficulty:    "easy",
				GroundTruthID: "a2",
				Answers: []domain.Answer{
					{ID: "a1", Content: "London"},
					{ID: "a2", Content: "Paris"},
					{ID: "a3", Content: "Berlin"},
				},
			},
		},
	}

	err := SaveBenchmarkDataset(originalDataset, datasetPath)
	require.NoError(t, err)

	_, err = os.Stat(datasetPath)
	require.NoError(t, err)

	// Load the dataset back without full validation, as it has fewer than the minimum required questions.
	loadedDataset, err := LoadBenchmarkDatasetNoValidation(datasetPath)
	require.NoError(t, err)

	assert.Equal(t, originalDataset.Metadata, loadedDataset.Metadata)
	assert.Equal(t, len(originalDataset.Questions), len(loadedDataset.Questions))

	for i, q := range originalDataset.Questions {
		assert.Equal(t, q.ID, loadedDataset.Questions[i].ID)
		assert.Equal(t, q.Question, loadedDataset.Questions[i].Question)
		assert.Equal(t, q.Domain, loadedDataset.Questions[i].Domain)
		assert.Equal(t, q.Difficulty, loadedDataset.Questions[i].Difficulty)
		assert.Equal(t, q.GroundTruthID, loadedDataset.Questions[i].GroundTruthID)
		assert.Equal(t, len(q.Answers), len(loadedDataset.Questions[i].Answers))
	}
}

// TestLoadBenchmarkDataset_Errors tests error handling when loading a benchmark dataset.
// It covers scenarios like non-existent files, invalid JSON, and failed validation.
func TestLoadBenchmarkDataset_Errors(t *testing.T) {
	tests := []struct {
		name    string
		setup   func(t *testing.T) string
		wantErr string
	}{
		{
			name: "non-existent file",
			setup: func(t *testing.T) string {
				return "/non/existent/path.json"
			},
			wantErr: "failed to read dataset file",
		},
		{
			name: "invalid JSON",
			setup: func(t *testing.T) string {
				tmpFile := filepath.Join(t.TempDir(), "invalid.json")
				err := os.WriteFile(tmpFile, []byte("not valid json"), 0644)
				require.NoError(t, err)
				return tmpFile
			},
			wantErr: "failed to parse dataset JSON",
		},
		{
			name: "invalid dataset structure",
			setup: func(t *testing.T) string {
				tmpFile := filepath.Join(t.TempDir(), "invalid_dataset.json")
				content := `{
					"metadata": {
						"name": "Test",
						"version": "1.0",
						"license": "MIT",
						"source": "test",
						"size": 100
					},
					"questions": []
				}`
				err := os.WriteFile(tmpFile, []byte(content), 0644)
				require.NoError(t, err)
				return tmpFile
			},
			wantErr: "dataset validation failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := tt.setup(t)
			_, err := LoadBenchmarkDataset(path)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantErr)
		})
	}
}

// Helper functions for creating test data.

// createValidQuestions generates a slice of valid benchmark questions.
func createValidQuestions(count int) []BenchmarkQuestion {
	questions := make([]BenchmarkQuestion, count)
	for i := 0; i < count; i++ {
		questions[i] = BenchmarkQuestion{
			ID:            fmt.Sprintf("q%d", i),
			Question:      fmt.Sprintf("Question %d", i),
			GroundTruthID: "a1",
			Answers: []domain.Answer{
				{ID: "a1", Content: "Answer 1"},
				{ID: "a2", Content: "Answer 2"},
			},
		}
	}
	return questions
}

// createQuestionsWithDuplicateID generates questions with a duplicate ID for testing validation.
func createQuestionsWithDuplicateID(count int) []BenchmarkQuestion {
	questions := createValidQuestions(count)
	if count > 1 {
		// Create a duplicate ID to test validation.
		questions[1].ID = questions[0].ID
	}
	return questions
}

// createQuestionsWithInvalidGroundTruth generates questions with an invalid ground truth ID.
func createQuestionsWithInvalidGroundTruth(count int) []BenchmarkQuestion {
	questions := createValidQuestions(count)
	if count > 0 {
		questions[0].GroundTruthID = "invalid"
	}
	return questions
}

// Tests for the improved benchmark dataset generator.

// TestGenerateSampleBenchmarkDatasetWithSeed tests the generation of sample benchmark datasets.
// It verifies dataset size, variety, and validity for different seeds and sizes.
func TestGenerateSampleBenchmarkDatasetWithSeed(t *testing.T) {
	tests := []struct {
		name string
		size int
		seed int64
	}{
		{
			name: "Small dataset with fixed seed",
			size: 10,
			seed: 12345,
		},
		{
			name: "Minimum size dataset",
			size: MinimumDatasetSize,
			seed: 67890,
		},
		{
			name: "Large dataset",
			size: 1000,
			seed: 11111,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dataset := GenerateSampleBenchmarkDataset(tt.size, tt.seed)

			require.NotNil(t, dataset)
			assert.Equal(t, tt.size, len(dataset.Questions))
			assert.Equal(t, tt.size, dataset.Metadata.Size)

			if tt.size >= MinimumDatasetSize {
				require.NoError(t, ValidateBenchmarkDataset(dataset))
			}

			domains := make(map[string]int)
			difficulties := make(map[string]int)

			for _, q := range dataset.Questions {
				domains[q.Domain]++
				difficulties[q.Difficulty]++

				assert.Equal(t, DefaultAnswerCount, len(q.Answers))

				found := false
				for _, ans := range q.Answers {
					if ans.ID == q.GroundTruthID {
						found = true
						break
					}
				}
				assert.True(t, found, "Question %s: ground truth ID %s not found", q.ID, q.GroundTruthID)
			}

			if tt.size >= 30 {
				for _, domain := range []string{DomainMath, DomainScience, DomainGeneral} {
					assert.Greater(t, domains[domain], 0, "No questions found for domain %s", domain)
				}
			}

			if tt.size >= 30 {
				for _, diff := range []string{DifficultyEasy, DifficultyMedium, DifficultyHard} {
					assert.Greater(t, difficulties[diff], 0, "No questions found for difficulty %s", diff)
				}
			}
		})
	}
}

// TestGenerateSampleBenchmarkDatasetDeterministic ensures that datasets generated with the same seed are identical.
func TestGenerateSampleBenchmarkDatasetDeterministic(t *testing.T) {
	seed := int64(42)
	size := 100

	dataset1 := GenerateSampleBenchmarkDataset(size, seed)
	dataset2 := GenerateSampleBenchmarkDataset(size, seed)

	require.Equal(t, len(dataset1.Questions), len(dataset2.Questions))

	for i := range dataset1.Questions {
		q1 := dataset1.Questions[i]
		q2 := dataset2.Questions[i]

		assert.Equal(t, q1.ID, q2.ID)
		assert.Equal(t, q1.Question, q2.Question)
		assert.Equal(t, q1.Domain, q2.Domain)
		assert.Equal(t, q1.Difficulty, q2.Difficulty)
		assert.Equal(t, q1.GroundTruthID, q2.GroundTruthID)

		require.Equal(t, len(q1.Answers), len(q2.Answers))

		for j := range q1.Answers {
			assert.Equal(t, q1.Answers[j].ID, q2.Answers[j].ID)
			assert.Equal(t, q1.Answers[j].Content, q2.Answers[j].Content)
		}
	}
}

// TestContentVariety checks for a diverse range of questions and answers in the generated dataset.
// It ensures that the dataset is not repetitive and covers various topics.
func TestContentVariety(t *testing.T) {
	dataset := GenerateSampleBenchmarkDataset(500, 99999)

	uniqueQuestions := make(map[string]bool)
	for _, q := range dataset.Questions {
		uniqueQuestions[q.Question] = true
	}

	// Ensure a reasonable variety of unique questions to avoid repetition.
	uniqueRatio := float64(len(uniqueQuestions)) / float64(len(dataset.Questions))
	assert.Greater(t, uniqueRatio, 0.2, "Low question variety: only %.2f%% unique questions", uniqueRatio*100)

	mathOperations := make(map[string]int)
	for _, q := range dataset.Questions {
		if q.Domain == DomainMath {
			if strings.Contains(q.Question, "+") || strings.Contains(q.Question, "get") {
				mathOperations["addition"]++
			} else if strings.Contains(q.Question, "×") {
				mathOperations["multiplication"]++
			} else if strings.Contains(q.Question, "-") {
				mathOperations["subtraction"]++
			} else if strings.Contains(q.Question, "÷") {
				mathOperations["division"]++
			}
		}
	}

	// Ensure a variety of math operations are present in the dataset.
	assert.GreaterOrEqual(t, len(mathOperations), 3, "Low math operation variety: only %d different operations found", len(mathOperations))
}

// TestImprovedLicenseValidation tests the license compatibility check.
// It verifies that the license validation is case-insensitive and handles various formats.
func TestImprovedLicenseValidation(t *testing.T) {
	tests := []struct {
		license string
		valid   bool
	}{
		{"MIT", true},
		{"mit", true},
		{"  MIT  ", true},
		{"Apache-2.0", true},
		{"apache 2.0", true},
		{"CC-BY", true},
		{"cc-by-4.0", true},
		{"GPL", false},
		{"Proprietary", false},
		{"", false},
	}

	for _, tt := range tests {
		t.Run(tt.license, func(t *testing.T) {
			result := isCompatibleLicense(tt.license)
			assert.Equal(t, tt.valid, result, "isCompatibleLicense(%q) = %v, want %v", tt.license, result, tt.valid)
		})
	}
}

// TestDuplicateAnswerContentValidation ensures that duplicate answer content within a single question is detected.
func TestDuplicateAnswerContentValidation(t *testing.T) {
	dataset := &BenchmarkDataset{
		Metadata: DatasetMetadata{
			Name:    "Test Dataset",
			Version: "1.0",
			License: "MIT",
			Source:  "test",
			Size:    500,
		},
		Questions: createValidQuestions(500),
	}

	// Add duplicate answer content to the first question to test validation.
	dataset.Questions[0].Answers = []domain.Answer{
		{ID: "a1", Content: "Same Answer"},
		{ID: "a2", Content: "Same Answer"},
		{ID: "a3", Content: "Different"},
	}

	err := ValidateBenchmarkDataset(dataset)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "duplicate answer content")
}

// TestAnswerDistributionValidation verifies that questions have a valid number of answers.
func TestAnswerDistributionValidation(t *testing.T) {
	dataset := &BenchmarkDataset{
		Metadata: DatasetMetadata{
			Name:    "Test Dataset",
			Version: "1.0",
			License: "MIT",
			Source:  "test",
			Size:    500,
		},
		Questions: createValidQuestions(500),
	}

	// Make one question have only one answer to test validation.
	dataset.Questions[0].Answers = []domain.Answer{
		{ID: "a1", Content: "Only Answer"},
	}

	err := ValidateBenchmarkDataset(dataset)
	require.Error(t, err)
	// The error originates from individual question validation, not the distribution check.
	assert.Contains(t, err.Error(), "question must have at least 2 candidate answers")
}

// TestGenerateSafeFakeSymbols tests the generation of safe fake chemical symbols.
// It ensures that the generated symbols are different from the real one.
func TestGenerateSafeFakeSymbols(t *testing.T) {
	tests := []struct {
		name     string
		symbol   string
		expected int // Expected number of fake symbols
	}{
		{
			name:     "Two letter symbol",
			symbol:   "He",
			expected: 3,
		},
		{
			name:     "Single letter symbol",
			symbol:   "H",
			expected: 3,
		},
		{
			name:     "Symbol with lowercase",
			symbol:   "Na",
			expected: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fakes := GenerateSafeFakeSymbols(tt.symbol)
			assert.Equal(t, tt.expected, len(fakes))

			for _, fake := range fakes {
				assert.NotEqual(t, tt.symbol, fake)
			}
		})
	}
}
