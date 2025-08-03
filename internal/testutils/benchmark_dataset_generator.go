// Package testutils provides utilities for testing, including mock objects and
// test data generators. These components are intended for internal use within
// the project's test suites and are not part of the public API.
package testutils

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/ahrav/go-gavel/internal/domain"
)

// GenerateSampleBenchmarkDataset creates a sample benchmark dataset for testing.
// The seed parameter controls randomization - use time.Now().UnixNano() for
// non-deterministic generation or a fixed value for reproducible tests.
// NOTE: This is for testing purposes only. For production benchmarks, a real
// dataset with proper licensing must be sourced.
func GenerateSampleBenchmarkDataset(size int, seed int64) *BenchmarkDataset {
	rng := rand.New(rand.NewSource(seed))

	dataset := &BenchmarkDataset{
		Metadata: DatasetMetadata{
			Name:        "Sample Benchmark Dataset",
			Version:     "1.0.0",
			License:     "MIT",
			Source:      "Generated for testing",
			Description: "A sample dataset generated for testing ensemble performance. NOT FOR PRODUCTION USE.",
			Size:        size,
		},
		Questions: make([]BenchmarkQuestion, 0, size),
	}

	domains := []string{DomainMath, DomainScience, DomainGeneral}
	difficulties := []string{DifficultyEasy, DifficultyMedium, DifficultyHard}

	for i := range size {
		domainName := domains[rng.Intn(len(domains))]
		difficulty := difficulties[rng.Intn(len(difficulties))]

		q := generateQuestionForDomain(rng, i, domainName, difficulty)
		dataset.Questions = append(dataset.Questions, q)
	}

	return dataset
}

// GenerateSampleBenchmarkDatasetDefault creates a dataset with a time-based seed
func GenerateSampleBenchmarkDatasetDefault(size int) *BenchmarkDataset {
	return GenerateSampleBenchmarkDataset(size, time.Now().UnixNano())
}

func generateQuestionForDomain(rng *rand.Rand, index int, domainName, difficulty string) BenchmarkQuestion {
	switch domainName {
	case DomainMath:
		return generateMathQuestion(rng, index, difficulty)
	case DomainScience:
		return generateScienceQuestion(rng, index, difficulty)
	default:
		return generateGeneralQuestion(rng, index, difficulty)
	}
}

func generateMathQuestion(rng *rand.Rand, index int, difficulty string) BenchmarkQuestion {
	template := MathQuestionTemplates[rng.Intn(len(MathQuestionTemplates))]

	params := template.GenerateParams(rng, difficulty)

	var correct int
	switch template.Category {
	case "addition", "word_problem_addition":
		correct = params[0].(int) + params[1].(int)
	case "multiplication":
		correct = params[0].(int) * params[1].(int)
	case "subtraction":
		correct = params[0].(int) - params[1].(int)
	case "division":
		correct = params[0].(int) / params[1].(int)
	}

	question := fmt.Sprintf(template.Format, params...)

	distractors := template.GenerateDistractors(rng, correct, params)

	answers := []domain.Answer{
		{ID: fmt.Sprintf("q%d_a1", index), Content: fmt.Sprintf("%d", correct)},
	}

	// Add distractors as answers
	for i, distractor := range distractors {
		answers = append(answers, domain.Answer{
			ID:      fmt.Sprintf("q%d_a%d", index, i+2),
			Content: distractor,
		})
	}

	// Shuffle answers
	rng.Shuffle(len(answers), func(i, j int) {
		answers[i], answers[j] = answers[j], answers[i]
	})

	// Find correct answer ID after shuffle
	var groundTruthID string
	for _, ans := range answers {
		if ans.Content == fmt.Sprintf("%d", correct) {
			groundTruthID = ans.ID
			break
		}
	}

	return BenchmarkQuestion{
		ID:            fmt.Sprintf("q%d", index),
		Question:      question,
		Answers:       answers,
		GroundTruthID: groundTruthID,
		Domain:        DomainMath,
		Difficulty:    difficulty,
	}
}

func generateScienceQuestion(rng *rand.Rand, index int, difficulty string) BenchmarkQuestion {
	// Randomly choose science question type
	questionType := rng.Intn(4) // 0: element, 1: planet, 2: biology, 3: mixed

	switch questionType {
	case 0:
		return generateElementQuestion(rng, index, difficulty)
	case 1:
		return generatePlanetQuestion(rng, index, difficulty)
	case 2:
		return generateBiologyQuestion(rng, index, difficulty)
	default:
		return generateMixedScienceQuestion(rng, index, difficulty)
	}
}

func generateElementQuestion(rng *rand.Rand, index int, difficulty string) BenchmarkQuestion {
	element := GetRandomElement(rng, index)

	var question string
	var correctAnswer string
	var wrongAnswers []string

	// Choose question type based on difficulty
	switch difficulty {
	case DifficultyEasy:
		// Symbol questions for easy
		question = fmt.Sprintf("What is the chemical symbol for %s?", element.Name)
		correctAnswer = element.Symbol
		wrongAnswers = GenerateSafeFakeSymbols(element.Symbol)
	case DifficultyMedium:
		// Atomic number questions for medium
		question = fmt.Sprintf("What is the atomic number of %s?", element.Name)
		correctAnswer = fmt.Sprintf("%d", element.Number)
		wrongAnswers = []string{
			fmt.Sprintf("%d", element.Number+rng.Intn(3)+1),
			fmt.Sprintf("%d", element.Number-rng.Intn(3)-1),
			fmt.Sprintf("%d", element.Number+rng.Intn(5)+5),
		}
	case DifficultyHard:
		// Group questions for hard
		question = fmt.Sprintf("Which group does %s belong to?", element.Name)
		correctAnswer = element.Group
		wrongAnswers = generateGroupDistractors(element.Group)
	}

	return createScienceQuestion(rng, index, question, correctAnswer, wrongAnswers, difficulty)
}

func generatePlanetQuestion(rng *rand.Rand, index int, difficulty string) BenchmarkQuestion {
	planet := GetRandomPlanet(rng, index)

	var question string
	var correctAnswer string
	var wrongAnswers []string

	switch rng.Intn(3) {
	case 0:
		// Order questions
		question = fmt.Sprintf("What is the position of %s from the Sun?", planet.Name)
		correctAnswer = fmt.Sprintf("%d", planet.Order)
		wrongAnswers = generateOrderDistractors(planet.Order)
	case 1:
		// Type questions
		question = fmt.Sprintf("What type of planet is %s?", planet.Name)
		correctAnswer = planet.Type
		wrongAnswers = generatePlanetTypeDistractors(planet.Type)
	default:
		// Moon questions
		question = fmt.Sprintf("Does %s have rings?", planet.Name)
		correctAnswer = fmt.Sprintf("%v", planet.RingsYes)
		if planet.RingsYes {
			wrongAnswers = []string{"false", "no", "none"}
		} else {
			wrongAnswers = []string{"true", "yes", "many"}
		}
	}

	return createScienceQuestion(rng, index, question, correctAnswer, wrongAnswers, difficulty)
}

func generateBiologyQuestion(rng *rand.Rand, index int, difficulty string) BenchmarkQuestion {
	term := GetRandomBiologyTerm(rng, index)

	question := fmt.Sprintf("Which of the following best describes %s?", term.Term)
	correctAnswer := term.Definition

	// Get wrong answers from other biology terms
	wrongAnswers := []string{}
	for _, other := range BiologyTerms {
		if other.Term != term.Term && len(wrongAnswers) < 3 {
			wrongAnswers = append(wrongAnswers, other.Definition)
		}
	}

	// Shuffle wrong answers
	rng.Shuffle(len(wrongAnswers), func(i, j int) {
		wrongAnswers[i], wrongAnswers[j] = wrongAnswers[j], wrongAnswers[i]
	})

	return createScienceQuestion(rng, index, question, correctAnswer, wrongAnswers[:3], difficulty)
}

func generateMixedScienceQuestion(rng *rand.Rand, index int, difficulty string) BenchmarkQuestion {
	// Mix different types of science questions
	switch rng.Intn(3) {
	case 0:
		return generateElementQuestion(rng, index, difficulty)
	case 1:
		return generatePlanetQuestion(rng, index, difficulty)
	default:
		return generateBiologyQuestion(rng, index, difficulty)
	}
}

func generateGeneralQuestion(rng *rand.Rand, index int, difficulty string) BenchmarkQuestion {
	concept := GetRandomGeneralConcept(rng, index)
	question := fmt.Sprintf("Which of the following best describes %s?", concept.Term)

	// Create answers
	answers := []domain.Answer{
		{ID: fmt.Sprintf("q%d_a1", index), Content: concept.Correct},
	}

	// Shuffle wrong answers and select 3
	wrong := make([]string, len(concept.Wrong))
	copy(wrong, concept.Wrong)
	rng.Shuffle(len(wrong), func(i, j int) {
		wrong[i], wrong[j] = wrong[j], wrong[i]
	})

	for i := 0; i < 3 && i < len(wrong); i++ {
		answers = append(answers, domain.Answer{
			ID:      fmt.Sprintf("q%d_a%d", index, i+2),
			Content: wrong[i],
		})
	}

	// Shuffle all answers
	rng.Shuffle(len(answers), func(i, j int) {
		answers[i], answers[j] = answers[j], answers[i]
	})

	// Find correct answer ID after shuffle
	var groundTruthID string
	for _, ans := range answers {
		if ans.Content == concept.Correct {
			groundTruthID = ans.ID
			break
		}
	}

	return BenchmarkQuestion{
		ID:            fmt.Sprintf("q%d", index),
		Question:      question,
		Answers:       answers,
		GroundTruthID: groundTruthID,
		Domain:        DomainGeneral,
		Difficulty:    difficulty,
	}
}

// Helper function to create a science question
func createScienceQuestion(
	rng *rand.Rand,
	index int,
	question,
	correctAnswer string,
	wrongAnswers []string,
	difficulty string,
) BenchmarkQuestion {
	// Create answers
	answers := []domain.Answer{
		{ID: fmt.Sprintf("q%d_a1", index), Content: correctAnswer},
	}

	for i, wrong := range wrongAnswers {
		if i < 3 { // Ensure we only have 4 answers total
			answers = append(answers, domain.Answer{
				ID:      fmt.Sprintf("q%d_a%d", index, i+2),
				Content: wrong,
			})
		}
	}

	// Shuffle answers using the provided RNG
	rng.Shuffle(len(answers), func(i, j int) {
		answers[i], answers[j] = answers[j], answers[i]
	})

	// Find correct answer ID after shuffle
	var groundTruthID string
	for _, ans := range answers {
		if ans.Content == correctAnswer {
			groundTruthID = ans.ID
			break
		}
	}

	return BenchmarkQuestion{
		ID:            fmt.Sprintf("q%d", index),
		Question:      question,
		Answers:       answers,
		GroundTruthID: groundTruthID,
		Domain:        DomainScience,
		Difficulty:    difficulty,
	}
}

// Helper functions for generating distractors
func generateGroupDistractors(correctGroup string) []string {
	groups := []string{
		"Noble Gas", "Alkali Metal", "Alkaline Earth Metal",
		"Transition Metal", "Post-transition Metal", "Metalloid",
		"Nonmetal", "Halogen", "Lanthanide", "Actinide",
	}

	distractors := []string{}
	for _, group := range groups {
		if group != correctGroup && len(distractors) < 3 {
			distractors = append(distractors, group)
		}
	}

	return distractors
}

func generateOrderDistractors(correct int) []string {
	distractors := []string{}

	// Add nearby positions
	if correct > 1 {
		distractors = append(distractors, fmt.Sprintf("%d", correct-1))
	}
	if correct < 8 {
		distractors = append(distractors, fmt.Sprintf("%d", correct+1))
	}

	// Add some other positions
	for i := 1; i <= 8 && len(distractors) < 3; i++ {
		if i != correct && i != correct-1 && i != correct+1 {
			distractors = append(distractors, fmt.Sprintf("%d", i))
		}
	}

	return distractors[:3]
}

func generatePlanetTypeDistractors(correct string) []string {
	types := []string{"Terrestrial", "Gas Giant", "Ice Giant", "Dwarf Planet"}
	distractors := []string{}

	for _, t := range types {
		if t != correct && len(distractors) < 3 {
			distractors = append(distractors, t)
		}
	}

	return distractors
}
