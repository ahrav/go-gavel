package testutils

import (
	"fmt"
	"math/rand"
)

// QuestionTemplate defines the structure for generating varied questions
type QuestionTemplate struct {
	// Format is the question template string with placeholders
	Format string

	// GenerateParams generates the parameters for the template based on difficulty
	GenerateParams func(rng *rand.Rand, difficulty string) []any

	// GenerateDistractors creates plausible wrong answers based on the correct answer
	GenerateDistractors func(rng *rand.Rand, correct any, params []any) []string

	// Category helps organize questions
	Category string
}

// MathQuestionTemplates provides various math question formats
var MathQuestionTemplates = []QuestionTemplate{
	{
		Format:   "What is %d + %d?",
		Category: "addition",
		GenerateParams: func(rng *rand.Rand, difficulty string) []any {
			a, b := generateMathOperands(rng, difficulty)
			return []any{a, b}
		},
		GenerateDistractors: func(rng *rand.Rand, correct any, params []any) []string {
			c := correct.(int)
			a, b := params[0].(int), params[1].(int)
			return generateAdditionDistractors(rng, c, a, b)
		},
	},
	{
		Format:   "What is %d × %d?",
		Category: "multiplication",
		GenerateParams: func(rng *rand.Rand, difficulty string) []any {
			a, b := generateMathOperands(rng, difficulty)
			return []any{a, b}
		},
		GenerateDistractors: func(rng *rand.Rand, correct any, params []any) []string {
			c := correct.(int)
			a, b := params[0].(int), params[1].(int)
			return generateMultiplicationDistractors(rng, c, a, b)
		},
	},
	{
		Format:   "If you have %d items and get %d more, how many do you have in total?",
		Category: "word_problem_addition",
		GenerateParams: func(rng *rand.Rand, difficulty string) []any {
			a, b := generateMathOperands(rng, difficulty)
			return []any{a, b}
		},
		GenerateDistractors: func(rng *rand.Rand, correct any, params []any) []string {
			c := correct.(int)
			a, b := params[0].(int), params[1].(int)
			return generateAdditionDistractors(rng, c, a, b)
		},
	},
	{
		Format:   "What is %d - %d?",
		Category: "subtraction",
		GenerateParams: func(rng *rand.Rand, difficulty string) []any {
			a, b := generateSubtractionOperands(rng, difficulty)
			return []any{a, b}
		},
		GenerateDistractors: func(rng *rand.Rand, correct any, params []any) []string {
			c := correct.(int)
			return generateSubtractionDistractors(rng, c)
		},
	},
	{
		Format:   "What is %d ÷ %d?",
		Category: "division",
		GenerateParams: func(rng *rand.Rand, difficulty string) []any {
			// Generate division problems with whole number results
			b := rng.Intn(9) + 2 // divisor between 2-10
			quotient := generateSingleOperand(rng, difficulty)
			a := b * quotient // ensure clean division
			return []any{a, b}
		},
		GenerateDistractors: func(rng *rand.Rand, correct any, params []any) []string {
			c := correct.(int)
			return generateDivisionDistractors(rng, c)
		},
	},
}

// Helper functions for generating operands
func generateMathOperands(rng *rand.Rand, difficulty string) (int, int) {
	var min, max int
	switch difficulty {
	case DifficultyEasy:
		min, max = EasyMin, EasyMax
	case DifficultyMedium:
		min, max = MediumMin, MediumMax
	case DifficultyHard:
		min, max = HardMin, HardMax
	default:
		min, max = EasyMin, EasyMax
	}

	a := rng.Intn(max-min) + min
	b := rng.Intn(max-min) + min
	return a, b
}

func generateSingleOperand(rng *rand.Rand, difficulty string) int {
	var min, max int
	switch difficulty {
	case DifficultyEasy:
		min, max = EasyMin, EasyMax
	case DifficultyMedium:
		min, max = MediumMin, MediumMax
	case DifficultyHard:
		min, max = HardMin, HardMax
	default:
		min, max = EasyMin, EasyMax
	}
	return rng.Intn(max-min) + min
}

func generateSubtractionOperands(rng *rand.Rand, difficulty string) (int, int) {
	a, b := generateMathOperands(rng, difficulty)
	// Ensure positive result
	if a < b {
		a, b = b, a
	}
	return a, b
}

// Distractor generation functions
func generateAdditionDistractors(rng *rand.Rand, correct, a, b int) []string {
	distractors := []string{}

	// Common mistakes
	distractors = append(distractors, fmt.Sprintf("%d", correct+rng.Intn(5)+1)) // Small overshoot
	distractors = append(distractors, fmt.Sprintf("%d", correct-rng.Intn(5)-1)) // Small undershoot
	distractors = append(distractors, fmt.Sprintf("%d", a*b))                   // Multiplication instead
	distractors = append(distractors, fmt.Sprintf("%d", correct+10))            // Off by 10
	distractors = append(distractors, fmt.Sprintf("%d", absInt(a-b)))           // Subtraction instead

	// Ensure we have enough unique distractors
	return selectUniqueDistractors(distractors, fmt.Sprintf("%d", correct), 3)
}

func generateMultiplicationDistractors(rng *rand.Rand, correct, a, b int) []string {
	distractors := []string{}

	// Common multiplication mistakes
	distractors = append(distractors, fmt.Sprintf("%d", correct+rng.Intn(10)+1))     // Small error
	distractors = append(distractors, fmt.Sprintf("%d", a+b))                        // Addition instead
	distractors = append(distractors, fmt.Sprintf("%d", correct+(rng.Intn(5)+1)*10)) // Magnitude error
	distractors = append(distractors, fmt.Sprintf("%d", (a+1)*b))                    // Off-by-one error
	distractors = append(distractors, fmt.Sprintf("%d", a*(b+1)))                    // Off-by-one error

	return selectUniqueDistractors(distractors, fmt.Sprintf("%d", correct), 3)
}

func generateSubtractionDistractors(rng *rand.Rand, correct int) []string {
	distractors := []string{}

	distractors = append(distractors, fmt.Sprintf("%d", correct+rng.Intn(5)+1))
	distractors = append(distractors, fmt.Sprintf("%d", correct-rng.Intn(5)-1))
	distractors = append(distractors, fmt.Sprintf("%d", correct+10))
	distractors = append(distractors, fmt.Sprintf("%d", absInt(correct)))

	return selectUniqueDistractors(distractors, fmt.Sprintf("%d", correct), 3)
}

func generateDivisionDistractors(rng *rand.Rand, correct int) []string {
	distractors := []string{}

	distractors = append(distractors, fmt.Sprintf("%d", correct+1))
	distractors = append(distractors, fmt.Sprintf("%d", correct-1))
	distractors = append(distractors, fmt.Sprintf("%d", correct*2))
	distractors = append(distractors, fmt.Sprintf("%d", correct/2+1))

	return selectUniqueDistractors(distractors, fmt.Sprintf("%d", correct), 3)
}

// Helper function to select unique distractors
func selectUniqueDistractors(candidates []string, correct string, count int) []string {
	seen := make(map[string]bool)
	seen[correct] = true

	result := []string{}
	for _, candidate := range candidates {
		if !seen[candidate] && len(result) < count {
			seen[candidate] = true
			result = append(result, candidate)
		}
	}

	// Fill remaining slots with generated values if needed
	for len(result) < count {
		fake := fmt.Sprintf("X%d", len(result))
		if !seen[fake] {
			result = append(result, fake)
		}
	}

	return result
}

func absInt(n int) int {
	if n < 0 {
		return -n
	}
	return n
}
