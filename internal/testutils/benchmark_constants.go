package testutils

// Dataset size constants
const (
	// MinimumDatasetSize is the minimum number of questions required for a valid benchmark dataset
	MinimumDatasetSize = 500

	// DefaultAnswerCount is the standard number of answer options per question
	DefaultAnswerCount = 4

	// MinimumAnswerCount is the minimum number of answer options required
	MinimumAnswerCount = 2
)

// Difficulty range constants for math questions
const (
	// Easy difficulty ranges
	EasyMin = 1
	EasyMax = 10

	// Medium difficulty ranges
	MediumMin = 10
	MediumMax = 50

	// Hard difficulty ranges
	HardMin = 50
	HardMax = 100
)

// Domain identifiers
const (
	DomainMath    = "math"
	DomainScience = "science"
	DomainGeneral = "general"
)

// Difficulty levels
const (
	DifficultyEasy   = "easy"
	DifficultyMedium = "medium"
	DifficultyHard   = "hard"
)

// Compatible licenses for benchmark datasets
var CompatibleLicenses = map[string]bool{
	"mit":           true,
	"apache-2.0":    true,
	"apache 2.0":    true,
	"cc-by":         true,
	"cc-by-4.0":     true,
	"cc0":           true,
	"public domain": true,
	"bsd":           true,
	"bsd-3-clause":  true,
}
