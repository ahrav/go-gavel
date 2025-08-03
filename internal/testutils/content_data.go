package testutils

import (
	"fmt"
	"math/rand"
	"strings"
	"unicode"
)

// Element represents a chemical element with its properties
type Element struct {
	Name   string
	Symbol string
	Number int
	Group  string // e.g., "Noble Gas", "Alkali Metal"
}

// ScienceElements contains an expanded list of chemical elements
var ScienceElements = []Element{
	// First 20 elements plus some common ones
	{Name: "Hydrogen", Symbol: "H", Number: 1, Group: "Nonmetal"},
	{Name: "Helium", Symbol: "He", Number: 2, Group: "Noble Gas"},
	{Name: "Lithium", Symbol: "Li", Number: 3, Group: "Alkali Metal"},
	{Name: "Beryllium", Symbol: "Be", Number: 4, Group: "Alkaline Earth Metal"},
	{Name: "Boron", Symbol: "B", Number: 5, Group: "Metalloid"},
	{Name: "Carbon", Symbol: "C", Number: 6, Group: "Nonmetal"},
	{Name: "Nitrogen", Symbol: "N", Number: 7, Group: "Nonmetal"},
	{Name: "Oxygen", Symbol: "O", Number: 8, Group: "Nonmetal"},
	{Name: "Fluorine", Symbol: "F", Number: 9, Group: "Halogen"},
	{Name: "Neon", Symbol: "Ne", Number: 10, Group: "Noble Gas"},
	{Name: "Sodium", Symbol: "Na", Number: 11, Group: "Alkali Metal"},
	{Name: "Magnesium", Symbol: "Mg", Number: 12, Group: "Alkaline Earth Metal"},
	{Name: "Aluminum", Symbol: "Al", Number: 13, Group: "Post-transition Metal"},
	{Name: "Silicon", Symbol: "Si", Number: 14, Group: "Metalloid"},
	{Name: "Phosphorus", Symbol: "P", Number: 15, Group: "Nonmetal"},
	{Name: "Sulfur", Symbol: "S", Number: 16, Group: "Nonmetal"},
	{Name: "Chlorine", Symbol: "Cl", Number: 17, Group: "Halogen"},
	{Name: "Argon", Symbol: "Ar", Number: 18, Group: "Noble Gas"},
	{Name: "Potassium", Symbol: "K", Number: 19, Group: "Alkali Metal"},
	{Name: "Calcium", Symbol: "Ca", Number: 20, Group: "Alkaline Earth Metal"},
	// Common elements
	{Name: "Iron", Symbol: "Fe", Number: 26, Group: "Transition Metal"},
	{Name: "Copper", Symbol: "Cu", Number: 29, Group: "Transition Metal"},
	{Name: "Zinc", Symbol: "Zn", Number: 30, Group: "Transition Metal"},
	{Name: "Silver", Symbol: "Ag", Number: 47, Group: "Transition Metal"},
	{Name: "Gold", Symbol: "Au", Number: 79, Group: "Transition Metal"},
	{Name: "Mercury", Symbol: "Hg", Number: 80, Group: "Transition Metal"},
	{Name: "Lead", Symbol: "Pb", Number: 82, Group: "Post-transition Metal"},
	{Name: "Uranium", Symbol: "U", Number: 92, Group: "Actinide"},
}

// Planet represents a planet in our solar system
type Planet struct {
	Name     string
	Order    int
	Type     string // "Terrestrial" or "Gas Giant" or "Ice Giant"
	Moons    int
	RingsYes bool
}

// SolarSystemPlanets contains planets in our solar system
var SolarSystemPlanets = []Planet{
	{Name: "Mercury", Order: 1, Type: "Terrestrial", Moons: 0, RingsYes: false},
	{Name: "Venus", Order: 2, Type: "Terrestrial", Moons: 0, RingsYes: false},
	{Name: "Earth", Order: 3, Type: "Terrestrial", Moons: 1, RingsYes: false},
	{Name: "Mars", Order: 4, Type: "Terrestrial", Moons: 2, RingsYes: false},
	{Name: "Jupiter", Order: 5, Type: "Gas Giant", Moons: 79, RingsYes: true},
	{Name: "Saturn", Order: 6, Type: "Gas Giant", Moons: 82, RingsYes: true},
	{Name: "Uranus", Order: 7, Type: "Ice Giant", Moons: 27, RingsYes: true},
	{Name: "Neptune", Order: 8, Type: "Ice Giant", Moons: 14, RingsYes: true},
}

// BiologyTerm represents a biology concept
type BiologyTerm struct {
	Term       string
	Definition string
	Category   string // e.g., "Cell Biology", "Genetics", "Ecology"
}

// BiologyTerms contains common biology terms and concepts
var BiologyTerms = []BiologyTerm{
	{
		Term:       "Photosynthesis",
		Definition: "The process by which plants convert light energy into chemical energy",
		Category:   "Plant Biology",
	},
	{
		Term:       "Mitosis",
		Definition: "The process of cell division that results in two identical daughter cells",
		Category:   "Cell Biology",
	},
	{
		Term:       "DNA",
		Definition: "The molecule that carries genetic information in living organisms",
		Category:   "Genetics",
	},
	{
		Term:       "Evolution",
		Definition: "The process by which species change over time through natural selection",
		Category:   "Evolutionary Biology",
	},
	{
		Term:       "Ecosystem",
		Definition: "A community of living organisms interacting with their environment",
		Category:   "Ecology",
	},
	{
		Term:       "Metabolism",
		Definition: "The chemical processes that occur within a living organism to maintain life",
		Category:   "Biochemistry",
	},
	{
		Term:       "Homeostasis",
		Definition: "The ability of an organism to maintain stable internal conditions",
		Category:   "Physiology",
	},
	{
		Term:       "Symbiosis",
		Definition: "A close relationship between two different species",
		Category:   "Ecology",
	},
}

// GeneralKnowledgeConcept represents a general knowledge topic
type GeneralKnowledgeConcept struct {
	Term     string
	Correct  string
	Wrong    []string
	Category string
}

// GeneralKnowledgeConcepts contains various general knowledge topics
var GeneralKnowledgeConcepts = []GeneralKnowledgeConcept{
	// Technology
	{
		Term:     "Algorithm",
		Correct:  "A step-by-step procedure for solving a problem",
		Wrong:    []string{"A type of mathematical equation", "A programming language", "A computer hardware component"},
		Category: "Technology",
	},
	{
		Term:     "Artificial Intelligence",
		Correct:  "Computer systems able to perform tasks that normally require human intelligence",
		Wrong:    []string{"A type of robot", "A programming language", "A computer virus"},
		Category: "Technology",
	},
	// Politics
	{
		Term:     "Democracy",
		Correct:  "A system of government where power is vested in the people",
		Wrong:    []string{"A system where one person holds absolute power", "A system where religious leaders govern", "A system where the military controls the government"},
		Category: "Politics",
	},
	{
		Term:     "Constitution",
		Correct:  "The fundamental principles and laws of a nation",
		Wrong:    []string{"A type of government building", "A political party", "A voting system"},
		Category: "Politics",
	},
	// Geography
	{
		Term:     "Continent",
		Correct:  "A large continuous mass of land",
		Wrong:    []string{"A type of ocean", "A mountain range", "A political boundary"},
		Category: "Geography",
	},
	{
		Term:     "Equator",
		Correct:  "An imaginary line around the Earth equally distant from both poles",
		Wrong:    []string{"The hottest place on Earth", "A type of climate", "The center of the Earth"},
		Category: "Geography",
	},
	// History
	{
		Term:     "Renaissance",
		Correct:  "A period of cultural rebirth in Europe from the 14th to 17th century",
		Wrong:    []string{"A type of art style", "A political movement", "A religious reformation"},
		Category: "History",
	},
	{
		Term:     "Industrial Revolution",
		Correct:  "The transition to new manufacturing processes in the 18th and 19th centuries",
		Wrong:    []string{"A political uprising", "A scientific discovery", "A type of machine"},
		Category: "History",
	},
	// Economics
	{
		Term:     "Inflation",
		Correct:  "A general increase in prices and fall in the purchasing value of money",
		Wrong:    []string{"An increase in wages", "A type of investment", "A banking system"},
		Category: "Economics",
	},
	{
		Term:     "Supply and Demand",
		Correct:  "The relationship between product availability and consumer desire",
		Wrong:    []string{"A type of business", "A government policy", "A banking regulation"},
		Category: "Economics",
	},
}

// GenerateSafeFakeSymbols generates plausible but incorrect chemical symbols
func GenerateSafeFakeSymbols(realSymbol string) []string {
	fakeSymbols := []string{}

	// Strategy 1: Change case safely
	if len(realSymbol) == 2 && unicode.IsUpper(rune(realSymbol[1])) {
		// Convert second letter to lowercase
		fakeSymbols = append(fakeSymbols, string(realSymbol[0])+strings.ToLower(string(realSymbol[1])))
	}

	// Strategy 2: Use nearby letters with bounds checking
	if len(realSymbol) >= 1 {
		firstChar := realSymbol[0]
		// Check bounds before manipulation
		if firstChar > 'A' && firstChar <= 'Z' {
			fakeSymbols = append(fakeSymbols, string(firstChar-1))
		}
		if firstChar >= 'A' && firstChar < 'Z' {
			fakeSymbols = append(fakeSymbols, string(firstChar+1))
		}
	}

	// Strategy 3: Use common fake symbols that don't exist
	commonFakes := []string{"Xy", "Zz", "Qq", "Jj", "Xx", "Yy", "Vv", "Ww"}
	for _, fake := range commonFakes {
		if fake != realSymbol && !isRealSymbol(fake) {
			fakeSymbols = append(fakeSymbols, fake)
		}
	}

	// Strategy 4: Swap letters if two-letter symbol
	if len(realSymbol) == 2 {
		swapped := string(realSymbol[1]) + string(realSymbol[0])
		if !isRealSymbol(swapped) {
			fakeSymbols = append(fakeSymbols, swapped)
		}
	}

	// Ensure we have at least 3 unique fake symbols
	seen := make(map[string]bool)
	result := []string{}
	for _, fake := range fakeSymbols {
		if !seen[fake] && len(result) < 3 {
			seen[fake] = true
			result = append(result, fake)
		}
	}

	// Fill remaining slots with safe generated values
	for len(result) < 3 {
		result = append(result, fmt.Sprintf("X%d", len(result)+1))
	}

	return result
}

// isRealSymbol checks if a symbol is a real chemical element symbol
func isRealSymbol(symbol string) bool {
	realSymbols := make(map[string]bool)
	for _, elem := range ScienceElements {
		realSymbols[elem.Symbol] = true
	}
	return realSymbols[symbol]
}

// GetRandomElement returns a random element based on index and optional randomizer
func GetRandomElement(rng *rand.Rand, index int) Element {
	if rng != nil {
		return ScienceElements[rng.Intn(len(ScienceElements))]
	}
	return ScienceElements[index%len(ScienceElements)]
}

// GetRandomPlanet returns a random planet
func GetRandomPlanet(rng *rand.Rand, index int) Planet {
	if rng != nil {
		return SolarSystemPlanets[rng.Intn(len(SolarSystemPlanets))]
	}
	return SolarSystemPlanets[index%len(SolarSystemPlanets)]
}

// GetRandomBiologyTerm returns a random biology term
func GetRandomBiologyTerm(rng *rand.Rand, index int) BiologyTerm {
	if rng != nil {
		return BiologyTerms[rng.Intn(len(BiologyTerms))]
	}
	return BiologyTerms[index%len(BiologyTerms)]
}

// GetRandomGeneralConcept returns a random general knowledge concept
func GetRandomGeneralConcept(rng *rand.Rand, index int) GeneralKnowledgeConcept {
	if rng != nil {
		return GeneralKnowledgeConcepts[rng.Intn(len(GeneralKnowledgeConcepts))]
	}
	return GeneralKnowledgeConcepts[index%len(GeneralKnowledgeConcepts)]
}
