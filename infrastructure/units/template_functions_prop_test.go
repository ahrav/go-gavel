package units

import (
	"math"
	"testing"
	"testing/quick"

	"github.com/stretchr/testify/assert"
)

// TestArithmeticProperties tests mathematical properties of arithmetic functions
func TestArithmeticProperties(t *testing.T) {
	funcMap := GetTemplateFuncMap()

	addFunc := funcMap["add"].(func(int, int) int)
	subFunc := funcMap["sub"].(func(int, int) int)
	mulFunc := funcMap["mul"].(func(int, int) int)
	divFunc := funcMap["div"].(func(int, int) int)
	modFunc := funcMap["mod"].(func(int, int) int)

	t.Run("add properties", func(t *testing.T) {
		// Property: Commutativity - a + b = b + a
		err := quick.Check(func(a, b int) bool {
			return addFunc(a, b) == addFunc(b, a)
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Addition should be commutative")

		// Property: Associativity - (a + b) + c = a + (b + c)
		err = quick.Check(func(a, b, c int) bool {
			// Avoid overflow by using smaller numbers
			if a > 1000 || a < -1000 || b > 1000 || b < -1000 || c > 1000 || c < -1000 {
				return true // Skip this test case
			}
			left := addFunc(addFunc(a, b), c)
			right := addFunc(a, addFunc(b, c))
			return left == right
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Addition should be associative")

		// Property: Identity element - a + 0 = a
		err = quick.Check(func(a int) bool {
			return addFunc(a, 0) == a && addFunc(0, a) == a
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Zero should be identity element for addition")

		// Property: Inverse operation - (a + b) - b = a
		err = quick.Check(func(a, b int) bool {
			// Avoid overflow
			if (b > 0 && a > math.MaxInt-b) || (b < 0 && a < math.MinInt-b) {
				return true // Skip overflow cases
			}
			sum := addFunc(a, b)
			return subFunc(sum, b) == a
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Subtraction should be inverse of addition")
	})

	t.Run("sub properties", func(t *testing.T) {
		// Property: Self subtraction - a - a = 0
		err := quick.Check(func(a int) bool {
			return subFunc(a, a) == 0
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Self subtraction should equal zero")

		// Property: Zero subtraction - a - 0 = a
		err = quick.Check(func(a int) bool {
			return subFunc(a, 0) == a
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Subtracting zero should return original value")

		// Property: Subtracting from zero - 0 - a = -a
		err = quick.Check(func(a int) bool {
			// Avoid MinInt overflow when negating
			if a == math.MinInt {
				return true
			}
			return subFunc(0, a) == -a
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Subtracting from zero should negate")

		// Property: Non-commutative - generally a - b ≠ b - a (except when a = b)
		err = quick.Check(func(a, b int) bool {
			if a == b {
				return true // Skip when equal
			}
			return subFunc(a, b) != subFunc(b, a)
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Subtraction should not be commutative (except when equal)")
	})

	t.Run("mul properties", func(t *testing.T) {
		// Property: Commutativity - a * b = b * a
		err := quick.Check(func(a, b int) bool {
			// Use smaller numbers to avoid overflow
			if a > 10000 || a < -10000 || b > 10000 || b < -10000 {
				return true
			}
			return mulFunc(a, b) == mulFunc(b, a)
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Multiplication should be commutative")

		// Property: Identity element - a * 1 = a
		err = quick.Check(func(a int) bool {
			return mulFunc(a, 1) == a && mulFunc(1, a) == a
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "One should be identity element for multiplication")

		// Property: Zero multiplication - a * 0 = 0
		err = quick.Check(func(a int) bool {
			return mulFunc(a, 0) == 0 && mulFunc(0, a) == 0
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Multiplication by zero should equal zero")

		// Property: Negation - a * (-1) = -a
		err = quick.Check(func(a int) bool {
			// Avoid MinInt overflow
			if a == math.MinInt {
				return true
			}
			return mulFunc(a, -1) == -a && mulFunc(-1, a) == -a
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Multiplication by -1 should negate")

		// Property: Associativity - (a * b) * c = a * (b * c)
		err = quick.Check(func(a, b, c int8) bool { // Use int8 to avoid overflow
			aInt, bInt, cInt := int(a), int(b), int(c)
			left := mulFunc(mulFunc(aInt, bInt), cInt)
			right := mulFunc(aInt, mulFunc(bInt, cInt))
			return left == right
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Multiplication should be associative")
	})

	t.Run("div properties", func(t *testing.T) {
		// Property: Self division - a / a = 1 (when a != 0)
		err := quick.Check(func(a int) bool {
			if a == 0 {
				return divFunc(a, a) == 0 // Our implementation returns 0 for 0/0
			}
			return divFunc(a, a) == 1
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Self division should equal one (except for zero)")

		// Property: Division by one - a / 1 = a
		err = quick.Check(func(a int) bool {
			return divFunc(a, 1) == a
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Division by one should return original value")

		// Property: Zero dividend - 0 / a = 0 (when a != 0)
		err = quick.Check(func(a int) bool {
			if a == 0 {
				return divFunc(0, a) == 0 // Our implementation returns 0 for 0/0
			}
			return divFunc(0, a) == 0
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Zero divided by anything should be zero")

		// Property: Division by zero safety - a / 0 = 0 (our safe implementation)
		err = quick.Check(func(a int) bool {
			return divFunc(a, 0) == 0
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Division by zero should return zero (safety feature)")

		// Property: Division by negative one - a / (-1) = -a
		err = quick.Check(func(a int) bool {
			// Avoid MinInt overflow
			if a == math.MinInt {
				return true
			}
			return divFunc(a, -1) == -a
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Division by -1 should negate")

		// Property: Relationship with multiplication - (a * b) / b = a (when b != 0 and no overflow)
		err = quick.Check(func(a int8, b int8) bool {
			if b == 0 {
				return true // Skip division by zero
			}
			aInt, bInt := int(a), int(b)
			product := mulFunc(aInt, bInt)
			return divFunc(product, bInt) == aInt
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Division should be inverse of multiplication")
	})

	t.Run("mod properties", func(t *testing.T) {
		// Property: Modulo by one - a % 1 = 0
		err := quick.Check(func(a int) bool {
			return modFunc(a, 1) == 0
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Modulo by one should always be zero")

		// Property: Self modulo - a % a = 0 (when a != 0)
		err = quick.Check(func(a int) bool {
			if a == 0 {
				return modFunc(a, a) == 0 // Our implementation returns 0 for 0%0
			}
			return modFunc(a, a) == 0
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Self modulo should be zero")

		// Property: Modulo by zero safety - a % 0 = 0 (our safe implementation)
		err = quick.Check(func(a int) bool {
			return modFunc(a, 0) == 0
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Modulo by zero should return zero (safety feature)")

		// Property: Zero modulo - 0 % a = 0 (when a != 0)
		err = quick.Check(func(a int) bool {
			if a == 0 {
				return modFunc(0, a) == 0 // Our implementation returns 0 for 0%0
			}
			return modFunc(0, a) == 0
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Zero modulo anything should be zero")

		// Property: Range constraint - |a % b| < |b| (when b != 0)
		err = quick.Check(func(a, b int) bool {
			if b == 0 {
				return modFunc(a, b) == 0 // Our implementation returns 0
			}
			result := modFunc(a, b)
			absResult := result
			if absResult < 0 {
				absResult = -absResult
			}
			absB := b
			if absB < 0 {
				absB = -absB
			}
			return absResult < absB
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Modulo result should be less than divisor in absolute value")

		// Property: Division relationship - a = (a / b) * b + (a % b) (when b != 0)
		err = quick.Check(func(a, b int) bool {
			if b == 0 {
				return true // Skip division by zero cases
			}
			quotient := divFunc(a, b)
			remainder := modFunc(a, b)
			return a == mulFunc(quotient, b)+remainder
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Division and modulo should satisfy the division algorithm")
	})
}

// TestArithmeticBoundaryProperties tests properties at integer boundaries
func TestArithmeticBoundaryProperties(t *testing.T) {
	funcMap := GetTemplateFuncMap()

	addFunc := funcMap["add"].(func(int, int) int)
	subFunc := funcMap["sub"].(func(int, int) int)
	mulFunc := funcMap["mul"].(func(int, int) int)
	divFunc := funcMap["div"].(func(int, int) int)
	modFunc := funcMap["mod"].(func(int, int) int)

	t.Run("max int boundary", func(t *testing.T) {
		// Test behavior at maximum integer values
		maxInt := math.MaxInt

		// Addition at boundary (will overflow, but should not panic)
		result := addFunc(maxInt, 1)
		assert.NotPanics(t, func() { addFunc(maxInt, 1) }, "Addition should not panic on overflow")
		t.Logf("MaxInt + 1 = %d (overflow expected)", result)

		// Division at boundary
		assert.Equal(t, maxInt, divFunc(maxInt, 1))
		assert.Equal(t, 1, divFunc(maxInt, maxInt))
		assert.Equal(t, 0, divFunc(maxInt, 0)) // Safe division by zero

		// Modulo at boundary
		assert.Equal(t, 0, modFunc(maxInt, maxInt))
		assert.Equal(t, 0, modFunc(maxInt, 0)) // Safe modulo by zero
	})

	t.Run("min int boundary", func(t *testing.T) {
		// Test behavior at minimum integer values
		minInt := math.MinInt

		// Subtraction at boundary (will overflow, but should not panic)
		result := subFunc(minInt, 1)
		assert.NotPanics(t, func() { subFunc(minInt, 1) }, "Subtraction should not panic on overflow")
		t.Logf("MinInt - 1 = %d (overflow expected)", result)

		// Division at boundary
		assert.Equal(t, minInt, divFunc(minInt, 1))
		assert.Equal(t, 1, divFunc(minInt, minInt))
		assert.Equal(t, 0, divFunc(minInt, 0)) // Safe division by zero

		// Multiplication at boundary
		assert.Equal(t, minInt, mulFunc(minInt, 1))
		assert.Equal(t, 0, mulFunc(minInt, 0))

		// Note: mulFunc(minInt, -1) would overflow to a positive number larger than MaxInt
		// This is expected Go behavior for integer overflow
	})

	t.Run("zero boundary", func(t *testing.T) {
		// All operations with zero should be well-defined
		assert.Equal(t, 42, addFunc(42, 0))
		assert.Equal(t, 42, addFunc(0, 42))
		assert.Equal(t, 42, subFunc(42, 0))
		assert.Equal(t, -42, subFunc(0, 42))
		assert.Equal(t, 0, mulFunc(42, 0))
		assert.Equal(t, 0, mulFunc(0, 42))
		assert.Equal(t, 0, divFunc(0, 42))
		assert.Equal(t, 0, divFunc(42, 0)) // Safe implementation
		assert.Equal(t, 0, modFunc(0, 42))
		assert.Equal(t, 0, modFunc(42, 0)) // Safe implementation
	})
}

// TestStringFunctionProperties tests mathematical properties of string functions
func TestStringFunctionProperties(t *testing.T) {
	funcMap := GetTemplateFuncMap()

	lowerFunc := funcMap["lower"].(func(string) string)
	upperFunc := funcMap["upper"].(func(string) string)
	trimFunc := funcMap["trim"].(func(string) string)

	t.Run("case conversion properties", func(t *testing.T) {
		// Property: Idempotence - f(f(x)) = f(x)
		err := quick.Check(func(s string) bool {
			lower1 := lowerFunc(s)
			lower2 := lowerFunc(lower1)
			return lower1 == lower2
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Lower case should be idempotent")

		err = quick.Check(func(s string) bool {
			upper1 := upperFunc(s)
			upper2 := upperFunc(upper1)
			return upper1 == upper2
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Upper case should be idempotent")

		// Property: Length preservation (for most strings, but not all due to Unicode)
		err = quick.Check(func(s string) bool {
			// This property doesn't hold for all Unicode strings (e.g., German ß -> SS)
			// but we can test that it's generally close
			lower := lowerFunc(s)
			upper := upperFunc(s)
			// Allow some tolerance for Unicode case mappings
			return len(lower) >= len(s)-10 && len(upper) >= len(s)-10
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Case conversion should approximately preserve length")
	})

	t.Run("trim properties", func(t *testing.T) {
		// Property: Idempotence - trim(trim(x)) = trim(x)
		err := quick.Check(func(s string) bool {
			trim1 := trimFunc(s)
			trim2 := trimFunc(trim1)
			return trim1 == trim2
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Trim should be idempotent")

		// Property: Length reduction - len(trim(s)) <= len(s)
		err = quick.Check(func(s string) bool {
			trimmed := trimFunc(s)
			return len(trimmed) <= len(s)
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Trim should never increase length")

		// Property: Trimmed strings have no leading/trailing basic whitespace
		err = quick.Check(func(s string) bool {
			trimmed := trimFunc(s)
			if len(trimmed) == 0 {
				return true
			}
			// Check for basic ASCII whitespace (space, tab, newline)
			first, last := trimmed[0], trimmed[len(trimmed)-1]
			return first != ' ' && first != '\t' && first != '\n' &&
				last != ' ' && last != '\t' && last != '\n'
		}, &quick.Config{MaxCount: 1000})
		assert.NoError(t, err, "Trimmed strings should not have leading/trailing basic whitespace")
	})
}
