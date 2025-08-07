package units

import (
	"bytes"
	"math"
	"strings"
	"testing"
	"text/template"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGetTemplateFuncMap(t *testing.T) {
	funcMap := GetTemplateFuncMap()

	// Verify FuncMap is not nil
	require.NotNil(t, funcMap, "GetTemplateFuncMap should return non-nil FuncMap")

	// Expected function names in the template function map
	expectedFunctions := []string{
		"add", "sub", "mul", "div", "mod",
		"contains", "truncate", "hasPrefix", "hasSuffix",
		"lower", "upper", "trim", "replace", "join", "split",
	}

	// Verify all expected functions are present
	assert.Len(t, funcMap, len(expectedFunctions), "FuncMap should contain exactly %d functions", len(expectedFunctions))

	for _, funcName := range expectedFunctions {
		assert.Contains(t, funcMap, funcName, "FuncMap should contain function '%s'", funcName)
		assert.NotNil(t, funcMap[funcName], "Function '%s' should not be nil", funcName)
	}
}

// TestArithmeticFunctions tests all arithmetic template functions
func TestArithmeticFunctions(t *testing.T) {
	funcMap := GetTemplateFuncMap()

	t.Run("add", func(t *testing.T) {
		tests := []struct {
			name     string
			a, b     int
			expected int
		}{
			{"positive numbers", 5, 3, 8},
			{"negative numbers", -5, -3, -8},
			{"mixed signs", -5, 3, -2},
			{"zero values", 0, 0, 0},
			{"add zero", 10, 0, 10},
			{"large numbers", math.MaxInt32, 1, math.MaxInt32 + 1},
			{"overflow boundary", math.MaxInt - 1, 1, math.MaxInt},
		}

		addFunc := funcMap["add"].(func(int, int) int)
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				result := addFunc(tt.a, tt.b)
				assert.Equal(t, tt.expected, result)
			})
		}
	})

	t.Run("sub", func(t *testing.T) {
		tests := []struct {
			name     string
			a, b     int
			expected int
		}{
			{"positive numbers", 10, 3, 7},
			{"negative numbers", -10, -3, -7},
			{"mixed signs", -5, 3, -8},
			{"zero values", 0, 0, 0},
			{"subtract zero", 10, 0, 10},
			{"subtract from zero", 0, 5, -5},
			{"large numbers", math.MaxInt32, -1, math.MaxInt32 + 1},
		}

		subFunc := funcMap["sub"].(func(int, int) int)
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				result := subFunc(tt.a, tt.b)
				assert.Equal(t, tt.expected, result)
			})
		}
	})

	t.Run("mul", func(t *testing.T) {
		tests := []struct {
			name     string
			a, b     int
			expected int
		}{
			{"positive numbers", 5, 3, 15},
			{"negative numbers", -5, -3, 15},
			{"mixed signs", -5, 3, -15},
			{"zero multiplication", 0, 100, 0},
			{"multiply by one", 42, 1, 42},
			{"multiply by negative one", 42, -1, -42},
		}

		mulFunc := funcMap["mul"].(func(int, int) int)
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				result := mulFunc(tt.a, tt.b)
				assert.Equal(t, tt.expected, result)
			})
		}
	})

	t.Run("div", func(t *testing.T) {
		tests := []struct {
			name     string
			a, b     int
			expected int
		}{
			{"positive division", 15, 3, 5},
			{"negative dividend", -15, 3, -5},
			{"negative divisor", 15, -3, -5},
			{"both negative", -15, -3, 5},
			{"division by one", 42, 1, 42},
			{"division by negative one", 42, -1, -42},
			{"integer division truncation", 7, 3, 2},
			{"division by zero", 10, 0, 0}, // Safety: returns 0 instead of panicking
			{"zero divided by number", 0, 5, 0},
		}

		divFunc := funcMap["div"].(func(int, int) int)
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				result := divFunc(tt.a, tt.b)
				assert.Equal(t, tt.expected, result)
			})
		}
	})

	t.Run("mod", func(t *testing.T) {
		tests := []struct {
			name     string
			a, b     int
			expected int
		}{
			{"positive modulo", 10, 3, 1},
			{"no remainder", 10, 5, 0},
			{"negative dividend", -10, 3, -1},
			{"negative divisor", 10, -3, 1},
			{"both negative", -10, -3, -1},
			{"modulo by one", 42, 1, 0},
			{"smaller dividend", 3, 10, 3},
			{"modulo by zero", 10, 0, 0}, // Safety: returns 0 instead of panicking
			{"zero modulo", 0, 5, 0},
		}

		modFunc := funcMap["mod"].(func(int, int) int)
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				result := modFunc(tt.a, tt.b)
				assert.Equal(t, tt.expected, result)
			})
		}
	})
}

// TestStringFunctions tests all string manipulation template functions
func TestStringFunctions(t *testing.T) {
	funcMap := GetTemplateFuncMap()

	t.Run("contains", func(t *testing.T) {
		tests := []struct {
			name      string
			s, substr string
			expected  bool
		}{
			{"contains substring", "hello world", "world", true},
			{"does not contain", "hello world", "xyz", false},
			{"empty substring", "hello", "", true},
			{"empty string", "", "hello", false},
			{"both empty", "", "", true},
			{"case sensitive", "Hello", "hello", false},
			{"unicode characters", "héllo wørld", "ørld", true},
			{"zero-width characters", "hello\u200bworld", "\u200b", true},
		}

		containsFunc := funcMap["contains"].(func(string, string) bool)
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				result := containsFunc(tt.s, tt.substr)
				assert.Equal(t, tt.expected, result)
			})
		}
	})

	t.Run("truncate", func(t *testing.T) {
		tests := []struct {
			name     string
			s        string
			length   int
			expected string
		}{
			{"normal truncation", "hello world", 5, "he..."},
			{"no truncation needed", "hello", 10, "hello"},
			{"exact length", "hello", 5, "hello"},
			{"zero length", "hello", 0, ""},
			{"negative length", "hello", -1, ""},
			{"length one", "hello", 1, "h"},
			{"length two", "hello", 2, "he"},
			{"length three", "hello", 3, "hel"},
			{"empty string", "", 5, ""},
			{"unicode characters", "héllo wørld", 8, "héll..."},
			{"very long string", strings.Repeat("a", 1000), 10, "aaaaaaa..."},
		}

		truncateFunc := funcMap["truncate"].(func(string, int) string)
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				result := truncateFunc(tt.s, tt.length)
				assert.Equal(t, tt.expected, result)
			})
		}
	})

	t.Run("hasPrefix", func(t *testing.T) {
		tests := []struct {
			name      string
			s, prefix string
			expected  bool
		}{
			{"has prefix", "hello world", "hello", true},
			{"does not have prefix", "hello world", "world", false},
			{"empty prefix", "hello", "", true},
			{"empty string", "", "hello", false},
			{"both empty", "", "", true},
			{"case sensitive", "Hello", "hello", false},
			{"unicode prefix", "héllo world", "héllo", true},
			{"prefix longer than string", "hi", "hello", false},
		}

		hasPrefixFunc := funcMap["hasPrefix"].(func(string, string) bool)
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				result := hasPrefixFunc(tt.s, tt.prefix)
				assert.Equal(t, tt.expected, result)
			})
		}
	})

	t.Run("hasSuffix", func(t *testing.T) {
		tests := []struct {
			name      string
			s, suffix string
			expected  bool
		}{
			{"has suffix", "hello world", "world", true},
			{"does not have suffix", "hello world", "hello", false},
			{"empty suffix", "hello", "", true},
			{"empty string", "", "hello", false},
			{"both empty", "", "", true},
			{"case sensitive", "Hello", "HELLO", false},
			{"unicode suffix", "hello wørld", "wørld", true},
			{"suffix longer than string", "hi", "hello", false},
		}

		hasSuffixFunc := funcMap["hasSuffix"].(func(string, string) bool)
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				result := hasSuffixFunc(tt.s, tt.suffix)
				assert.Equal(t, tt.expected, result)
			})
		}
	})

	t.Run("lower", func(t *testing.T) {
		tests := []struct {
			name     string
			s        string
			expected string
		}{
			{"uppercase letters", "HELLO", "hello"},
			{"mixed case", "Hello World", "hello world"},
			{"already lowercase", "hello", "hello"},
			{"empty string", "", ""},
			{"numbers and symbols", "Hello123!", "hello123!"},
			{"unicode characters", "HÉLLO WØRLD", "héllo wørld"},
			{"turkish i problem", "İstanbul", strings.ToLower("İstanbul")}, // Note: This tests Unicode handling
		}

		lowerFunc := funcMap["lower"].(func(string) string)
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				result := lowerFunc(tt.s)
				assert.Equal(t, tt.expected, result)
			})
		}
	})

	t.Run("upper", func(t *testing.T) {
		tests := []struct {
			name     string
			s        string
			expected string
		}{
			{"lowercase letters", "hello", "HELLO"},
			{"mixed case", "Hello World", "HELLO WORLD"},
			{"already uppercase", "HELLO", "HELLO"},
			{"empty string", "", ""},
			{"numbers and symbols", "hello123!", "HELLO123!"},
			{"unicode characters", "héllo wørld", "HÉLLO WØRLD"},
		}

		upperFunc := funcMap["upper"].(func(string) string)
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				result := upperFunc(tt.s)
				assert.Equal(t, tt.expected, result)
			})
		}
	})

	t.Run("trim", func(t *testing.T) {
		tests := []struct {
			name     string
			s        string
			expected string
		}{
			{"leading and trailing spaces", "  hello  ", "hello"},
			{"only leading spaces", "  hello", "hello"},
			{"only trailing spaces", "hello  ", "hello"},
			{"no spaces", "hello", "hello"},
			{"empty string", "", ""},
			{"only spaces", "   ", ""},
			{"tabs and newlines", "\t\nhello\n\t", "hello"},
			{"unicode whitespace", "\u2000hello\u2000", "hello"},
		}

		trimFunc := funcMap["trim"].(func(string) string)
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				result := trimFunc(tt.s)
				assert.Equal(t, tt.expected, result)
			})
		}
	})

	t.Run("replace", func(t *testing.T) {
		tests := []struct {
			name        string
			s, old, new string
			expected    string
		}{
			{"simple replacement", "hello world", "world", "universe", "hello universe"},
			{"multiple occurrences", "hello hello", "hello", "hi", "hi hi"},
			{"no match", "hello world", "xyz", "abc", "hello world"},
			{"empty old string", "hello", "", "x", "xhxexlxlxox"}, // ReplaceAll with empty old inserts between each char
			{"empty new string", "hello world", "world", "", "hello "},
			{"empty input", "", "old", "new", ""},
			{"replace with same", "hello", "hello", "hello", "hello"},
			{"unicode replacement", "héllo wørld", "ørld", "earth", "héllo wearth"},
		}

		replaceFunc := funcMap["replace"].(func(string, string, string) string)
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				result := replaceFunc(tt.s, tt.old, tt.new)
				assert.Equal(t, tt.expected, result)
			})
		}
	})

	t.Run("join", func(t *testing.T) {
		tests := []struct {
			name     string
			elems    []string
			sep      string
			expected string
		}{
			{"normal join", []string{"a", "b", "c"}, ",", "a,b,c"},
			{"empty separator", []string{"a", "b", "c"}, "", "abc"},
			{"single element", []string{"hello"}, ",", "hello"},
			{"empty slice", []string{}, ",", ""},
			{"nil slice", nil, ",", ""},
			{"empty elements", []string{"", "b", ""}, ",", ",b,"},
			{"space separator", []string{"hello", "world"}, " ", "hello world"},
			{"unicode separator", []string{"a", "b", "c"}, "⭐", "a⭐b⭐c"},
		}

		joinFunc := funcMap["join"].(func([]string, string) string)
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				result := joinFunc(tt.elems, tt.sep)
				assert.Equal(t, tt.expected, result)
			})
		}
	})

	t.Run("split", func(t *testing.T) {
		tests := []struct {
			name     string
			s, sep   string
			expected []string
		}{
			{"normal split", "a,b,c", ",", []string{"a", "b", "c"}},
			{"single element", "hello", ",", []string{"hello"}},
			{"empty string", "", ",", []string{""}},
			{"empty separator", "abc", "", []string{"a", "b", "c"}},
			{"separator not found", "hello", ",", []string{"hello"}},
			{"consecutive separators", "a,,b", ",", []string{"a", "", "b"}},
			{"leading separator", ",a,b", ",", []string{"", "a", "b"}},
			{"trailing separator", "a,b,", ",", []string{"a", "b", ""}},
			{"unicode separator", "a⭐b⭐c", "⭐", []string{"a", "b", "c"}},
		}

		splitFunc := funcMap["split"].(func(string, string) []string)
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				result := splitFunc(tt.s, tt.sep)
				assert.Equal(t, tt.expected, result)
			})
		}
	})
}

// TestTemplateIntegration tests that functions work correctly within template execution
func TestTemplateIntegration(t *testing.T) {
	funcMap := GetTemplateFuncMap()

	tests := []struct {
		name     string
		template string
		data     interface{}
		expected string
	}{
		{
			name:     "arithmetic in template",
			template: `{{add 5 3}} {{sub 10 4}} {{mul 3 2}} {{div 15 3}} {{mod 10 3}}`,
			data:     nil,
			expected: "8 6 6 5 1",
		},
		{
			name:     "string functions in template",
			template: `{{if contains .text "world"}}found{{end}} {{truncate .text 8}}`,
			data:     map[string]string{"text": "hello world"},
			expected: "found hello...",
		},
		{
			name:     "prefix and suffix checks",
			template: `{{if hasPrefix .name "test"}}prefix{{end}} {{if hasSuffix .file ".go"}}go file{{end}}`,
			data:     map[string]string{"name": "test123", "file": "main.go"},
			expected: "prefix go file",
		},
		{
			name:     "case conversion",
			template: `{{lower .upper}} {{upper .lower}}`,
			data:     map[string]string{"upper": "HELLO", "lower": "world"},
			expected: "hello WORLD",
		},
		{
			name:     "string manipulation pipeline",
			template: `{{replace (lower (trim .text)) "old" "new"}}`,
			data:     map[string]string{"text": "  OLD TEXT  "},
			expected: "new text",
		},
		{
			name:     "array operations",
			template: `{{join .items ", "}} -> {{split (join .items ",") ","}}`,
			data:     map[string][]string{"items": {"a", "b", "c"}},
			expected: "a, b, c -> [a b c]",
		},
		{
			name:     "complex template with loops",
			template: `{{range $i, $v := .items}}{{if $i}}, {{end}}{{add $i 1}}: {{upper $v}}{{end}}`,
			data:     map[string][]string{"items": {"apple", "banana", "cherry"}},
			expected: "1: APPLE, 2: BANANA, 3: CHERRY",
		},
		{
			name:     "nested function calls",
			template: `{{truncate (upper (trim .text)) 5}}`,
			data:     map[string]string{"text": "  hello world  "},
			expected: "HE...",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpl, err := template.New("test").Funcs(funcMap).Parse(tt.template)
			require.NoError(t, err, "Template should parse successfully")

			var buf bytes.Buffer
			err = tmpl.Execute(&buf, tt.data)
			require.NoError(t, err, "Template should execute successfully")

			result := buf.String()
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestTemplateFunctionErrorHandling tests error handling in template context
func TestTemplateFunctionErrorHandling(t *testing.T) {
	funcMap := GetTemplateFuncMap()

	// Test that division by zero doesn't panic in template
	t.Run("division by zero in template", func(t *testing.T) {
		tmpl, err := template.New("test").Funcs(funcMap).Parse(`{{div 10 0}} {{mod 10 0}}`)
		require.NoError(t, err)

		var buf bytes.Buffer
		err = tmpl.Execute(&buf, nil)
		require.NoError(t, err, "Template execution should not fail even with division by zero")

		result := buf.String()
		assert.Equal(t, "0 0", result, "Division by zero should return 0")
	})

	// Test that all functions are callable from templates without panics
	t.Run("all functions callable", func(t *testing.T) {
		templateStr := `
			{{add 1 2}}
			{{sub 5 3}}
			{{mul 2 3}}
			{{div 10 2}}
			{{mod 10 3}}
			{{contains "hello" "ell"}}
			{{truncate "hello world" 5}}
			{{hasPrefix "hello" "he"}}
			{{hasSuffix "hello" "lo"}}
			{{lower "HELLO"}}
			{{upper "hello"}}
			{{trim "  hello  "}}
			{{replace "hello" "l" "L"}}
			{{join (split "a,b,c" ",") "|"}}
		`

		tmpl, err := template.New("test").Funcs(funcMap).Parse(templateStr)
		require.NoError(t, err)

		var buf bytes.Buffer
		err = tmpl.Execute(&buf, nil)
		require.NoError(t, err, "All template functions should be executable without errors")

		result := strings.TrimSpace(buf.String())
		assert.NotEmpty(t, result, "Template should produce output")
	})
}
