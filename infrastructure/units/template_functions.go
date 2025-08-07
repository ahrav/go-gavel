// Package units provides evaluation units and template functions for the go-gavel framework.
//
// The template functions in this package extend Go's standard template capabilities
// with operations essential for evaluation prompt generation and result processing.
// These functions are designed to be stateless, thread-safe, and deterministic,
// making them suitable for concurrent template execution in evaluation workflows.
//
// Template functions are organized into three categories:
//   - Arithmetic operations for index manipulation and scoring calculations
//   - String operations for content analysis and text processing
//   - String transformations for normalization and formatting
//
// All functions handle edge cases gracefully, returning safe defaults rather than
// panicking, which is critical for template execution in production evaluation systems.
package units

import (
	"strings"
	"text/template"
)

// GetTemplateFuncMap returns the standard template function map for evaluation units.
//
// The returned FuncMap is immutable and thread-safe, suitable for concurrent use
// across multiple template executions. Functions are optimized for common evaluation
// scenarios such as formatting prompts, processing answers, and calculating scores.
//
// Usage in evaluation units:
//
//	tmpl, err := template.New("prompt").Funcs(GetTemplateFuncMap()).Parse(config.Prompt)
func GetTemplateFuncMap() template.FuncMap {
	return template.FuncMap{
		// add performs integer addition.
		// Common use: converting 0-based to 1-based indexing.
		// Template usage: {{add $index 1}}
		"add": func(a, b int) int {
			return a + b
		},

		// contains reports whether substr is within s.
		// Template usage: {{if contains $answer.Content "keyword"}}
		"contains": func(s, substr string) bool {
			return strings.Contains(s, substr)
		},

		// truncate limits string length, adding "..." if truncated.
		// Returns empty string if length <= 0.
		// Preserves full string if already within limit.
		// Template usage: {{truncate $content 100}}
		"truncate": func(s string, length int) string {
			if length <= 0 {
				return ""
			}
			if len(s) <= length {
				return s
			}
			// Reserve space for ellipsis when length allows.
			if length > 3 {
				return s[:length-3] + "..."
			}
			return s[:length]
		},

		// sub performs integer subtraction.
		// Template usage: {{sub $total $used}}
		"sub": func(a, b int) int {
			return a - b
		},

		// mul performs integer multiplication.
		// Template usage: {{mul $score 100}}
		"mul": func(a, b int) int {
			return a * b
		},

		// div performs integer division.
		// Returns 0 if divisor is 0 to prevent template panics.
		// Template usage: {{div $total $count}}
		"div": func(a, b int) int {
			if b == 0 {
				return 0
			}
			return a / b
		},

		// mod returns the remainder of a/b.
		// Returns 0 if divisor is 0.
		// Template usage: {{if eq (mod $i 2) 0}}even{{end}}
		"mod": func(a, b int) int {
			if b == 0 {
				return 0
			}
			return a % b
		},

		// hasPrefix reports whether s begins with prefix.
		// Template usage: {{if hasPrefix $name "test_"}}
		"hasPrefix": func(s, prefix string) bool {
			return strings.HasPrefix(s, prefix)
		},

		// hasSuffix reports whether s ends with suffix.
		// Template usage: {{if hasSuffix $file ".json"}}
		"hasSuffix": func(s, suffix string) bool {
			return strings.HasSuffix(s, suffix)
		},

		// lower returns s with all Unicode letters mapped to lowercase.
		// Template usage: {{lower $text}}
		"lower": func(s string) string {
			return strings.ToLower(s)
		},

		// upper returns s with all Unicode letters mapped to uppercase.
		// Template usage: {{upper $code}}
		"upper": func(s string) string {
			return strings.ToUpper(s)
		},

		// trim removes leading and trailing whitespace.
		// Template usage: {{trim $userInput}}
		"trim": func(s string) string {
			return strings.TrimSpace(s)
		},

		// replace returns s with all instances of old replaced by new.
		// Template usage: {{replace $text "\n" " "}}
		"replace": func(s, old, new string) string {
			return strings.ReplaceAll(s, old, new)
		},

		// join concatenates elements with separator between them.
		// Template usage: {{join $items ", "}}
		"join": func(elems []string, sep string) string {
			return strings.Join(elems, sep)
		},

		// split slices s into substrings separated by sep.
		// Returns slice of length 1 containing s if sep is not present.
		// Template usage: {{range split $csv ","}}{{.}}{{end}}
		"split": func(s, sep string) []string {
			return strings.Split(s, sep)
		},
	}
}
