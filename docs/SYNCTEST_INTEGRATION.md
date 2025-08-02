# Synctest Integration Guide

This document describes how to use Go's experimental `synctest` package for testing timing-based code in the go-gavel project.

## Overview

The `synctest` package provides controlled time simulation for testing concurrent code that relies on timing. It's particularly useful for testing retry logic, timeouts, and other time-sensitive operations without waiting for real time to pass.

## Prerequisites

- Go 1.24+ (synctest is experimental in Go 1.24)
- `GOEXPERIMENT=synctest` environment variable set when running tests

## Benefits

1. **Speed**: Tests that would normally take seconds or minutes to run due to delays complete in milliseconds
2. **Determinism**: Eliminates timing-related test flakiness
3. **Precision**: Allows exact verification of timing behavior
4. **Isolation**: Each test runs in its own "bubble" with controlled time

## Usage Patterns

### Basic Synctest Test Structure

```go
//go:build goexperiment.synctest

package mypackage

import (
    "testing"
    "testing/synctest"
    "time"
)

func TestWithSynctest(t *testing.T) {
    synctest.Run(func() {
        // Your test code here
        // Time is controlled within this function
        start := time.Now()
        time.Sleep(1 * time.Second)
        elapsed := time.Since(start)
        // elapsed will be exactly 1 second
    })
}
```

### Testing Retry Logic

Example from `infrastructure/llm/retry_client_synctest_test.go`:

```go
func TestComplete_RetryWithSynctest(t *testing.T) {
    synctest.Run(func() {
        // Configure retry client with realistic delays
        config := RetryConfig{
            MaxAttempts:   2,
            BaseDelay:     1 * time.Second,
            MaxDelay:      10 * time.Second,
            JitterPercent: 0.0,
        }
        
        // Test runs quickly but simulates full delay periods
        start := time.Now()
        response, err := retryClient.Complete(context.Background(), "test prompt", nil)
        elapsed := time.Since(start)
        
        // Can verify exact timing without waiting
        expectedDelay := 1*time.Second + 2*time.Second
        if elapsed < expectedDelay {
            t.Errorf("Expected at least %v elapsed time, got %v", expectedDelay, elapsed)
        }
    })
}
```

### Testing Context Cancellation

```go
func TestComplete_ContextCancellationWithSynctest(t *testing.T) {
    synctest.Run(func() {
        ctx, cancel := context.WithCancel(context.Background())
        
        // Cancel context after a delay in separate goroutine
        go func() {
            time.Sleep(500 * time.Millisecond)
            cancel()
        }()
        
        // Test cancellation behavior deterministically
        response, err := retryClient.Complete(ctx, "test prompt", nil)
        // Context cancellation timing is precise and predictable
    })
}
```

## File Organization

### Separate Files for Synctest Tests

Create separate test files with build constraints for synctest tests:

```go
//go:build goexperiment.synctest

package mypackage
// synctest tests here
```

File naming convention: `*_synctest_test.go`

### Benefits of Separation

1. Tests can run without synctest dependency in normal CI
2. Clear separation between regular and timing-based tests
3. Easy to enable/disable synctest tests based on environment

## Running Synctest Tests

### Local Development

```bash
# Run synctest tests specifically
GOEXPERIMENT=synctest go test -v ./infrastructure/llm -run Synctest

# Run all tests including synctest
GOEXPERIMENT=synctest go test ./...
```

### CI Integration

Update CI workflows to optionally run synctest tests:

```yaml
- name: Run Synctest Tests
  run: GOEXPERIMENT=synctest go test -v ./...
  env:
    GOEXPERIMENT: synctest
```

## Best Practices

### 1. Use Realistic Delays

With synctest, you can use realistic production delays without impacting test speed:

```go
// ✅ Good: Use realistic delays
config := RetryConfig{
    BaseDelay: 1 * time.Second,
    MaxDelay:  30 * time.Second,
}

// ❌ Avoid: Artificially short delays just for testing
config := RetryConfig{
    BaseDelay: 1 * time.Millisecond,
    MaxDelay:  10 * time.Millisecond,
}
```

### 2. Verify Timing Precisely

```go
start := time.Now()
// ... operation that should take specific time
elapsed := time.Since(start)

// Can verify exact timing with synctest
if elapsed != expectedDuration {
    t.Errorf("Expected exactly %v, got %v", expectedDuration, elapsed)
}
```

### 3. Test Edge Cases

Use synctest to test timing edge cases that would be difficult to reproduce reliably:

```go
// Test exactly when context cancellation occurs
go func() {
    time.Sleep(499 * time.Millisecond) // Just before first retry
    cancel()
}()
```

### 4. Combine with Traditional Tests

Use synctest tests to complement, not replace, traditional unit tests:

- Traditional tests: Business logic, error handling, data transformation
- Synctest tests: Timing behavior, retry logic, timeout handling

## Limitations

1. **Experimental**: synctest is experimental and may change
2. **Go Version**: Requires Go 1.24+ with experimental flag
3. **External I/O**: Only works with time-based operations within the bubble
4. **Network Calls**: Real network operations are not controlled by synctest

## Migration Strategy

### Phase 1: Add Synctest Tests (Current)

- Create separate synctest test files for timing-critical components
- Keep existing traditional tests unchanged
- Run synctest tests in development and optionally in CI

### Phase 2: Enhance Coverage

- Add synctest tests for all timing-dependent code
- Use synctest for integration tests involving multiple timing components
- Document timing contracts more precisely

### Phase 3: Production Readiness

- When synctest becomes stable (likely Go 1.25+), integrate more deeply
- Consider using synctest for all timing-based tests
- Update CI to always run synctest tests

## Examples in Codebase

- `infrastructure/llm/retry_client_synctest_test.go`: Retry logic testing
- Future: middleware timeout testing, circuit breaker testing

## Troubleshooting

### Build Constraints

If tests don't run:
1. Ensure `//go:build goexperiment.synctest` is the first line
2. Check that `GOEXPERIMENT=synctest` is set
3. Verify Go version is 1.24+

### Time Mismatch Errors

If timing assertions fail:
1. Ensure all time operations are within the synctest.Run bubble
2. Check for external time dependencies (network, file I/O)
3. Verify goroutines are properly synchronized within the bubble

### Performance Issues

If tests are slower than expected:
1. Check for operations outside the synctest bubble
2. Ensure proper use of synctest.Wait() for goroutine synchronization
3. Verify no external blocking operations