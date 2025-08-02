# go-gavel

An evaluation engine for comparing LLM outputs using composable, reusable units.

## Project Structure

This is a monorepo containing two Go modules:
- Main module at root (`/`) - Core implementation and daemon
- SDK module at `/sdk/evalengine/` - Public-facing library for external consumers

## Requirements

- Go 1.24.4+
- golangci-lint v2+

## Getting Started

```bash
# Install dependencies
go mod download

# Run tests
make test

# Run linting
make lint

# Fix auto-fixable lint issues
make fix-lint

# Run all pre-commit checks
make pre-commit
```

## Development Workflow

### Pre-Commit Hooks

This project uses git pre-commit hooks to ensure code quality. The hooks automatically run tests and linting before each commit.

#### Installing Pre-Commit Hooks

```bash
# Install the pre-commit hook
make install-hooks
```

#### Running Pre-Commit Checks Manually

```bash
# Run all pre-commit checks without committing
make pre-commit
```

#### Skipping Pre-Commit Hooks

If you need to skip the pre-commit hooks temporarily (not recommended):

```bash
git commit --no-verify
```

### Available Make Targets

- `make test` - Run all tests with race detection and coverage
- `make lint` - Run golangci-lint
- `make fix-lint` - Auto-fix linting issues where possible
- `make pre-commit` - Run all pre-commit checks (tests + lint)
- `make install-hooks` - Install git pre-commit hooks
- `make build` - Build the daemon binary
- `make clean` - Clean build artifacts and cache

## Architecture

See [docs/architecture.md](docs/architecture.md) for detailed architecture documentation.
