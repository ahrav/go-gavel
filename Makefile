.PHONY: test lint build clean pre-commit fix-lint install-hooks

# Default target
all: test lint

test:
	go test -race -cover ./...

lint:
	golangci-lint run

fix-lint:
	golangci-lint run --fix

build:
	go build -o bin/evald ./cmd/evald

clean:
	rm -rf bin/
	go clean -cache

# Pre-commit runs all checks to ensure code quality before committing
pre-commit: test lint
	@echo "✅ All pre-commit checks passed!"

# Install git hooks for the project
install-hooks:
	@echo "Installing git pre-commit hook..."
	@mkdir -p .git/hooks
	@echo '#!/bin/sh' > .git/hooks/pre-commit
	@echo '# Auto-generated pre-commit hook' >> .git/hooks/pre-commit
	@echo 'echo "Running pre-commit checks..."' >> .git/hooks/pre-commit
	@echo 'make pre-commit' >> .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "✅ Git pre-commit hook installed successfully!"
	@echo "To skip hooks, use: git commit --no-verify"