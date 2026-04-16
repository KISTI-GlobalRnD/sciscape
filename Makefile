.PHONY: install install-dev install-rust install-rust-text install-python test test-rust test-python clean build

# Install everything via pip (Rust crates + Python package)
install:
	pip install ./rust ./rust-text .

# Editable install for development
install-dev:
	pip install -e ./rust -e ./rust-text -e ".[dev,viz,arrow,openalex,web]"

# Individual targets
install-rust:
	pip install ./rust

install-rust-text:
	pip install ./rust-text

install-python:
	pip install -e .

# Run all tests (Rust + Python)
test: test-rust test-python

test-rust:
	cd rust && cargo test --release
	cd rust-text && cargo test --release

test-python:
	python -m pytest tests/ -q --tb=short

# Build distributable package
build:
	python -m build

# Clean build artifacts
clean:
	rm -rf rust/target rust-text/target build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
