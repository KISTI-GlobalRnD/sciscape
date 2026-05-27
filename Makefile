.PHONY: install install-dev install-rust install-rust-text install-python test test-rust test-python release-check clean build docs

PYTHON ?= uv run --extra dev python
PIP ?= uv pip
CARGO ?= cargo

# Install everything via pip (Rust crates + Python package)
install:
	$(PIP) install ./rust ./rust-text .

# Editable install for development
install-dev:
	$(PIP) install -e ./rust -e ./rust-text -e ".[dev,viz,arrow,openalex,web]"

# Individual targets
install-rust:
	$(PIP) install ./rust

install-rust-text:
	$(PIP) install ./rust-text

install-python:
	$(PIP) install -e .

# Run all tests (Rust + Python)
test: test-rust test-python

test-rust:
	cd rust && $(CARGO) test --release
	cd rust-text && $(CARGO) test --release

test-python:
	$(PYTHON) -m pytest tests/ -q --tb=short

release-check:
	./scripts/release_check.sh

# Build distributable package
build:
	$(PYTHON) -m build

# Generate API documentation
docs:
	$(PYTHON) -m pdoc sciscape.clustering.auto_gamma sciscape.clustering.hierarchical \
		sciscape.clustering.integer_remap sciscape.clustering.config \
		sciscape.linkage.combine sciscape.linkage.filters \
		sciscape.openalex.edges sciscape.openalex.pipeline \
		sciscape.visualization.consensus sciscape.visualization.edge_landscape \
		sciscape.landscape \
		--output-directory docs/api --no-show-source

# Clean build artifacts
clean:
	rm -rf rust/target rust-text/target build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
