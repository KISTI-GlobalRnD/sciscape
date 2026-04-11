.PHONY: install install-dev install-rust install-rust-text install-python test clean

# Install everything via pip (Rust crates + Python package)
install:
	pip install ./rust ./rust-text .

# Editable install for development
install-dev:
	pip install -e ./rust -e ./rust-text -e .

# Individual targets
install-rust:
	pip install ./rust

install-rust-text:
	pip install ./rust-text

install-python:
	pip install -e .

# Run all tests
test:
	python -m pytest tests/ -q

# Clean Rust build artifacts
clean:
	rm -rf rust/target rust-text/target
