.PHONY: install install-rust install-rust-text install-python test clean

# Install everything: Rust crates + Python package
install: install-rust install-rust-text install-python

# Rust Leiden backend (clustering)
install-rust:
	cd rust && maturin develop --release

# Rust text backend (keyword extraction)
install-rust-text:
	cd rust-text && maturin develop --release

# Python package (editable)
install-python:
	pip install -e .

# Run all tests
test:
	python -m pytest tests/ -q

# Clean Rust build artifacts
clean:
	rm -rf rust/target rust-text/target
