.DEFAULT_GOAL = help

UV := uv

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[1;34mmake %-10s\033[0m%s\n", $$1, $$2}'

.PHONY: install
install: ## sync the development environment and install pre-commit hooks
	@printf "Syncing environment...\n"
	$(UV) sync --all-groups
	$(UV) run pre-commit install

.PHONY: tests
tests: ## run tests
	@printf "Running tests...\n"
	$(UV) run pytest --html=report.html --self-contained-html --cov-report=xml

.PHONY: lint
lint: ## check formatting and lint rules
	@printf "Linting...\n"
	$(UV) run ruff check .
	$(UV) run ruff format --check .

.PHONY: format
format: ## apply formatting and autofixes
	@printf "Formatting...\n"
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

.PHONY: typecheck
typecheck: ## run static type checking
	@printf "Type checking...\n"
	$(UV) run mypy

.PHONY: docs
docs: ## build the doc site (CI)
	@printf "Building doc site...\n"
	$(UV) run --group docs mkdocs build

.PHONY: docs-serve
docs-serve: ## preview the docs locally
	@printf "Serving doc site...\n"
	$(UV) run --group docs mkdocs serve

.PHONY: lock
lock: ## verify that uv.lock is in sync with pyproject.toml
	@printf "Checking lock...\n"
	$(UV) lock --check

.PHONY: clean
clean: ## clean project directory
	@printf "Cleaning project...\n"
	rm -f report.html coverage.xml .coverage
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf **/__pycache__/
	rm -rf site/ dist/
	rm -f examples/*.md.tmp
