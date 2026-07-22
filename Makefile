PORT ?= 5001
IMAGE_NAME ?= ai-dial-adapter-vertexai
PLATFORM ?= linux/amd64
DOCKER ?= docker
POETRY ?= poetry
POETRY_PYTHON ?= python
PYDANTIC_V2 ?= 1
VENV_DIR ?= .venv

# Check for CI environment
# Empty string means false in Makefile
CI ?=
NOX_DEFAULT_VENV_BACKEND := $(if $(CI),none,virtualenv)

.PHONY: all init_env install build serve clean lint format test integration_tests docker_build docker_run

-include .env.dev
export

all: build

init_env:
	$(if $(CI),,$(POETRY) env use $(POETRY_PYTHON))

install: init_env
	$(POETRY) install

build: install
	$(POETRY) build

serve: install
	$(POETRY) run uvicorn "aidial_adapter_vertexai.app:app" \
		--reload --host "0.0.0.0" --port $(PORT) \
		--workers=1 --env-file ./.env

clean:
	$(POETRY) run python -m scripts.clean
	$(POETRY) env remove --all

install_git_hooks: install
	$(VENV_DIR)/bin/pre-commit install

lint: install
	$(POETRY) run nox -s lint

format: install
	$(POETRY) run nox -s format

test: install
	$(POETRY) run nox -s test

integration_tests: install
	$(POETRY) run nox -s integration_tests

docker_test:
	$(DOCKER) build --platform $(PLATFORM) -f Dockerfile.test -t $(IMAGE_NAME):test .
	$(DOCKER) run --platform $(PLATFORM) --rm $(IMAGE_NAME):test

docker_serve:
	$(DOCKER) build --platform $(PLATFORM) -t $(IMAGE_NAME):dev .
	$(DOCKER) run --platform $(PLATFORM) --env-file ./.env --rm -p $(PORT):5000 $(IMAGE_NAME):dev

help:
	@echo '===================='
	@echo 'build                        - build the source and wheels archives'
	@echo 'clean                        - clean virtual env and build artifacts'
	@echo 'install_git_hooks            - install the git hooks'
	@echo '-- LINTING --'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo '-- RUN --'
	@echo 'serve                        - run the dev server locally'
	@echo 'docker_serve                 - run the dev server from the docker'
	@echo '-- TESTS --'
	@echo 'test                         - run unit tests'
	@echo 'docker_test                  - run unit tests from the docker'
	@echo 'integration_tests            - run integration tests'
