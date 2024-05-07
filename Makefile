PORT ?= 5001
IMAGE_NAME ?= ai-dial-adapter-vertexai
PLATFORM ?= linux/amd64
DEV_PYTHON ?= 3.11
DOCKER ?= docker
ARGS=

.PHONY: all install build serve clean lint format test integration_tests docker_build docker_run

all: build

install:
	poetry env use python$(DEV_PYTHON)
	poetry install

build: install
	poetry build

serve: install
	poetry run uvicorn "aidial_adapter_vertexai.app:app" \
		--reload --host "0.0.0.0" --port $(PORT) \
		--workers=1 --env-file ./.env

clean:
	poetry run python -m scripts.clean
	poetry env remove --all

lint: install
	poetry run nox -s lint

format: install
	poetry run nox -s format

test: install
	poetry run nox -s test

integration_tests: install
	poetry run nox -s integration_tests

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
	@echo '-- LINTING --'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo '-- RUN --'
	@echo 'serve                        - run the dev server locally'
	@echo 'docker_serve                 - run the dev server from the docker'
	@echo '-- TESTS --'
	@echo 'test                         - run unit tests'
	@echo 'test ARGS=<test_file>        - run all tests in file'
	@echo 'docker_test                  - run unit tests from the docker'
	@echo 'integration_tests            - run integration tests'