## Overview

The project implements [AI DIAL API](https://epam-rail.com/dial_api) for language models and embeddings from [Vertex AI](https://console.cloud.google.com/vertex-ai).

## Supported models

The following models support `POST SERVER_URL/openai/deployments/DEPLOYMENT_NAME/chat/completions` endpoint along with optional support of `/tokenize` and `/truncate_prompt` endpoints:

|Model|Deployment name|Modality|`/tokenize`|`/truncate_prompt`|tools/functions support|
|---|---|---|---|---|---|
|Gemini 1.5 Pro|gemini-1.5-pro-preview-0409|(text/pdf/image/audio/video)-to-text|✅|❌|✅|
|Gemini 1.0 Pro Vision|gemini-pro-vision|(text/pdf/image/video)-to-text|✅|❌|❌|
|Gemini 1.0 Pro|gemini-pro|text-to-text|✅|❌|✅|
|Imagen 2|imagegeneration@005|text-to-image|✅|✅|❌|
|PaLM 2 Chat Bison|chat-bison@001|text-to-text|✅|✅|❌|
|PaLM 2 Chat Bison|chat-bison@002|text-to-text|✅|✅|❌|
|PaLM 2 Chat Bison|chat-bison-32k@002|text-to-text|✅|✅|❌|
|Codey for Code Chat|codechat-bison@001|text-to-text|✅|✅|❌|
|Codey for Code Chat|codechat-bison@002|text-to-text|✅|✅|❌|
|Codey for Code Chat|codechat-bison-32k@002|text-to-text|✅|✅|❌|

The models that support `/truncate_prompt` do also support `max_prompt_tokens` request parameter.

The following models support `SERVER_URL/openai/deployments/DEPLOYMENT_NAME/embeddings` endpoint:

|Model|Deployment name|Modality|
|---|---|---|
|Embeddings for Text|textembedding-gecko@001|text-to-embedding|

## Developer environment

This project uses [Python>=3.11](https://www.python.org/downloads/) and [Poetry>=1.6.1](https://python-poetry.org/) as a dependency manager.

Check out Poetry's [documentation on how to install it](https://python-poetry.org/docs/#installation) on your system before proceeding.

To install requirements:

```sh
poetry install
```

This will install all requirements for running the package, linting, formatting and tests.

### IDE configuration

The recommended IDE is [VSCode](https://code.visualstudio.com/).
Open the project in VSCode and install the recommended extensions.

The VSCode is configured to use PEP-8 compatible formatter [Black](https://black.readthedocs.io/en/stable/index.html).

Alternatively you can use [PyCharm](https://www.jetbrains.com/pycharm/).

Set-up the Black formatter for PyCharm [manually](https://black.readthedocs.io/en/stable/integrations/editors.html#pycharm-intellij-idea) or
install PyCharm>=2023.2 with [built-in Black support](https://blog.jetbrains.com/pycharm/2023/07/2023-2/#black).

## Run

Run the development server:

```sh
make serve
```

Open `localhost:5001/docs` to make sure the server is up and running.

## Environment Variables

Copy `.env.example` to `.env` and customize it for your environment:

|Variable|Default|Description|
|---|---|---|
|GOOGLE_APPLICATION_CREDENTIALS||Filepath to JSON with [credentials](https://cloud.google.com/docs/authentication/application-default-credentials#GAC)|
|DEFAULT_REGION||Default region for Vertex AI (e.g. "us-central1")|
|GCP_PROJECT_ID||GCP project ID|
|LOG_LEVEL|INFO|Log level. Use DEBUG for dev purposes and INFO in prod|
|AIDIAL_LOG_LEVEL|WARNING|AI DIAL SDK log level|
|WEB_CONCURRENCY|1|Number of workers for the server|
|TEST_SERVER_URL|http://0.0.0.0:5001|Server URL used in the integration tests|
|DIAL_URL||URL of the core DIAL server. Optional. Used to access images stored in the DIAL File storage|

### Docker

Run the server in Docker:

```sh
make docker_serve
```

## Lint

Run the linting before committing:

```sh
make lint
```

To auto-fix formatting issues run:

```sh
make format
```

## Test

Run unit tests locally:

```sh
make test
```

Run unit tests in Docker:

```sh
make docker_test
```

Run integration tests locally:

```sh
make integration_tests
```

## Clean

To remove the virtual environment and build artifacts:

```sh
make clean
```