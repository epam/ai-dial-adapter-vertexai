## Overview

The project implements [AI DIAL API](https://epam-rail.com/dial_api) for language models and embeddings from [Vertex AI](https://console.cloud.google.com/vertex-ai).

## Supported models

The following models support `POST SERVER_URL/openai/deployments/DEPLOYMENT_NAME/chat/completions` endpoint along with optional support of `/tokenize` and `/truncate_prompt` endpoints:

|Model|Deployment name|Modality|`/tokenize`|`/truncate_prompt`|tools/functions support|
|---|---|---|---|---|---|
|Gemini 2.0 Pro|gemini-2.0-pro-exp-02-05|(text/pdf/image/audio/video)-to-text|✅|✅|✅|
|Gemini 2.0 Flash Thinking|gemini-2.0-flash-thinking-exp-01-21|text-to-text|✅|✅|❌|
|Gemini 2.0 Flash|gemini-2.0-flash-(exp\|001)|(text/pdf/image/audio/video)-to-text|✅|✅|✅|
|Gemini 2.0 Flash Lite|gemini-2.0-flash-lite-preview-02-05|(text/pdf/image/audio/video)-to-text|✅|✅|❌|
|Gemini 1.5 Pro|gemini-1.5-pro-(preview-0409\|001\|002)|(text/pdf/image/audio/video)-to-text|✅|✅|✅|
|Gemini 1.5 Flash|gemini-1.5-flash-(001\|002)|(text/pdf/image/audio/video)-to-text|✅|✅|✅|
|Gemini 1.0 Pro Vision|gemini-pro-vision|(text/pdf/image/video)-to-text|✅|✅|❌|
|Gemini 1.0 Pro|gemini-pro|text-to-text|✅|✅|✅|
|Claude 3.7 Sonnet|claude-3-7-sonnet@20250219|(text/image)-to-text|✅|✅|✅|
|Claude 3 Opus|claude-3-opus@20240229|(text/image)-to-text|✅|✅|✅|
|Claude 3.5 Sonnet v2|claude-3-5-sonnet-v2@20241022|(text/image)-to-text|✅|✅|✅|
|Claude 3.5 Sonnet|claude-3-5-sonnet@20240620|(text/image)-to-text|✅|✅|✅|
|Claude 3.5 Haiku|claude-3-5-haiku@20241022|text-to-text|✅|✅|✅|
|Claude 3 Haiku|claude-3-haiku@20240307|(text/image)-to-text|✅|✅|✅|
|Imagen 2|imagegeneration@005|text-to-image|✅|✅|❌|
|PaLM 2 Chat Bison|chat-bison@001|text-to-text|✅|✅|❌|
|PaLM 2 Chat Bison|chat-bison@002|text-to-text|✅|✅|❌|
|PaLM 2 Chat Bison|chat-bison-32k@002|text-to-text|✅|✅|❌|
|Codey for Code Chat|codechat-bison@001|text-to-text|✅|✅|❌|
|Codey for Code Chat|codechat-bison@002|text-to-text|✅|✅|❌|
|Codey for Code Chat|codechat-bison-32k@002|text-to-text|✅|✅|❌|

The models that support `/truncate_prompt` do also support `max_prompt_tokens` request parameter.

The following models support `SERVER_URL/openai/deployments/DEPLOYMENT_NAME/embeddings` endpoint:

|Model|Deployment name|Language support|Modality|
|---|---|---|---|
|Gecko Embeddings for Text V1|textembedding-gecko@001|English|text-to-embedding|
|Gecko Embeddings for Text V3|textembedding-gecko@003|English|text-to-embedding|
|Embeddings for Text|text-embedding-004|English|text-to-embedding|
|Gecko Embeddings for Text Multilingual|textembedding-gecko-multilingual@001|Multilingual|text-to-embedding|
|Embeddings for Text Multilingual|text-multilingual-embedding-002|Multilingual|text-to-embedding|
|Multimodal embeddings|multimodalembedding@001|English|(text/image)-to-embedding|

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
|DIAL_URL||URL of the core DIAL server. Optional. Used to access images stored in the DIAL File storage|
|COMPATIBILITY_MAPPING|{}|A JSON dictionary that maps VertexAI deployments that **aren't supported** by the Adapter to the VertexAI deployments that **are supported** by the Adapter _(see the [Supported models](#supported-models)_ section). Find more details in the [compatibility mode](#running-unsupported-vertexai-models-in-the-compatibility-mode) section.|

## Running unsupported VertexAI models in the compatibility mode

The Adapter supports a predefined list of VertexAI deployments. The [Supported models](#supported-models) section lists the models. These models could be accessed via `/openai/deployments/{deployment_name}/(chat_completions|embeddings)` endpoints. The Adapter won't recognize any other deployment name and will result in 404 error.

Now, suppose VertexAI has just released a new version of a model, e.g. `gemini-2.0-flash-006` which is a better version of an older `gemini-2.0-flash-001` model.

Immediately after the release, the former model is **unsupported** by the Adapter, but the latter is **supported**.
Therefore, the request to `openai/deployments/gemini-2.0-flash-006/chat/completions` will result in 404 error.

It will take some time for the Adapter to catch up with VertexAI - support the v6 model and publish the release with the fix.

What to do in the meantime? Presumably, the v6 model is backward compatible with v1, so we may try to run v6 in **the compatibility mode** - that is to convince the Adapter to process v6 request as if it's v1 request with the only difference that the final upstream request to AWS Bedrock will be to v6 and not v1.

The `COMPATIBILITY_MAPPING` env variable enables exactly this scenario.

When it's defined like this:

```json
COMPATIBILITY_MAPPING={"gemini-2.0-flash-006": "gemini-2.0-flash-001"}
```

the Adapter will be able to handle requests to the `gemini-2.0-flash-006` deployment.
The requests will be processed by the same pipeline as `gemini-2.0-flash-001`, but the call to AWS Bedrock will be done to `gemini-2.0-flash-006` deployment name.

Naturally, this will only work if the APIs of v1 and v6 deployments are compatible:

1. The requests utilizing the modalities supported by both v1 and v6 will work just fine.
2. However, the requests with modalities that are supported by v6 and aren't supported by v1, won't be processed correctly. You will have to wait until the Adapter supports the v6 deployment natively.

When a version of the Adapter supporting the v6 model is released, you may migrate to it and safely remove the entry from the `COMPATIBILITY_MAPPING` dictionary.

Note that a mapping such as this one would be ineffectual:

```json
COMPATIBILITY_MAPPING={"gemini-2.0-flash-006": "imagegeneration@005"}
```

since the APIs and capabilities of these two models are drastically different.

## Load balancing

If you use DIAL Core load balancing mechanism, you can provide `extraData` upstream setting the region and the project to use for a particular upstream:

```json
{
  "upstreams": [
    {
      "extraData": {
        "project": "project1",
        "region": "us-central1"
      }
    },
    {
      "extraData": {
        "project": "project1",
        "region": "us-east5"
      }
    },
    {
      "extraData": {
        "project": "project2"
      }
    }
  ]
}
```

The fields in the extra data override the corresponding environment variables:

|`extraData` field|Env variable|
|---|---|
|`region`|`DEFAULT_REGION`|
|`project`|`GCP_PROJECT_ID`|

> [!NOTE]
> The region and project configuration is only supported for Gemini 2.0 and Anthropic models.

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
