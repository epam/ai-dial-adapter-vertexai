# Overview

The project implements [AI DIAL API](https://epam-rail.com/dial_api) for language models and embeddings from [Vertex AI](https://console.cloud.google.com/vertex-ai).

## Supported models

### Chat completion models

The following models support `POST $SERVER_ORIGIN/openai/deployments/$DEPLOYMENT_NAME/chat/completions` endpoint along with an optional support of the feature endpoints:

* `POST $SERVER_ORIGIN/openai/deployments/$DEPLOYMENT_NAME/tokenize`
* `POST $SERVER_ORIGIN/openai/deployments/$DEPLOYMENT_NAME/truncate_prompt`
* `POST $SERVER_ORIGIN/openai/deployments/$DEPLOYMENT_NAME/configuration`

|Model|Deployment name|Modality|`/tokenize`|`/truncate_prompt`|tools/functions support|`/configuration`|
|---|---|---|---|---|---|---|
|Gemini 2.5 Flash|gemini-2.5-flash|(text/pdf/image/audio/video)-to-text|✅|✅|✅|✅|
|Gemini 2.5 Flash Image|gemini-2.5-flash-image-preview|(text/image)-to-(text/image)|✅|✅|✅|✅|
|Gemini 2.5 Pro|gemini-2.5-pro|(text/pdf/image/audio/video)-to-text|✅|✅|✅|✅|
|Gemini 2.0 Flash Lite|gemini-2.0-flash-lite-001|(text/pdf/image/audio/video)-to-text|✅|✅|✅|❌|
|Gemini 2.0 Flash|gemini-2.0-flash-(exp\|001)|(text/pdf/image/audio/video)-to-text|✅|✅|✅|❌|
|Gemini 1.5 Pro|gemini-1.5-pro-002|(text/pdf/image/audio/video)-to-text|✅|✅|✅|❌|
|Gemini 1.5 Flash|gemini-1.5-flash-002|(text/pdf/image/audio/video)-to-text|✅|✅|✅|❌|
|Claude 4 Opus|claude-opus-4@20250514|(pdf/text/image)-to-text|✅|✅|✅|✅|
|Claude 4 Sonnet|claude-sonnet-4@20250514|(pdf/text/image)-to-text|✅|✅|✅|✅|
|Claude 3.7 Sonnet|claude-3-7-sonnet@20250219|(pdf/text/image)-to-text|✅|✅|✅|✅|
|Claude 3 Opus|claude-3-opus@20240229|(text/image)-to-text|✅|✅|✅|✅|
|Claude 3.5 Sonnet v2|claude-3-5-sonnet-v2@20241022|(pdf/text/image)-to-text|✅|✅|✅|✅|
|Claude 3.5 Sonnet|claude-3-5-sonnet@20240620|(pdf/text/image)-to-text|✅|✅|✅|✅|
|Claude 3.5 Haiku|claude-3-5-haiku@20241022|(pdf/text)-to-text|✅|✅|✅|✅|
|Claude 3 Haiku|claude-3-haiku@20240307|(text/image)-to-text|✅|✅|✅|✅|
|Imagen 4.0|imagen-4.0-(generate-preview-06-06\|fast-generate-preview-06-06\|ultra-generate-preview-06-06\|generate-001\|fast-generate-001\|ultra-generate-001)|text-to-image|✅|✅|❌|✅|
|Imagen 3.0|imagen-3.0-(generate-001\|generate-002\|fast-generate-001)|text-to-image|✅|✅|❌|✅|
|Imagen 2|imagegeneration@005|text-to-image|✅|✅|❌|✅|
|PaLM 2 Chat Bison|chat-bison@001|text-to-text|✅|✅|❌|❌|
|PaLM 2 Chat Bison|chat-bison@002|text-to-text|✅|✅|❌|❌|
|PaLM 2 Chat Bison|chat-bison-32k@002|text-to-text|✅|✅|❌|❌|
|Codey for Code Chat|codechat-bison@001|text-to-text|✅|✅|❌|❌|
|Codey for Code Chat|codechat-bison@002|text-to-text|✅|✅|❌|❌|
|Codey for Code Chat|codechat-bison-32k@002|text-to-text|✅|✅|❌|❌|

The models that support `/truncate_prompt` do also support `max_prompt_tokens` chat completion request parameter.

#### Configurable models

Certain models support configuration via the `/configuration` endpoint.
GET request to this endpoint returns the schema of the model configuration in [JSON Schema](https://json-schema.org/) format.
Such models expect that `custom_fields.configuration` field of the `chat/completions` request will contain a JSON value that conforms to the schema.
The `custom_fields.configuration` field is optional iff each field in the schema is optional too.

##### Imagen models

The Imagen models support configuration of parameters specific for image-generation such as negative prompt, aspect ratio and watermarking. See the complete list of configurable parameters at the [Imagen API documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/imagen-api#python_1).

```json
{
  "messages": [{"role": "user", "content": "forest meadow"}],
  "custom_fields": {
    "configuration": {
      "add_watermark": false,
      "negative_prompt": "trees",
      "aspect_ratio": "16:9"
    }
  }
}
```

##### Gemini 2.5 models

The Gemini 2.5 series models support configuration of [thinking](https://ai.google.dev/gemini-api/docs/thinking) parameters:

```json
{
  "custom_fields": {
    "configuration": {
      "thinking": {
        "include_thoughts": true,
        "thinking_budget": 2048
      }
    }
  }
}
```

The thought summaries are printed into a dedicated `Thinking` stage.
The content of the thinking stage isn't provided to the model in subsequent requests.

##### Claude models

The Claude models accept a configuration flag that enables document citations in the generated output. The flag is false by default.

```json
{
  "custom_fields": {
    "configuration": {
      "enable_citations": true
    }
  }
}
```

Not every Claude model supports citations. Refer to the [official documentation](https://docs.anthropic.com/en/docs/build-with-claude/citations) before utilizing any flags.

Besides that Claude models support beta flags.
The whole list of flags could be found in the [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/anthropic_beta_param.py).

The most notable beta flags are:

|Configuration|Comment|Scope|
|---|---|---|
|`{"betas": ["token-efficient-tools-2025-02-19"]}`|[Token-efficient tool use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/token-efficient-tool-use)|Claude 3.7 Sonnet|
|`{"betas": ["output-128k-2025-02-19"]}`|[Extended output length](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#extended-output-capabilities-beta)|Claude 3.7 Sonnet|

Not every model supports all flags. Refer to the official documentation before utilizing any flags.

### Embedding models

The following models support `$SERVER_ORIGIN/openai/deployments/$DEPLOYMENT_NAME/embeddings` endpoint:

|Model|Deployment name|Language support|Modality|
|---|---|---|---|
|Gecko Embeddings for Text V1|textembedding-gecko@001|English|text-to-embedding|
|Gecko Embeddings for Text V3|textembedding-gecko@003|English|text-to-embedding|
|Embeddings for Text|text-embedding-004|English|text-to-embedding|
|Gecko Embeddings for Text Multilingual|textembedding-gecko-multilingual@001|Multilingual|text-to-embedding|
|Embeddings for Text Multilingual|text-multilingual-embedding-002|Multilingual|text-to-embedding|
|Multimodal embeddings|multimodalembedding@001|English|(text/image)-to-embedding|

## Environment variables

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
|COMPATIBILITY_MAPPING|{}|A JSON dictionary that maps VertexAI deployments that **aren't supported** by the Adapter to the VertexAI deployments that **are supported** by the Adapter _(see the [Supported models](#supported-models)_ section). Find more details in the [compatibility mode](#compatibility-mode) section.|
|CLAUDE_DEFAULT_MAX_TOKENS|1536|The default value of `max_tokens` chat completion parameter if it is not provided in the request.<br>**:warning: Using the variable is discouraged**.<br>Consider configuring the default in the DIAL Core Config instead as demonstrated in the [example below](#default-max_tokens-for-claude-models).|

## Default `max_tokens` for Claude models

Unlike Gemini models, Claude models require the `max_tokens` parameter in the chat completion request.

We recommend configuring `max_tokens` default value on a per-model basis in the DIAL Core Config, for example:

```json
{
    "models": {
        "dial-claude-deployment-id": {
            "type": "chat",
            "description": "...",
            "endpoint": "...",
            "defaults": {
                "max_tokens": 2048
            }
        }
    }
}
```

If the default is missing in the DIAL Core Config, it will be taken from the `CLAUDE_DEFAULT_MAX_TOKENS` environment variable.
However, we strongly recommend not to rely on this variable and instead configure the defaults in the DIAL Core Config.
Such a **per-model** configuration is operationally cleaner since all the information relevant to tokens _(like pricing and token limits)_ is kept in the same place.

The default value set in the DIAL Core Config takes precedence over the one configured in the adapter.

Make sure the default doesn't exceed Claude's [max output tokens](https://docs.anthropic.com/en/docs/about-claude/models/all-models#model-comparison-table), otherwise, you will receive an error like this one: `max_tokens: 10000 > 8192, which is the maximum allowed number of output tokens for claude-3...)`.

## Compatibility mode

The Adapter supports a predefined list of VertexAI deployments. The [Supported models](#supported-models) section lists the models. These models could be accessed via `/openai/deployments/{deployment_name}/(chat_completions|embeddings)` endpoints. The Adapter won't recognize any other deployment name and will result in `404` error.

Now, suppose VertexAI has just released a new version of a model, e.g. `gemini-2.0-flash-006` which is a better version of an older `gemini-2.0-flash-001` model.

Immediately after the release, the former model is **unsupported** by the Adapter, but the latter is **supported**.
Therefore, the request to `openai/deployments/gemini-2.0-flash-006/chat/completions` will result in 404 error.

It will take some time for the Adapter to catch up with VertexAI - support the v6 model and publish the release with the fix.

What to do in the meantime? Presumably, the v6 model is backward compatible with v1, so we may try to run v6 in **the compatibility mode** - that is to convince the Adapter to process v6 request as if it's v1 request with the only difference that the final upstream request to AWS Bedrock will be to v6 and not v1.

The `COMPATIBILITY_MAPPING` env variable enables exactly this scenario.

When it's defined like this:

```txt
COMPATIBILITY_MAPPING={"gemini-2.0-flash-006": "gemini-2.0-flash-001"}
```

the Adapter will be able to handle requests to the `gemini-2.0-flash-006` deployment.
The requests will be processed by the same pipeline as `gemini-2.0-flash-001`, but the call to AWS Bedrock will be done to `gemini-2.0-flash-006` deployment name.

Naturally, this will only work if the APIs of v1 and v6 deployments are compatible:

1. The requests utilizing the modalities supported by both v1 and v6 will work just fine.
2. However, the requests with modalities that are supported by v6 and aren't supported by v1, won't be processed correctly. You will have to wait until the Adapter supports the v6 deployment natively.

When a version of the Adapter supporting the v6 model is released, you may migrate to it and safely remove the entry from the `COMPATIBILITY_MAPPING` dictionary.

Note that a mapping such as this one would be ineffectual:

```txt
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
    },
    {
      "key": "api-key"
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
> The region and project configuration is only supported for Gemini>=2 and Anthropic models.

### Global endpoint

Use the `global` region to enable the [global endpoint](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#global-endpoint):

```json
{
  "upstreams": [
    {
      "extraData": {
        "region": "global"
      }
    }
  ]
}
```

> [!NOTE]
> The global endpoint is supported only for [certain models](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#supported_models) and has a few other [limitations](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#limitations).

## Prompt caching

### Implicit caching

Gemini 2.5 models support [implicit context caching](https://ai.google.dev/gemini-api/docs/caching?lang=python#implicit-caching).

Any request over a certain amount of tokens will be automatically cached. The token threshold triggering caching for Gemini 2.5 Flash is 1024 and for Gemini 2.5 Pro - 4096.

Set `autoCachingSupported` flag in the DIAL Core config for a deployment of interest to enable this feature:

```json
{
  "models": {
    "my-dial-gemini-deployment": {
      "type": "chat",
      "displayName": "Gemini 2.5 Flash",
      "endpoint": "$VERTEXAI_ADAPTER_ORIGIN/openai/deployments/gemini-2.5-flash/chat/completions",
      "upstreams": [
        {
          "extraData": {
            "region": "us-central1"
          }
        },
        {
          "extraData": {
            "region": "us-east5"
          }
        }
      ],
      "features": {
        "autoCachingSupported": true
      }
    }
  }
}
```

On a cache hit, `usage.prompt_tokens_details.cached_tokens` field reports the number of cached prompt tokens.

## Authentication

### GCP Vertex AI

Access to GCP Vertex AI is authenticated via Application Default Credentials ([ADC](https://cloud.google.com/docs/authentication/application-default-credentials)) with region and project configured either:

1. globally via `DEFAULT_REGION` and `GCP_PROJECT_ID` environment vars, or
2. on a [per upstream basis](#load-balancing) via `upstreams.extraData` fields in DIAL Core Config.

### Anthropic API / Google AI Platform

Gemini>=2 and Anthropic deployments could be accessed via API key. The API keys should be configured per-upstream in the DIAL Core config:

```json
{
  "models": {
    "gemini-2.0-flash-lite-001": {
      "endpoint": "...",
      "upstreams": [
        {
          "key": "gemini-api-key"
        }
      ]
    },
    "claude-3-5-sonnet-20241022": {
      "endpoint": "...",
      "upstreams": [
        {
          "key": "anthropic-api-key"
        }
      ]
    }
  }
}
```

Keep in mind that the same Anthropic models have [different identifiers](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) in Anthropic API and GPC Vertex AI.

E.g. `claude-3-5-sonnet-v2@20241022` in GCP Vertex AI corresponds to `claude-3-5-sonnet-20241022` in Anthropic API.

The adapter uses deployment identifiers from **GCP Vertex AI**.
Therefore, in order to use Anthropic API model you need to map its identifier to a corresponding identifier in GCP Vertex AI using the [compatibility mapping](#compatibility-mode):

```txt
COMPATIBILITY_MAPPING={"claude-3-5-sonnet-20241022":"claude-3-5-sonnet-v2@20241022"}
```

Otherwise, the adapter will return 404 on requests to `claude-3-5-sonnet-20241022`.

## Development

Frequently used actions are automated via `make` targets.

### Install

To install the project dependencies required for running the server, linting and formatting the source code, and running the tests:

```sh
make install
```

### Serve locally

To run the development server:

```sh
make serve
```

Open `localhost:5001/docs` to make sure the server is up and running.

### Serve from the Docker container

To run the server from the Docker container:

```sh
make docker_serve
```

### Lint

Don't forget to run the linting before committing:

```sh
make lint
```

To auto-fix formatting issues run:

```sh
make format
```

### Test

To run the unit tests locally:

```sh
make test
```

To run the unit tests from the Docker container:

```sh
make docker_test
```

To run the integration tests locally:

```sh
make integration_tests
```

### Clean

To remove the virtual environment and build artifacts:

```sh
make clean
```

### IDE

The recommended IDE is [VSCode](https://code.visualstudio.com/).
Open the project in VSCode and install the recommended extensions.

The VSCode is configured to use PEP-8 compatible formatter [Black](https://black.readthedocs.io/en/stable/index.html).

Alternatively you can use [PyCharm](https://www.jetbrains.com/pycharm/).

Set-up the Black formatter for PyCharm [manually](https://black.readthedocs.io/en/stable/integrations/editors.html#pycharm-intellij-idea) or
install PyCharm>=2023.2 with [built-in Black support](https://blog.jetbrains.com/pycharm/2023/07/2023-2/#black).
