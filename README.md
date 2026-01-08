<h1 align="center">
  DIAL VertexAI Adapter
</h1>
<p align="center">
  <p align="center">
  <a href="https://dialx.ai/">
    <img src="https://dialx.ai/dialx_logo.svg" alt="About DIALX">
  </a>
</p>
<h4 align="center">
  <a href="https://discord.gg/ukzj9U9tEe">
    <img src="https://img.shields.io/static/v1?label=DIALX%20Community%20on&message=Discord&color=blue&logo=Discord&style=flat-square" alt="Discord">
  </a>
</h4>

- [Overview](#overview)
  - [Supported models](#supported-models)
    - [Chat completion models](#chat-completion-models)
      - [Image editing in Gemini 2.5 Flash Image](#image-editing-in-gemini-25-flash-image)
      - [Configurable models](#configurable-models)
        - [Imagen models](#imagen-models)
        - [Veo models](#veo-models)
        - [Gemini 2.5, Gemini 3 models](#gemini-25-gemini-3-models)
        - [Gemini 2.5 Flash Image model](#gemini-25-flash-image-model)
        - [Claude models](#claude-models)
    - [Embedding models](#embedding-models)
  - [Environment variables](#environment-variables)
    - [Default `max_tokens` for Claude models](#default-max_tokens-for-claude-models)
  - [Compatibility mode](#compatibility-mode)
    - [Compatibility configuration in DIAL Core config](#compatibility-configuration-in-dial-core-config)
    - [Compatibility configuration in Adapter](#compatibility-configuration-in-adapter)
  - [Load balancing](#load-balancing)
    - [Global endpoint](#global-endpoint)
  - [Prompt caching](#prompt-caching)
    - [Implicit caching](#implicit-caching)
  - [Authentication](#authentication)
    - [GCP Vertex AI](#gcp-vertex-ai)
    - [Anthropic API / Google AI Platform](#anthropic-api--google-ai-platform)
    - [Anthropic Foundry](#anthropic-foundry)
  - [Development](#development)
    - [Development environment](#development-environment)
    - [IDE configuration](#ide-configuration)
    - [Make on Windows](#make-on-windows)
    - [Run](#run)
    - [Lint](#lint)
    - [Test](#test)
    - [Clean](#clean)

---

# Overview

LLM Adapters unify the APIs of respective LLMs to align with the Unified Protocol of DIAL Core. Each Adapter operates within a dedicated container. Multi-modality allows supporting non-textual communications such as image-to-text, text-to-image, file transfers and more.

The project implements [AI DIAL API](https://dialx.ai/dial_api) for language models and embedding models from [Vertex AI](https://console.cloud.google.com/vertex-ai).

![ai-dial-core](https://docs.dialx.ai/assets/images/adapters-62587fb74cfb1c4225c20c08273ec5bc.svg)

---

## Supported models

### Chat completion models

The following models support `POST $SERVER_ORIGIN/openai/deployments/$MODEL_ID/chat/completions` endpoint along with an optional support of the feature endpoints:

- `POST $SERVER_ORIGIN/openai/deployments/$MODEL_ID/tokenize`
- `POST $SERVER_ORIGIN/openai/deployments/$MODEL_ID/truncate_prompt`
- `POST $SERVER_ORIGIN/openai/deployments/$MODEL_ID/configuration`

|Model|Model ID|Modality|`/tokenize`|`/truncate_prompt`|tools/functions support|`/configuration`|
|---|---|---|---|---|---|---|
|Gemini 3 Pro|gemini-3-pro-preview|(text/pdf/image/audio/video)-to-text|✅|✅|✅|✅|
|Gemini 3 Pro Image|gemini-3-pro-image-preview|(text/image)-to-(text/image)|✅|✅|✅|✅|
|Gemini 2.5 Flash|gemini-2.5-flash|(text/pdf/image/audio/video)-to-text|✅|✅|✅|✅|
|Gemini 2.5 Flash Image|gemini-2.5-flash-image|(text/image)-to-(text/image)|✅|✅|✅|✅|
|Gemini 2.5 Pro|gemini-2.5-pro|(text/pdf/image/audio/video)-to-text|✅|✅|✅|✅|
|Gemini 2.0 Flash Lite|gemini-2.0-flash-lite-001|(text/pdf/image/audio/video)-to-text|✅|✅|✅|❌|
|Gemini 2.0 Flash|gemini-2.0-flash-exp|(text/pdf/image/audio/video)-to-(text/image)|✅|✅|✅|❌|
|Gemini 2.0 Flash|gemini-2.0-flash-001|(text/pdf/image/audio/video)-to-text|✅|✅|✅|❌|
|Claude 4.5 Sonnet|claude-sonnet-4-5@20250929|(pdf/text/image)-to-text|✅|✅|✅|✅|
|Claude 4.5 Haiku|claude-haiku-4-5@20251001|(pdf/text/image)-to-text|✅|✅|✅|✅|
|Claude 4.1 Opus|claude-opus-4-1@20250805|(pdf/text/image)-to-text|✅|✅|✅|✅|
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
|Veo 3.1 Fast Generate|veo-3.1-fast-generate-(001\|preview)|text-to-video|✅|✅|❌|✅|
|Veo 3.1 Generate|veo-3.1-generate-(001\|preview)|text-to-video|✅|✅|❌|✅|
|Veo 3.0 Fast Generate|veo-3.0-fast-generate-(001\|preview)|text-to-video|✅|✅|❌|✅|
|Veo 3.0 Generate|veo-3.0-generate-(001\|preview)|text-to-video|✅|✅|❌|✅|

The models that support `/truncate_prompt` do also support `max_prompt_tokens` chat completion request parameter.

#### Image editing in Gemini 2.5 Flash Image

Gemini 2.5 Flash Image and Gemini 3 Pro Image models support both image generation and image editing.
This enables the following use case:

```txt
user: generate an image of a cat sitting on a sofa
assistant: <image attachment #1>
user: replace the cat with a dog
assistant: <image attachment #2>
```

This scenario works out of the box with API integrations.

However, it wouldn't work for interactions via [DIAL Chat](https://github.com/epam/ai-dial-chat/tree/0.40.0), since it removes all attachments from the assistant messages by default. To change this default behavior, set the `assistantAttachmentsInRequestSupported` flag to `true` in the [DIAL Core configuration](https://github.com/epam/ai-dial-core/blob/0.37.0/docs/dynamic-settings/models.md#modelsmodel_namefeatures) for the Gemini deployment in question:

```json
{
  "models": {
    "dial-gemini-deployment-id": {
      "type": "chat",
      "endpoint": "${VERTEXAI_ADAPTER_ORIGIN}/openai/deployments/gemini-2.5-flash-image/chat/completions",
      "features": {
        "assistantAttachmentsInRequestSupported": true
      }
    }
  }
}
```

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

##### Veo models

The Veo models support configuration of parameters specific for video-generation such as aspect ratio, compression quality and duration seconds. See the complete list of configurable parameters at the [Veo API documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/veo-video-generation#parameters).

```json
{
  "messages": [{"role": "user", "content": "forest meadow"}],
  "custom_fields": {
    "configuration": {
      "aspect_ratio": "16:9",
      "compression_quality": "optimized",
      "duration_seconds": 4
    }
  }
}
```

##### Gemini 2.5, Gemini 3 models

The Gemini 2.5 and Gemini 3 series models support configuration of the [thinking parameters](https://ai.google.dev/gemini-api/docs/thinking):

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

The thought summaries are printed into a dedicated `Thinking` stage given that `include_thoughts` is set to `true`.

The token budget for thinking may be set to be unlimited via `thinking_budget=-1`.

The Gemini 3 series model also supports [thinking_level](https://ai.google.dev/gemini-api/docs/gemini-3?thinking=low#thinking_level) parameter that could be set via the [reasoning_effort](https://platform.openai.com/docs/api-reference/chat/create#chat_create-reasoning_effort) field from the OpenAI API:

```json
{
  "messages": [{"role": "user", "content": "Explain quantum computing in simple terms."}],
  "reasoning_effort": "none|minimum|low|medium|high"
}
```

> [!NOTE]
> You cannot use both `reasoning_effort` and the `thinking_budget` parameters in the same request.

##### Gemini 2.5 Flash Image model

The Gemini 2.5 Flash Image support configuration of parameters controlling generation of images:

```json
{
  "custom_fields": {
    "configuration": {
      "image_config": {
        "aspect_ratio": "21:9",
        "image_size": "4K"
      }
    }
  }
}
```

Consult the [documentation](https://ai.google.dev/gemini-api/docs/image-generation#aspect_ratios_and_image_size) for the possible values of these parameters and their defaults.

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

---

### Embedding models

The following models support `$SERVER_ORIGIN/openai/deployments/$MODEL_ID/embeddings` endpoint:

|Model|Model ID|Language support|Modality|
|---|---|---|---|
|Gemini Embeddings|gemini-embedding-001|Multilingual|text-to-embedding|
|Gecko Embeddings for Text V1|textembedding-gecko@001|English|text-to-embedding|
|Gecko Embeddings for Text V3|textembedding-gecko@003|English|text-to-embedding|
|Embeddings for Text|text-embedding-(004\|005)|English|text-to-embedding|
|Gecko Embeddings for Text Multilingual|textembedding-gecko-multilingual@001|Multilingual|text-to-embedding|
|Embeddings for Text Multilingual|text-multilingual-embedding-002|Multilingual|text-to-embedding|
|Multimodal embeddings|multimodalembedding@001|English|(text/image)-to-embedding|

---

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
|COMPATIBILITY_MAPPING|{}|**Deprecated** in favour of [compatibility configuration in DIAL Core config](#compatibility-configuration-in-dial-core-config). A JSON dictionary that maps VertexAI deployments that **aren't supported** by the Adapter to the VertexAI deployments that **are supported** by the Adapter _(see the [Supported models](#supported-models)_ section). Find more details in the [compatibility mode](#compatibility-configuration-in-adapter) section.|
|CLAUDE_DEFAULT_MAX_TOKENS|1536|The default value of `max_tokens` chat completion parameter if it is not provided in the request.<br>**:warning: Using the variable is discouraged**.<br>Consider configuring the default in the DIAL Core Config instead as demonstrated in the [example below](#default-max_tokens-for-claude-models).|
|GOOGLE_GENAI_MAX_RETRY_ATTEMPTS|0|How many times to retry Google GenAI chat model requests when the provider returns a retriable error|
|ANTHROPIC_MAX_RETRY_ATTEMPTS|0|How many times to retry Anthropic chat model requests when the provider returns a retriable error|

### Default `max_tokens` for Claude models

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

The Adapter supports a predefined list of VertexAI deployments. The [Supported models](#supported-models) section lists the models. These models could be accessed via `/openai/deployments/$MODEL_ID/(chat_completions|embeddings)` endpoints. The Adapter won't recognize any other deployment name and will result in `404` error.

Now, suppose VertexAI has just released a new version of a model, e.g. `gemini-2.0-flash-006` that is a better version of an older `gemini-2.0-flash-001` model.

Immediately after the release, the former model is **unsupported** by the Adapter, but the latter is **supported**.
Therefore, the request to `openai/deployments/gemini-2.0-flash-006/chat/completions` will result in 404 error.

It will take some time for the Adapter to catch up with VertexAI - support the v6 model and publish the release with the fix.

What to do in the meantime? Presumably, the v6 model is backward compatible with v1, so we may try to run v6 in **the compatibility mode** - that is to convince the Adapter to process v6 request as if it's v1 request with the only difference that the final upstream request to GCP VertexAI will be to v6 and not v1.

There are two way to enable compatibility mode in the adapter.

### Compatibility configuration in DIAL Core config

It's possible to define compatible model on per-upstream basis in the DIAL Core configuration.

E.g. the following configuration enables `gemini-2.0-flash-006` model _(a hypothetical model that isn't supported by the Adapter natively)_ via `gemini-2.0-flash-001` model _(that is supported by the Adapter natively)_:

```json
{
  "models": {
    "dial-deployment-id-for-claude-3-5": {
      "type": "chat",
      "endpoint": "${ADAPTER_ORIGIN}/deployments/gemini-2.0-flash-006/chat/completions",
      "upstreams": [
        {
          "extraData": {
            "compatible_model_id": "gemini-2.0-flash-001"
          }
        }
      ]
    }
  }
}
```

The given configuration enables the adapter to handle requests to the `gemini-2.0-flash-006` deployment.
The requests will be processed by the same pipeline as `gemini-2.0-flash-001`, but the call to GCP VertexAI will be done to `gemini-2.0-flash-006` deployment name.

Naturally, this will only work if the APIs of v1 and v6 deployments are compatible:

1. The requests utilizing the modalities supported by both v1 and v6 will work just fine.
2. However, the requests with modalities that are supported by v6 and aren't supported by v1, won't be processed correctly. You will have to wait until the adapter supports the v6 deployment natively.

When a version of the adapter supporting the v6 model is released, you may migrate to it and safely remove the `compatible_model_id` from the DIAL Core config.

Note that setting `compatible_model_id=imagen-4.0-generate-001` will be ineffectual, since the APIs of the two model and their capabilities are drastically different.

> [!IMPORTANT]
> If the DIAL deployment has many upstreams, the `compatible_model_id` field should be set in all of the upstreams.

### Compatibility configuration in Adapter

> [!IMPORTANT]
> Model compatibility configuration via `COMPATIBILITY_MAPPING` env variable has been deprecated in favour of [configuration on the DIAL Core side](#compatibility-configuration-in-dial-core-config) in DIAL Core config.

`COMPATIBILITY_MAPPING` env variable enables compatibility mode on the adapter level.
It hold a mapping from unsupported deployment ids to supported deployment ids.

E.g. the following mapping enables `gemini-2.0-flash-006` via `gemini-2.0-flash-001`:

```ini
COMPATIBILITY_MAPPING={"gemini-2.0-flash-006": "gemini-2.0-flash-001"}
```

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
      "endpoint": "${VERTEXAI_ADAPTER_ORIGIN}/openai/deployments/gemini-2.5-flash/chat/completions",
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
    "gemini-dial-deployment-id": {
      "endpoint": "${ADAPTER_ORIGIN}/deployments/gemini-2.0-flash-lite-001/chat/completions",
      "upstreams": [
        {
          "key": "gemini-api-key"
        }
      ]
    },
    "claude-dial-deployment-id": {
      "endpoint": "${ADAPTER_ORIGIN}/deployments/claude-3-5-sonnet-20241022/chat/completions",
      "upstreams": [
        {
          "key": "anthropic-api-key",
          "extraData": {
            "compatible_model_id": "claude-3-5-sonnet-v2@20241022"
          }
        }
      ]
    }
  }
}
```

Keep in mind that the same Anthropic models have [different identifiers](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) in Anthropic API and GPC Vertex AI.

E.g. `claude-3-5-sonnet-v2@20241022` in GCP Vertex AI corresponds to `claude-3-5-sonnet-20241022` in Anthropic API.

The VertexAI adapter uses model names from **GCP Vertex AI**. Therefore, in order to use **Anthropic API** model name you need to specify the corresponding name from **GCP Vertex AI** in the [compatible_model_id](#compatibility-configuration-in-dial-core-config) field. Otherwise, the adapter returns 404.

### Anthropic Foundry

The adapter supports access to Claude models from [Azure](https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-models/how-to/use-foundry-models-claude?view=foundry&preserve-view=true&tabs=python) [Foundry](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry) service.

```json
{
  "models": {
    "claude-dial-deployment-id": {
      "endpoint": "${VERTEXAI_ADAPTER_ORIGIN}/openai/deployments/claude-sonnet-4-520250929/chat/completions",
      "upstreams": [
        {
          "endpoint": "https://${AZURE_FOUNDRY_RESOURCE_NAME1}.services.ai.azure.com/anthropic/v1/messages",
          "key": "optional-azure-foundry-api-key1"
        },
        {
          "endpoint": "https://${AZURE_FOUNDRY_RESOURCE_NAME2}.services.ai.azure.com/anthropic/v1/messages",
          "key": "optional-azure-foundry-api-key2"
        }
      ]
    }
  }
}
```

The [DefaultAzureCredential](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) is used to authenticate requests to Azure when an API key is not provided in the upstream configuration.

Since the models names in Azure Foundry are different from the ones in GCP Vertex AI, you need to map them onto supported [deployment names](#supported-models) using the [compatibility mapping](#compatibility-mode):

```txt
COMPATIBILITY_MAPPING={"claude-sonnet-4-520250929":"claude-sonnet-4-5@20250929"}
```

---

## Development

### Development environment

This project uses [Python>=3.11](https://www.python.org/downloads/) and [Poetry>=2.1.1](https://python-poetry.org/) as a dependency manager.

Check out Poetry's [documentation on how to install it](https://python-poetry.org/docs/#installation) on your system before proceeding.

To install requirements:

```sh
poetry install
```

This will install all requirements for running the package, linting, formatting and tests.

### IDE configuration

The recommended IDE is [VS Code](https://code.visualstudio.com/).
Open the project in VS Code and install the recommended extensions.
VS Code is configured to use PEP-8 compatible formatter [Black](https://black.readthedocs.io/en/stable/index.html).

Alternatively you can use [PyCharm](https://www.jetbrains.com/pycharm/).
Set up the Black in PyCharm [manually](https://black.readthedocs.io/en/stable/integrations/editors.html#pycharm-intellij-idea) or
install PyCharm>=2023.2 with [built-in Black support](https://blog.jetbrains.com/pycharm/2023/07/2023-2/#black).

### Make on Windows

As of now, Windows distributions do not include the make tool. To run make commands, the tool can be installed using
the following command (since [Windows 10](https://learn.microsoft.com/en-us/windows/package-manager/winget/)):

```sh
winget install GnuWin32.Make
```

For convenience, the tool folder can be added to the PATH environment variable as `C:\Program Files (x86)\GnuWin32\bin`.
The command definitions inside Makefile should be cross-platform to keep the development environment setup simple.

### Run

Run the development server locally:

```sh
make serve
```

Run the server from a Docker container:

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
