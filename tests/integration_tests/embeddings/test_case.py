from dataclasses import dataclass, field
from typing import Any, List

import openai
from openai.types import CreateEmbeddingResponse

from tests.integration_tests.embeddings.test_embeddings import ModelSpec
from tests.utils.exception import ExpectedException, expected_exception


@dataclass
class EmbeddingsTestCase:
    __test__ = False

    spec: ModelSpec
    input: str | List[str] = field(default_factory=list)
    custom_input: List[Any] | None = None
    encoding_format: str | None = None
    embedding_type: str | None = None
    embedding_instr: str | None = None
    dimensions: int | None = None

    @property
    def extra_body(self) -> dict:
        custom_fields = {}
        if self.embedding_instr:
            custom_fields["instruction"] = self.embedding_instr
        if self.embedding_type:
            custom_fields["type"] = self.embedding_type

        body = {
            "custom_input": self.custom_input,
            "custom_fields": custom_fields,
            "dimensions": self.dimensions,
        }
        if self.encoding_format is not None:
            body["encoding_format"] = self.encoding_format
        return body

    async def run(
        self,
        client: openai.AsyncAzureOpenAI,
        error: ExpectedException | None = None,
    ):
        async def _call() -> CreateEmbeddingResponse:
            model_id = self.spec.deployment.value
            return await client.embeddings.create(
                model=model_id, input=self.input, extra_body=self.extra_body
            )

        if error is None:
            self._check_embedding_response(await _call())
        else:
            async with expected_exception(error):
                await _call()

    def _check_embedding_response(self, resp: CreateEmbeddingResponse) -> None:
        n_inputs = 1 if isinstance(self.input, str) else len(self.input)
        n_inputs += len(self.custom_input) if self.custom_input else 0
        dimensions = self.dimensions or self.spec.default_dimensions

        assert len(resp.data) == n_inputs
        assert len(resp.data[0].embedding) == dimensions
