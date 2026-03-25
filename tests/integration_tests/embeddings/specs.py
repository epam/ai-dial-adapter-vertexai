from dataclasses import dataclass
from typing import List

from aidial_adapter_vertexai.deployments import EmbeddingsDeployment as D


@dataclass
class ModelSpec:
    deployment: D
    supports_titles: bool
    supports_types: List[str]
    supports_instr: bool
    default_dimensions: int
    supports_dimensions: bool


_BASIC_EMBEDDING_TYPES: List[str] = [
    "CLASSIFICATION",
    "CLUSTERING",
    "RETRIEVAL_DOCUMENT",
    "RETRIEVAL_QUERY",
    "SEMANTIC_SIMILARITY",
    "FACT_VERIFICATION",
    "QUESTION_ANSWERING",
]

EMBEDDING_TYPES = _BASIC_EMBEDDING_TYPES + ["CODE_RETRIEVAL_QUERY"]

MULTI_MODAL_SPEC = ModelSpec(
    deployment=D.MULTI_MODAL_EMBEDDING_1,
    supports_types=[],
    supports_titles=False,
    supports_instr=False,
    default_dimensions=1408,
    supports_dimensions=True,
)

GEMINI_MULTI_MODAL_SPEC = ModelSpec(
    deployment=D.GEMINI_EMBEDDING_2_PREVIEW,
    supports_types=EMBEDDING_TYPES,
    supports_titles=True,
    supports_instr=False,
    default_dimensions=3072,
    supports_dimensions=True,
)

SPECS: List[ModelSpec] = [
    ModelSpec(
        deployment=D.TEXT_GEMINI_EMBEDDING_1,
        supports_types=EMBEDDING_TYPES,
        supports_titles=True,
        supports_instr=False,
        default_dimensions=3072,
        supports_dimensions=True,
    ),
    ModelSpec(
        deployment=D.TEXT_EMBEDDING_4,
        supports_types=_BASIC_EMBEDDING_TYPES,
        supports_titles=True,
        supports_instr=False,
        default_dimensions=768,
        supports_dimensions=True,
    ),
    ModelSpec(
        deployment=D.TEXT_EMBEDDING_5,
        supports_types=EMBEDDING_TYPES,
        supports_titles=True,
        supports_instr=False,
        default_dimensions=768,
        supports_dimensions=True,
    ),
    ModelSpec(
        deployment=D.TEXT_MULTILINGUAL_EMBEDDING_2,
        supports_types=_BASIC_EMBEDDING_TYPES,
        supports_titles=True,
        supports_instr=False,
        default_dimensions=768,
        supports_dimensions=True,
    ),
    ModelSpec(
        deployment=D.GEMINI_EMBEDDING_2_PREVIEW,
        supports_types=EMBEDDING_TYPES,
        supports_titles=True,
        supports_instr=False,
        default_dimensions=3072,
        supports_dimensions=True,
    ),
    MULTI_MODAL_SPEC,
    GEMINI_MULTI_MODAL_SPEC,
]
