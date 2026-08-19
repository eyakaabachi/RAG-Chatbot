"""
Typed contracts for the RAG pipeline.

Follows the 'seven patterns' article: the LLM never returns a free-text
string. It fills a schema with citations and self-assessment fields,
and the answer is validated before it reaches the user.
"""
from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    section: str | None = None
    quote: str = Field(..., description="Verbatim span the answer is grounded in")


class AnswerContract(BaseModel):
    # Pattern 4: two booleans, not one confidence float
    answer_found: bool
    complete_answer_found: bool

    value: str | None = Field(None, description="The answer itself, typed as text for this prototype")
    citations: list[Citation] = Field(default_factory=list)

    confidence: float = Field(0.0, ge=0.0, le=1.0)
    language_detected: str | None = None
    caveat: str | None = Field(None, description="Known unknown the model could not resolve")


class Chunk(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    section: str | None = None
    language: str | None = None


class RetrievedChunk(BaseModel):
    chunk: Chunk
    embedding_score: float
    keyword_score: float
    structure_score: float
    final_score: float
