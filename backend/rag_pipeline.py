"""
RAG pipeline for the document chatbot prototype.

Design choices, on purpose, tied to the conversation with Kezhan:

1. Retrieval is hybrid and language-aware (his answer to the multilingual
   question): embeddings carry the cross-lingual semantic signal,
   structure (headings, article numbers) is language-invariant and
   weighted highest, keyword matching is a per-language BONUS, never
   the deciding signal.
2. Generation never returns a free string. It fills AnswerContract.
3. Left out on purpose (see README): no vector DB, no reranker model,
   no auth, no chunk-level caching, no async batching. In-memory numpy
   cosine similarity, rebuilt on startup, is enough for a demo corpus
   of a few documents.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

import numpy as np
from huggingface_hub import InferenceClient

from schemas import AnswerContract, Chunk, RetrievedChunk

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_PROVIDER = "hf-inference"
CHUNK_MAX_CHARS = 700
TOP_K = 4

# Minimal per-language keyword dictionaries. This is the "fragile" signal
# Kezhan described: bonus only, and it must be duplicated per language
# rather than assumed to transfer.
KEYWORD_DICTIONARIES = {
    "fr": {
        "franchise": "deductible",
        "sinistre": "claim",
        "garantie": "coverage",
        "resiliation": "termination",
        "delai": "deadline",
        "indemnisation": "compensation",
    },
    "en": {
        "deductible": "deductible",
        "claim": "claim",
        "coverage": "coverage",
        "termination": "termination",
        "deadline": "deadline",
        "compensation": "compensation",
    },
}

FR_STOPWORDS = {"le", "la", "les", "de", "des", "du", "un", "une", "et", "est",
                "que", "qui", "dans", "pour", "sur", "vous", "quel", "quelle",
                "quels", "quelles", "comment", "quand"}
EN_STOPWORDS = {"the", "a", "an", "of", "is", "are", "and", "what", "how",
                "when", "for", "on", "in", "does", "do", "which"}


def detect_language(text: str) -> str:
    """Cheap heuristic language detection, good enough for FR/EN in a demo.
    Swap for `langdetect` or `fasttext` before this touches real volume."""
    words = set(re.findall(r"[a-zàâäéèêëïîôöùûüç]+", text.lower()))
    fr_hits = len(words & FR_STOPWORDS)
    en_hits = len(words & EN_STOPWORDS)
    return "fr" if fr_hits >= en_hits else "en"


# ---------------------------------------------------------------------------
# Ingestion / chunking
# ---------------------------------------------------------------------------

HEADING_RE = re.compile(r"^(Article\s+\d+|ARTICLE\s+\d+|Section\s+\d+|[A-Z][A-Z \-]{6,})\s*$")


def parse_and_chunk(doc_id: str, raw_text: str) -> list[Chunk]:
    """Section-aware chunking: split on headings first (structure signal),
    then hard-wrap long sections so no chunk blows the context budget."""
    lines = raw_text.splitlines()
    sections: list[tuple[str | None, list[str]]] = []
    current_heading, buf = None, []

    for line in lines:
        if HEADING_RE.match(line.strip()):
            if buf:
                sections.append((current_heading, buf))
            current_heading, buf = line.strip(), []
        else:
            buf.append(line)
    if buf:
        sections.append((current_heading, buf))

    chunks: list[Chunk] = []
    for heading, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        # hard-wrap so a chunk never exceeds CHUNK_MAX_CHARS
        for i in range(0, len(body), CHUNK_MAX_CHARS):
            piece = body[i:i + CHUNK_MAX_CHARS].strip()
            if not piece:
                continue
            chunks.append(Chunk(
                doc_id=doc_id,
                chunk_id=str(uuid.uuid4())[:8],
                text=piece,
                section=heading,
                language=detect_language(piece),
            ))
    return chunks


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class DocumentIndex:
    def __init__(self):
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN not set. Create a Hugging Face token with "
                "Inference Providers permission and set it as an environment variable."
            )

        self.embedding_client = InferenceClient(
            provider=HF_PROVIDER,
            api_key=hf_token,
        )
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def _embed(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        # E5 models are trained with task prefixes for asymmetric retrieval.
        prefix = "query: " if is_query else "passage: "
        inputs = [prefix + text for text in texts]

        result = self.embedding_client.feature_extraction(
            inputs,
            model=EMBEDDING_MODEL_NAME,
        )
        embeddings = np.asarray(result, dtype=np.float32)

        # Some inference backends return token-level embeddings. Mean-pool
        # those outputs before normalization; sentence-level outputs pass through.
        if embeddings.ndim == 3:
            embeddings = embeddings.mean(axis=1)

        if embeddings.ndim != 2:
            raise RuntimeError(
                f"Unexpected embedding shape from Hugging Face: {embeddings.shape}"
            )

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-12, None)
        return embeddings

    def add_document(self, doc_id: str, raw_text: str):
        new_chunks = parse_and_chunk(doc_id, raw_text)
        self.chunks.extend(new_chunks)
        self._rebuild_embeddings()

    def _rebuild_embeddings(self):
        if not self.chunks:
            self.embeddings = None
            return
        texts = [c.text for c in self.chunks]
        self.embeddings = self._embed(texts, is_query=False)

    def load_folder(self, folder: str):
        for path in Path(folder).glob("*.txt"):
            self.add_document(path.stem, path.read_text(encoding="utf-8"))

    # -- hybrid retrieval ---------------------------------------------------
    def search(self, query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query_lang = detect_language(query)
        query_emb = self._embed([query], is_query=True)[0]
        cosine_scores = self.embeddings @ query_emb  # (n_chunks,)

        results: list[RetrievedChunk] = []
        for chunk, emb_score in zip(self.chunks, cosine_scores, strict=True):
            keyword_score = self._keyword_score(query, chunk, query_lang)
            structure_score = self._structure_score(query, chunk)

            # Weighting mirrors the multilingual answer: structure and
            # embeddings dominate, keyword is a bonus on top.
            final = (0.55 * float(emb_score)
                     + 0.20 * structure_score
                     + 0.25 * keyword_score)

            results.append(RetrievedChunk(
                chunk=chunk,
                embedding_score=float(emb_score),
                keyword_score=keyword_score,
                structure_score=structure_score,
                final_score=final,
            ))

        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:top_k]

    @staticmethod
    def _keyword_score(query: str, chunk: Chunk, query_lang: str) -> float:
        chunk_lang = chunk.language or query_lang
        # Cross-language keyword hit: query in FR, chunk in EN (or vice versa)
        # still scores via the shared canonical concept, but at a discount,
        # since we can't assume the vocab literally matches.
        dict_query = KEYWORD_DICTIONARIES.get(query_lang, {})
        query_terms = {v for k, v in dict_query.items() if k in query.lower()}
        if not query_terms:
            return 0.0

        dict_chunk = KEYWORD_DICTIONARIES.get(chunk_lang, {})
        chunk_concepts = {v for k, v in dict_chunk.items() if k in chunk.text.lower()}

        hits = len(query_terms & chunk_concepts)
        discount = 1.0 if chunk_lang == query_lang else 0.7
        return min(1.0, hits * 0.5) * discount

    @staticmethod
    def _structure_score(query: str, chunk: Chunk) -> float:
        if not chunk.section:
            return 0.0
        section_words = set(re.findall(r"[a-zàâäéèêëïîôöùûüç]+", chunk.section.lower()))
        query_words = set(re.findall(r"[a-zàâäéèêëïîôöùûüç]+", query.lower()))
        overlap = section_words & query_words
        return min(1.0, len(overlap) * 0.5)


# ---------------------------------------------------------------------------
# Generation (typed contract)
# ---------------------------------------------------------------------------

def build_prompt(query: str, retrieved: list[RetrievedChunk]) -> str:
    context_blocks = []
    for r in retrieved:
        c = r.chunk
        context_blocks.append(
            f"[doc:{c.doc_id} | chunk:{c.chunk_id} | section:{c.section or 'n/a'}]\n{c.text}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    return f"""You are a document question-answering assistant. Answer ONLY from
the context below. If the answer is not in the context, set answer_found=false.
If it is only partially covered, set complete_answer_found=false and explain
the gap in `caveat`. Every citation quote must be copied verbatim from the
context, never paraphrased.

Return ONLY a single JSON object, no markdown fences, no commentary before
or after it. Every field below is required, use null for anything unknown:
{{
  "answer_found": true or false,
  "complete_answer_found": true or false,
  "value": "string or null",
  "citations": [{{"doc_id": "string", "chunk_id": "string", "section": "string or null", "quote": "string"}}],
  "confidence": 0.0 to 1.0,
  "language_detected": "fr or en",
  "caveat": "string or null"
}}

Context:
{context}

Question: {query}

JSON:"""


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"


def call_llm(prompt: str) -> str:
    """Thin wrapper so main.py / tests can mock this out.

    Uses Groq's free tier (OpenAI-compatible endpoint) rather than a paid
    API. No credit card required: https://console.groq.com -> API Keys.
    JSON mode (response_format=json_object) is used to keep schema
    adherence tight, since open models are weaker at this than Claude
    without it.
    """
    import requests

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
            "(no credit card required)."
        )

    resp = requests.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def generate_answer(query: str, retrieved: list[RetrievedChunk]) -> AnswerContract:
    if not retrieved:
        return AnswerContract(
            answer_found=False,
            complete_answer_found=False,
            confidence=0.0,
            caveat="No documents indexed yet.",
        )

    prompt = build_prompt(query, retrieved)
    raw = call_llm(prompt)

    # Defensive parsing: a schema-constrained call can still return prose
    # around the JSON, so extract the object before validating.
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        data = json.loads(raw[start:end])
        return AnswerContract(**data)
    except (ValueError, json.JSONDecodeError) as e:
        return AnswerContract(
            answer_found=False,
            complete_answer_found=False,
            confidence=0.0,
            caveat=f"Generation contract validation failed: {e}",
        )
