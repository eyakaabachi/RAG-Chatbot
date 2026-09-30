
from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

import numpy as np

from schemas import AnswerContract, Chunk, RetrievedChunk



EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # FR/EN/DE/LU-adjacent
HF_EMBEDDING_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL_NAME}"
CHUNK_MAX_CHARS = 700
TOP_K = 4


KEYWORD_DICTIONARIES = {
    "fr": {
        "franchise": "deductible",
        "sinistre": "claim",
        "garantie": "coverage",
        "resiliation": "termination",
        "delai": "deadline",
        "indemnisation": "compensation",
        "vol": "theft",
        "voler": "theft",
        "cambriolage": "theft",
        "effraction": "theft",
        "incendie": "fire",
        "degat des eaux": "water_damage",
        "catastrophe naturelle": "natural_disaster",
    },
    "en": {
        "deductible": "deductible",
        "claim": "claim",
        "coverage": "coverage",
        "termination": "termination",
        "deadline": "deadline",
        "compensation": "compensation",
        "theft": "theft",
        "stolen": "theft",
        "steal": "theft",
        "burglary": "theft",
        "fire": "fire",
        "water damage": "water_damage",
        "natural disaster": "natural_disaster",
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




def embed_texts(texts: list[str]) -> np.ndarray:
    """Call HF's Inference API for sentence embeddings. Returns an
    (n_texts, dim) array, L2-normalized so a dot product is cosine
    similarity."""
    import requests

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN not set. Get a free token at https://huggingface.co/settings/tokens "
            "(read scope is enough)."
        )

    resp = requests.post(
        HF_EMBEDDING_URL,
        headers={"Authorization": f"Bearer {hf_token}"},
        json={"inputs": texts, "options": {"wait_for_model": True}},
        timeout=60,
    )
    resp.raise_for_status()
    data = np.array(resp.json(), dtype=np.float32)


    if data.ndim == 3:
        data = data.mean(axis=1)

    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return data / norms



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




class DocumentIndex:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def add_document(self, doc_id: str, raw_text: str):
        new_chunks = parse_and_chunk(doc_id, raw_text)
        self.chunks.extend(new_chunks)
        self._rebuild_embeddings()

    def _rebuild_embeddings(self):
        if not self.chunks:
            self.embeddings = None
            return
        texts = [c.text for c in self.chunks]
        self.embeddings = embed_texts(texts)

    def load_folder(self, folder: str):
        for path in Path(folder).glob("*.txt"):
            self.add_document(path.stem, path.read_text(encoding="utf-8"))

    
    def search(self, query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query_lang = detect_language(query)
        query_emb = embed_texts([query])[0]
        cosine_scores = self.embeddings @ query_emb  

        results: list[RetrievedChunk] = []
        for chunk, emb_score in zip(self.chunks, cosine_scores, strict=True):
            keyword_score = self._keyword_score(query, chunk, query_lang)
            structure_score = self._structure_score(query, chunk)

            
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

If different documents in the context give different or conflicting
answers (for example, different numbers for the same question), set
answer_found=true and complete_answer_found=false, cite each conflicting
passage separately, and — this matters — still write a `value` that
summarizes the conflict in one sentence (for example: "The reporting
deadline differs by document: 7 business days in the English policy vs.
5 working days in the French policy."). NEVER leave `value` null when
answer_found is true, even if the full answer needs the caveat and
citations to be understood completely.

Do not infer that a specific item or scenario is covered or excluded
just because it seems to fall under a broader category named in the
text, if the text itself does not make that link explicit. A named
category can carry conditions (a location, a method, a cause) that the
question's specific case may or may not satisfy — treat those
conditions as load-bearing, not as incidental wording.

Worked example of this exact trap: context says "coverage applies to
theft following forced entry into the premises." Question: "my phone
was stolen from a cafe table, am I covered?" WRONG: answer_found=true,
value="Yes, theft is covered." (this drops the forced-entry condition
and answers a different, easier question). RIGHT: answer_found=true,
complete_answer_found=false, value="The policy covers theft that
involves forced entry into the insured premises; a phone taken from a
public table doesn't clearly meet that condition, so coverage can't be
confirmed from this text alone.", caveat="The specific scenario
(theft without forced entry, outside the premises) is not directly
addressed.". Apply this same discipline to every question: state
precisely what the text says, then say plainly if the question's
specific case isn't clearly covered by that wording, rather than
resolving the ambiguity in either direction yourself.

Return ONLY a single JSON object, no markdown fences, no commentary before
or after it. Every field below is required, use null only for fields
that are genuinely inapplicable, never for `value` when answer_found is
true:
{{
  "answer_found": true or false,
  "complete_answer_found": true or false,
  "value": "string, required whenever answer_found is true, or null only if answer_found is false",
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


def _fill_missing_value(answer: AnswerContract) -> AnswerContract:
    
    if answer.answer_found and not answer.value:
        if answer.caveat:
            answer.value = answer.caveat
        elif answer.citations:
            answer.value = "See the cited passages below; no single summary was generated."
    return answer


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

    
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        data = json.loads(raw[start:end])
        answer = AnswerContract(**data)
    except (ValueError, json.JSONDecodeError) as e:
        return AnswerContract(
            answer_found=False,
            complete_answer_found=False,
            confidence=0.0,
            caveat=f"Generation contract validation failed: {e}",
        )

    return _fill_missing_value(answer)
