"""
Unit tests for the pieces of the pipeline that don't require downloading
the embedding model. Kept model-free on purpose so CI runs in seconds,
not minutes: retrieval scoring math, chunking, and the typed contract are
all things we can verify deterministically. The full DocumentIndex
(which loads sentence-transformers) is exercised manually / in a slower
integration test, not on every push.
"""
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from rag_pipeline import DocumentIndex, detect_language, parse_and_chunk  # noqa: E402
from schemas import AnswerContract, Chunk, Citation  # noqa: E402


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def test_detect_language_french():
    assert detect_language("Quel est le délai de déclaration d'un sinistre ?") == "fr"


def test_detect_language_english():
    assert detect_language("What is the claim reporting deadline?") == "en"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def test_parse_and_chunk_splits_on_headings():
    text = (
        "Article 1\n"
        "First section body text.\n"
        "Article 2\n"
        "Second section body text.\n"
    )
    chunks = parse_and_chunk("doc1", text)
    assert len(chunks) == 2
    assert chunks[0].section == "Article 1"
    assert chunks[1].section == "Article 2"
    assert "First section" in chunks[0].text
    assert "Second section" in chunks[1].text


def test_parse_and_chunk_wraps_long_sections():
    long_body = "word " * 500  # far beyond CHUNK_MAX_CHARS
    text = f"Article 1\n{long_body}"
    chunks = parse_and_chunk("doc1", text)
    assert len(chunks) > 1
    assert all(c.section == "Article 1" for c in chunks)


def test_parse_and_chunk_assigns_language_per_chunk():
    text = "Article 1\nLa franchise est fixee a 250 euros."
    chunks = parse_and_chunk("doc1", text)
    assert chunks[0].language == "fr"


# ---------------------------------------------------------------------------
# Hybrid scoring (keyword + structure), no embedding model needed
# ---------------------------------------------------------------------------

def test_keyword_score_same_language_hit():
    chunk = Chunk(doc_id="d", chunk_id="c1", text="La franchise est de 250 euros.",
                  section="Article 2", language="fr")
    score = DocumentIndex._keyword_score("quelle est la franchise ?", chunk, "fr")
    assert score > 0.0


def test_keyword_score_cross_language_is_discounted():
    fr_chunk = Chunk(doc_id="d", chunk_id="c1", text="La franchise est de 250 euros.",
                      section="Article 2", language="fr")
    en_chunk = Chunk(doc_id="d", chunk_id="c2", text="The deductible is 300 dollars.",
                      section="Section 2", language="en")

    same_lang_score = DocumentIndex._keyword_score("what is the deductible?", en_chunk, "en")
    cross_lang_score = DocumentIndex._keyword_score("what is the deductible?", fr_chunk, "en")

    assert same_lang_score > cross_lang_score


def test_keyword_score_no_match_returns_zero():
    chunk = Chunk(doc_id="d", chunk_id="c1", text="Coverage applies to fire damage.",
                  section="Section 4", language="en")
    score = DocumentIndex._keyword_score("what is the weather today?", chunk, "en")
    assert score == 0.0


def test_structure_score_rewards_heading_overlap():
    chunk = Chunk(doc_id="d", chunk_id="c1", text="irrelevant body",
                  section="Article 5 Resiliation", language="fr")
    high = DocumentIndex._structure_score("comment fonctionne la resiliation ?", chunk)
    low = DocumentIndex._structure_score("quelle est la franchise ?", chunk)
    assert high > low


def test_structure_score_zero_without_section():
    chunk = Chunk(doc_id="d", chunk_id="c1", text="irrelevant", section=None, language="en")
    assert DocumentIndex._structure_score("any question", chunk) == 0.0


# ---------------------------------------------------------------------------
# Typed contract
# ---------------------------------------------------------------------------

def test_answer_contract_requires_two_booleans():
    contract = AnswerContract(answer_found=True, complete_answer_found=False, confidence=0.6)
    assert contract.answer_found is True
    assert contract.complete_answer_found is False


def test_answer_contract_rejects_confidence_out_of_range():
    with pytest.raises(ValidationError):
        AnswerContract(answer_found=True, complete_answer_found=True, confidence=1.5)


def test_citation_requires_quote():
    with pytest.raises(ValidationError):
        Citation(doc_id="d", chunk_id="c1")  # missing required `quote`
