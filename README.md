# Document Assistant — Prototype

A small, deployable RAG chatbot built to answer Kezhan's brief: prove domain
knowledge, prove I can ship something a business user could open in a
browser, and fold in the multilingual retrieval question from our
conversation instead of just arguing it in a message.

## What it does

Two synthetic insurance-policy documents, one in French and one in English,
same structure, different wording. Ask a question in either language and
the pipeline retrieves the right section from either document and answers
with a typed contract: value, citations, confidence, and — critically — an
honest `answer_found` / `complete_answer_found` flag instead of a single
fuzzy confidence number.

## Tools used and why

| Piece | Tool | Why this one |
|---|---|---|
| Backend | FastAPI + Uvicorn | matches the stack Kezhan suggested, minimal boilerplate |
| Embeddings | `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) | small, free, local, genuinely multilingual (FR/EN/DE), no API cost for retrieval |
| Vector search | plain numpy cosine similarity | corpus is a handful of documents — a vector DB would be solving a problem I don't have yet (see "left out") |
| Keyword signal | hand-written per-language dictionary | deliberately a bonus, not a pillar — this is the fragile signal from the multilingual conversation |
| Structure signal | heading/article-number regex | language-invariant, weighted highest alongside embeddings |
| Generation | Groq API (`openai/gpt-oss-120b`), JSON-mode | free tier, no credit card, fast, strong enough for schema-constrained extraction per Pattern 6 |
| Validation | Pydantic (`AnswerContract`) | the model's output is parsed and typed, never trusted as raw text |
| Frontend | plain HTML/CSS/JS served by FastAPI | one service to deploy, no build step, no framework overhead for a prototype this size |
| Deployment | Docker on Hugging Face Spaces or Render free tier | one container, both platforms just need the Dockerfile |

## Exact build map (the order I actually built it in)

1. **Define the corpus.** Two synthetic policy documents, FR and EN, same six
   sections each, deliberately worded differently (`franchise` vs.
   `deductible`) so a naive keyword-only system would fail on one language.
2. **Typed contract first** (`schemas.py`). Decide the shape of a correct
   answer before writing any retrieval or generation code — this is the
   contract everything else has to satisfy.
3. **Parsing + chunking** (`rag_pipeline.py::parse_and_chunk`). Split on
   headings/article numbers first, so structure is captured as metadata
   instead of being flattened away.
4. **Embedding index** (`DocumentIndex`). Multilingual sentence-transformer,
   rebuilt in memory on startup and on every upload.
5. **Hybrid retrieval** (`DocumentIndex.search`). Weighted sum of embedding
   similarity, structure match, and keyword bonus — the weights encode the
   answer Kezhan gave: structure and embeddings dominate, keyword is a
   discount-weighted extra signal, and it's cross-language aware (FR query
   can still get a keyword bonus on an EN chunk, at a discount).
6. **Generation** (`generate_answer`). JSON-only prompt against Groq's free
   API (`openai/gpt-oss-120b`) with JSON mode forced, defensive parsing,
   fallback to `answer_found=False` with a caveat if the model's output
   doesn't validate against the schema, rather than surfacing a broken
   answer. JSON mode matters more here than it would with a frontier
   model: open models are weaker at spontaneous schema adherence, so the
   explicit `response_format` and the tightened prompt wording do real
   work.
7. **FastAPI wiring** (`main.py`). Endpoints for `/api/ask`,
   `/api/upload`, `/api/health`, `/api/metrics`, plus the static frontend
   mounted at `/`. Rate limiting and structured JSON logging added here
   too (see "Production hardening" below).
8. **Frontend** (`frontend/index.html`). Single page, no framework, shows
   the status flag and citations explicitly rather than hiding them —
   the point of the typed contract is lost if the UI just prints a string.
9. **Tests** (`tests/test_rag_pipeline.py`). Unit tests for chunking,
   language detection, and the hybrid scoring math — deliberately
   model-free so they run in seconds and can gate every push in CI.
10. **Containerize** (`Dockerfile`). One image, works on HF Spaces (port
    7860) and Render (`$PORT`) without changes.
11. **CI/CD** (`.github/workflows/ci.yml`). Lint + test on every push,
    Docker image built and published to GHCR on merge to `main`.
12. **Infra-as-code** (`render.yaml`). Declarative deploy config instead
    of manual dashboard clicks.
13. **Deploy.**

## Production hardening

A few things added once the happy-path prototype worked, specifically
because this sits on a free tier and is meant to be looked at by someone
evaluating engineering judgment, not just the RAG logic:

- **Rate limiting** (`slowapi`): `/api/ask` capped at 10 requests/minute
  per IP, `/api/upload` at 5/minute. The free Groq quota and the local
  embedding compute are both finite; an unrestricted public endpoint is
  the fastest way to burn through either.
- **Structured logging**: every `/api/ask` call logs a single JSON line
  with question length, detected language, the answer/completeness
  flags, confidence, top retrieval score, and latency. No user text is
  logged verbatim, only its length, so this is safe to ship to a log
  aggregator without becoming a privacy problem.
- **`/api/metrics`**: a lightweight in-memory counter endpoint (request
  count, error count, answer-found rate, average latency). Not
  Prometheus, not a paid observability service, just enough to see if
  the deployed instance is healthy without SSHing in.

## Engineering practices (CI/CD, testing, infra-as-code)

This isn't just a Jupyter notebook wrapped in an API. The repo is set up
the way I'd want a teammate to hand it to me:

- **Automated tests** (`tests/`, `pytest`) cover chunking, language
  detection, and the hybrid retrieval scoring — the parts of the pipeline
  most likely to silently regress. Model-loading is deliberately excluded
  from the fast test suite so CI stays quick.
- **Linting** (`ruff`, `pyproject.toml`) plus a **pre-commit hook**
  (`.pre-commit-config.yaml`) so style issues get caught before a commit,
  not in review.
- **CI pipeline** (`.github/workflows/ci.yml`, GitHub Actions, free for
  public repos): every push and PR runs lint + tests; every push to
  `main` that passes also builds the Docker image and publishes it to
  GitHub Container Registry (`ghcr.io`), tagged with both `latest` and
  the commit SHA — a reproducible, versioned artifact, not a folder that
  "works on my machine."
- **Infrastructure as code** (`render.yaml`): the Render service
  definition — runtime, health check path, autodeploy — lives in the
  repo as a Blueprint, not as manual settings in a dashboard someone else
  can't see or reproduce.
- **`Makefile`**: `make install`, `make lint`, `make test`, `make run` —
  one command per common task instead of remembering flag combinations.

## Running locally

```bash
cd doc-chatbot
make install                  # deps + ruff/pytest/pre-commit
export GROQ_API_KEY=gsk_...   # free key from https://console.groq.com
make run
# open http://localhost:8000
```

Getting a free Groq key: sign up at console.groq.com (no credit card), go to
API Keys, create one. The free tier's rate limits are generous enough for
demoing this project comfortably.

## Getting this into GitHub (needed for CI/CD)

```bash
cd doc-chatbot
git init
git add .
git commit -m "Document assistant prototype"
gh repo create doc-chatbot --public --source=. --push
# no gh CLI? create an empty repo on github.com, then:
# git remote add origin https://github.com/<you>/doc-chatbot.git
# git branch -M main && git push -u origin main
```

Once pushed, the Actions tab will show lint + tests running on the push,
and on `main` a Docker image publishing to
`ghcr.io/<you>/doc-chatbot:latest` — public and pullable by anyone,
including Kezhan, without him needing to build it himself.

## Deploying (Render, free tier, infra-as-code)

1. New → Blueprint → connect the GitHub repo. Render reads `render.yaml`
   automatically and creates the service from it.
2. Set `GROQ_API_KEY` in the dashboard when prompted (kept out of the repo
   on purpose — `render.yaml` marks it `sync: false`).
3. Every push to `main` that passes CI redeploys automatically.

## Deploying (Hugging Face Spaces, free tier, alternative)

1. Create a new Space, SDK = Docker.
2. Push this folder's contents to the Space repo (the `Dockerfile` at root
   is picked up automatically), or point the Space at the GHCR image built
   by CI instead of rebuilding from source.
3. In Space settings, add `GROQ_API_KEY` as a secret.
4. Space builds and serves on port 7860 automatically.

## Cost

Everything here is free: Groq's API free tier for generation, the
embedding model runs locally, GitHub Actions is free for public repos,
GHCR is free for public images, and Hugging Face Spaces / Render both
have free hosting tiers. The one thing to watch is Groq's rate limits if
the deployed instance gets hammered with requests — the in-app rate
limiter (10 req/min per IP on `/api/ask`) is there specifically to keep
that from happening.

## What I left out, on purpose

- **No real vector database** (FAISS/Chroma/pgvector). Numpy cosine over a
  few hundred chunks is fast enough for a demo and keeps the deploy
  footprint tiny. This is the first thing to swap once the corpus stops
  fitting comfortably in memory.
- **No reranker model.** The hybrid score (embedding + structure + keyword)
  does the job of a reranker here because the corpus is small and the
  domain vocabulary is known. A reranker earns its latency once the corpus
  is large enough that top-k candidates are noisier.
- **No persistent storage.** The index rebuilds from the `data/sample_docs`
  folder on every restart. Fine for a demo, wrong for anything with
  uploaded documents that need to survive a redeploy.
- **No auth, no per-user session, no persistent metrics.** Rate limiting
  is in place, but anyone with the link can still query it, and
  `/api/metrics` resets on every restart since it's in-memory. Fine for a
  synthetic demo corpus, not for anything with real documents behind it.
- **No evaluation harness.** I eyeballed the FR/EN answers rather than
  scoring retrieval rank systematically, unlike the measurement approach
  in the retrieval-failures article. Worth doing before trusting this on
  a real corpus, and a natural next test to add to CI.
- **Language detection is a stopword heuristic, not `langdetect` or
  `fasttext`.** Good enough for two languages that don't share much
  vocabulary; would misfire on closer language pairs or short queries.
- **Generation runs on a free open model (`openai/gpt-oss-120b` via
  Groq), not a frontier model.** JSON mode plus a tightened prompt closes
  most of the gap, but an open model is still more likely to drift
  off-schema on edge cases (ambiguous questions, mixed-language context)
  than Claude or GPT-4-class models would be. Worth stress-testing before
  trusting this on real documents rather than the synthetic demo corpus.

## What would break first at real volume

The **in-memory numpy index rebuilt on every startup** is the first wall.
At a few hundred chunks it's instant; past a few tens of thousands of
chunks, both the encode-on-startup cost and the O(n) cosine scan per query
become the bottleneck — that's the point to move to FAISS or a hosted
vector store with an on-disk/persistent index and incremental updates
instead of a full rebuild.

The **per-language keyword dictionary** is the second wall, and it's the
one from our conversation: it's hand-maintained and language-specific, so
it doesn't scale to a fifth or sixth language without someone curating
terms for it. At real volume this should either be generated from a
domain glossary automatically, or the weighting should shift further
toward structure and embeddings and treat keyword as an even smaller
bonus than it already is.
