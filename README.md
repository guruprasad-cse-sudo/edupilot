# EduPilot — AI Faculty Assistant for OBE Assessment Generation

EduPilot is a multi-user Streamlit web application that helps faculty create
Outcome-Based Education (OBE) assessments — quizzes, internal tests, and
semester exams — complete with Bloom's taxonomy mapping, course-outcome (CO)
alignment, quality review, and polished downloadable documents.

Built for **Dayananda Sagar Academy of Technology and Management**, but the
institution branding is a single constant and easy to change.

## What it does

1. A faculty member signs in and fills out a short form: course name & code,
   department, semester, faculty name, assessment type, difficulty mix,
   duration, marks, topics, and an optional test date.
2. A multi-agent AI pipeline generates the full assessment.
3. The result is shown in the app with analytics (Bloom coverage, marks
   distribution, CO mapping) and an automated quality review.
4. The assessment can be downloaded as **Markdown (.md)**, **Word (.docx)**,
   or **PDF** — formatted with institution branding, test date, semester,
   faculty name, and answer keys.
5. Every generation is saved to the user's personal history for later
   viewing and re-download.

## The AI pipeline

Generation runs through a 6-stage agent pipeline (see `orchestrator.py`):

| Stage | Agent | Role |
|-------|-------|------|
| 1. Planning | `PlanningAgent` | Turns the form input into a structured `AssessmentPlan` |
| 2. Retrieval (RAG) | `RAGModule` | Pulls relevant context from uploaded syllabus/knowledge documents |
| 3. Generation | `AssessmentAgent` | Produces the assessment (questions, marks, Bloom levels, CO mapping) via the Groq LLM |
| 4. Analytics | `AnalyticsAgent` | Computes deterministic metrics: Bloom coverage, difficulty & marks distribution |
| 5. Review | `ReviewerAgent` | LLM-based quality audit — strengths, weaknesses, suggestions |
| 6. Export prep | `DownloadAgent` | Renders the assessment into downloadable documents |

- **LLM provider:** Groq — default model `openai/gpt-oss-120b`
  (override with `GROQ_MODEL_NAME`).
- **Assessment types:** Internal Assessment, Quiz (multiple-choice with
  4 options rendered as radio buttons in-app), Assignment, Semester
  Examination, Viva, Role Play (full classroom scenarios with facilitation
  guides), and Question Bank (descriptive-only reference repository with
  topic-prefixed questions).

### Batched generation for large question banks

Large requests that would exceed the Groq output-token budget are generated
in **multiple balanced LLM calls** and merged automatically:

- Question counts are first allocated **fairly across topics**
  (largest-remainder method, quotas differ by at most 1), then split into
  batches of at most `BATCH_SIZE` questions — so no topic is starved and no
  tiny tail batch is produced (e.g. 25 questions → batches of 9, 8, 8).
- Batches are spaced by a **TPM-aware cooldown** (explained below).
  Rate-limit errors (413/429) still trigger sleeping retries automatically
  as a safety net.
- Merged results are renumbered Q1…Qn, and the UI shows live per-batch
  progress during generation.

#### What "TPM-aware cooldown" means

Groq's free tier has a speed limit: roughly **8000 tokens per minute (TPM)**
per model. The tricky part is that Groq counts the *requested* output budget
(`GROQ_MAX_TOKENS`, 6000 by default) against that limit **the moment a
request starts** — not when the answer comes back, and not based on actual
usage. Two 6000-token requests inside the same minute would exceed 8000 and
the second one is rejected instantly with an HTTP 413 error.

So between batches, the app must wait until about a minute has passed since
the **previous request began**.

- **Naive approach:** sleep a fixed 62 s after every batch finishes. But a
  batch itself takes 30–40 s to generate, and that time already counts
  toward the one-minute window — so this over-waits badly.
- **TPM-aware approach:** the app records the clock time when each batch's
  request *starts*, and before the next batch sleeps only the *remainder*
  of the window. Example: batch 1 starts at 0:00 and finishes at 0:40 →
  only 22 more seconds of waiting are needed, and batch 2 fires at 0:62.

This cut a 25-question bank from ~3.5 minutes to ~2 minutes with zero
rate-limit errors. The window length is configurable via
`BATCH_INTER_CALL_DELAY_S` (default 62 s — the 60 s rolling window plus a
small safety margin). If a batch needs a retry (e.g. malformed output), the
timer resets to the retry's start time so the next batch never fires too
early.

### Other UI features

- **Edit & Regenerate** — reopen any generated assessment's plan in the
  form, tweak it, and regenerate.
- **Settings page** — per-user saved defaults (department, semester,
  faculty name) that pre-fill the generation form.
- **RAG:** FAISS vector store with `BAAI/bge-small-en-v1.5` embeddings over
  PDF/DOCX/TXT documents in `knowledge/`. The RAG stage is **non-fatal**: if
  embeddings aren't available in the environment, the pipeline still works,
  just without retrieval context.

## Project layout

```
artifacts/edupilot/
├── app.py               # Streamlit UI — pages: Home, Generate, History, Settings
├── orchestrator.py      # 6-stage pipeline coordinator
├── agent.py             # The specialized agents (planning, generation, analytics, review, download)
├── rag.py               # Document ingestion + FAISS semantic retrieval
├── prompts.py           # All LLM prompts, incl. JSON schema enforcement
├── models.py            # Typed dataclasses: AssessmentPlan, Question, Metadata, enums
├── downloads.py         # Markdown / Word / PDF export engine (ReportLab for PDF)
├── auth.py              # Login, session cookies, per-user settings & data isolation
├── config.py            # Env-var driven configuration
├── logging_utils.py     # Shared logging setup
├── verify_robustness.py # Standalone validation utility
├── test_batching.py     # Unit tests for batched generation (splitting, timing, merging)
├── run.sh               # Supervised launcher (auto-restarts Streamlit on crash)
├── requirements.txt
├── users.yaml           # Registered users (bcrypt-hashed passwords)
├── knowledge/           # Shared syllabus & reference documents for RAG
├── vectorstore/         # FAISS index built from knowledge/
└── runs/<username>/     # Per-user assessment history (JSON)
```

## Authentication & data

- Login via `streamlit-authenticator` with credentials in `users.yaml`
  (bcrypt hashes) and signed session cookies.
- Each user gets an isolated history directory (`runs/<username>/`) and
  personal saved defaults (department, semester, faculty name) editable on
  the **Settings** page.
- The knowledge base (`knowledge/` + `vectorstore/`) is shared by all users.

## Configuration

| Variable | Required | Purpose |
|----------|----------|---------|
| `GROQ_API_KEY` | Yes | Groq LLM access |
| `SESSION_SECRET` | Yes | Signing key for auth cookies |
| `GROQ_MODEL_NAME` | No | LLM model (default `openai/gpt-oss-120b`) |
| `GROQ_MAX_TOKENS` | No | Output-token budget per LLM call (default `6000` — sized to fit Groq's free-tier 8000 tokens/min limit, which counts the *requested* budget up front) |
| `BATCH_SIZE` | No | Max questions per LLM call in batched generation (default `10`) |
| `BATCH_THRESHOLD` | No | Question count above which generation always batches (default `30`; the per-type token estimate usually triggers batching earlier) |
| `BATCH_INTER_CALL_DELAY_S` | No | Length of the rolling TPM cooldown window in seconds (default `62`) |
| `LOG_LEVEL` | No | Logging verbosity (default `INFO`) |

## Running

```bash
cd artifacts/edupilot
pip install -r requirements.txt
bash run.sh        # supervised: serves on port 8000 under /edupilot
```

Or directly:

```bash
python -m streamlit run app.py --server.port 8000 --server.baseUrlPath /edupilot
```

## Tech stack

Python · Streamlit · LangChain (≥0.3) · Groq · FAISS ·
sentence-transformers · python-docx · ReportLab · streamlit-authenticator

## Notes

- **MCP (Model Context Protocol) is not used** — the app talks to Groq
  directly through LangChain; there are no MCP servers or clients.
- Assessments are generated as strict JSON and parsed into typed models, so
  malformed LLM output is detected rather than silently exported.
