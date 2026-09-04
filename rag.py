"""
EduPilot AI Faculty Assistant
Module: rag.py
Version: 4.1.0
Author: EduPilot Team
Purpose: Retrieval-Augmented Generation (RAG) layer. Handles ingestion of
         knowledge-base documents (PDF, DOCX, TXT) into a FAISS vector store
         using HuggingFace sentence-transformers embeddings, and provides
         semantic retrieval with source attribution.

         Key design decisions:
         - Embedding model: all-MiniLM-L6-v2 (sentence-transformers)
         - Chunking: RecursiveCharacterTextSplitter, chunk_size=500, overlap=50
         - DOCX support included via python-docx
         - Lazy loading: the FAISS index is loaded/built on first retrieve() call
         - Index version-stamped in vectorstore/meta.json; mismatch triggers
           a warning and optional rebuild
         - Empty/missing knowledge dir is handled gracefully (returns 0 / [])
         - force_rebuild writes to a temp location then swaps to avoid corruption
"""

from __future__ import annotations

import gc
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from config import config
from logging_utils import get_logger
from models import SourceAttribution

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt", ".docx")

# ---------------------------------------------------------------------------
# Process-wide embeddings cache — avoids reloading the ONNX model on every
# RAGModule instantiation (one model load per process, not per pipeline run).
# ---------------------------------------------------------------------------
_EMBEDDINGS_CACHE: dict = {}

# Number of chunks embedded per FAISS.from_documents()/add_documents() call
# during ingest(). Keeps peak memory roughly constant regardless of
# knowledge-base size — see the comment in ingest() for why this matters
# on memory-constrained hosts.
EMBEDDING_BATCH_SIZE = 32
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50
META_FILENAME: str = "meta.json"
INDEX_FILENAME: str = "index.faiss"
PKL_FILENAME: str = "index.pkl"


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class KnowledgeBaseEmptyError(RuntimeError):
    """Raised when retrieval is attempted on an empty or missing index."""


class RAGIngestionError(RuntimeError):
    """Raised when document ingestion fails unrecoverably."""


# ---------------------------------------------------------------------------
# Document loading helpers
# ---------------------------------------------------------------------------

def _load_pdf(path: Path) -> List[Tuple[str, int]]:
    """Load a PDF file and return (text, page_number) pairs.

    Args:
        path: Absolute path to the PDF file.

    Returns:
        List of (page_text, 1-based page number) tuples. Empty pages are
        skipped.

    Raises:
        RAGIngestionError: If the file cannot be parsed.
    """
    try:
        from pypdf import PdfReader  # lazy import — keep startup fast
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((text, i))
        logger.debug("Loaded PDF %s: %d non-empty pages", path.name, len(pages))
        return pages
    except Exception as exc:
        raise RAGIngestionError(f"Failed to load PDF '{path}': {exc}") from exc


def _load_txt(path: Path) -> List[Tuple[str, int]]:
    """Load a plain-text file and return it as a single (text, 0) pair.

    Args:
        path: Absolute path to the text file.

    Returns:
        List containing a single (full_text, 0) tuple (page 0 = n/a).

    Raises:
        RAGIngestionError: If the file cannot be read.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return []
        logger.debug("Loaded TXT %s: %d chars", path.name, len(text))
        return [(text, 0)]
    except Exception as exc:
        raise RAGIngestionError(f"Failed to load TXT '{path}': {exc}") from exc


def _load_docx(path: Path) -> List[Tuple[str, int]]:
    """Load a DOCX file and return paragraph text as a single (text, 0) pair.

    Args:
        path: Absolute path to the DOCX file.

    Returns:
        List containing a single (full_text, 0) tuple.

    Raises:
        RAGIngestionError: If the file cannot be parsed.
    """
    try:
        from docx import Document as DocxDocument  # lazy import
        doc = DocxDocument(str(path))
        text = "\n".join(
            para.text for para in doc.paragraphs if para.text.strip()
        ).strip()
        if not text:
            return []
        logger.debug("Loaded DOCX %s: %d chars", path.name, len(text))
        return [(text, 0)]
    except Exception as exc:
        raise RAGIngestionError(f"Failed to load DOCX '{path}': {exc}") from exc


def _load_document(path: Path) -> List[Tuple[str, int]]:
    """Dispatch to the appropriate loader based on file extension.

    Args:
        path: Path to the document.

    Returns:
        List of (text, page_number) tuples.

    Raises:
        RAGIngestionError: For unsupported or unreadable files.
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path)
    elif ext == ".txt":
        return _load_txt(path)
    elif ext == ".docx":
        return _load_docx(path)
    else:
        raise RAGIngestionError(f"Unsupported file type: '{path.suffix}'")


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def _chunk_text(
    text: str,
    source_name: str,
    page_number: int,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[dict]:
    """Split text into overlapping chunks with metadata.

    Args:
        text: Raw text to split.
        source_name: Human-readable document name (filename) for attribution.
        page_number: Originating page (0 if N/A for TXT/DOCX).
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap in characters between successive chunks.

    Returns:
        List of dicts with keys: ``text``, ``source``, ``page``,
        ``chunk_index``.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # lazy

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    raw_chunks = splitter.split_text(text)
    return [
        {
            "text": chunk,
            "source": source_name,
            "page": page_number,
            "chunk_index": idx,
        }
        for idx, chunk in enumerate(raw_chunks)
    ]


# ---------------------------------------------------------------------------
# Format helper (module-level, used by downstream agents)
# ---------------------------------------------------------------------------

def format_rag_context(
    results: List[Tuple[str, SourceAttribution]],
    max_chunks: int = 5,
) -> str:
    """Serialise retrieve() results into a prompt-ready context block.

    Each chunk is presented with its source reference so the LLM can
    attribute generated questions to specific documents.

    Args:
        results: Output of :meth:`RAGModule.retrieve` — a list of
            ``(chunk_text, SourceAttribution)`` tuples ordered by
            relevance descending.
        max_chunks: Maximum number of chunks to include (safety cap).

    Returns:
        A formatted multi-line string ready for injection into an LLM prompt.
        Returns an empty string when *results* is empty.

    Example output::

        [Source 1: lecture_notes.txt, page 0, score 0.87]
        Arrays store elements of the same type...

        [Source 2: syllabus.pdf, page 3, score 0.82]
        The course covers data structures including...
    """
    if not results:
        return ""

    parts: List[str] = []
    for i, (chunk_text, attr) in enumerate(results[:max_chunks], start=1):
        page_label = f", page {attr.page_number}" if attr.page_number else ""
        header = (
            f"[Source {i}: {attr.document_name}{page_label}, "
            f"score {attr.relevance_score:.2f}]"
        )
        parts.append(f"{header}\n{chunk_text.strip()}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# RAGModule
# ---------------------------------------------------------------------------

class RAGModule:
    """Manages the FAISS vector store and retrieval pipeline.

    The RAG module indexes documents from the knowledge directory and
    retrieves relevant chunks at query time. Each retrieved chunk carries
    source attribution metadata used downstream for display and export.

    Usage::

        rag = RAGModule()
        # Ingest all documents in knowledge/ (no-op if already indexed):
        count = rag.ingest()
        # Retrieve top-5 chunks for a query:
        results = rag.retrieve("binary search tree traversal", top_k=5)
        context_str = format_rag_context(results)
    """

    def __init__(self) -> None:
        """Initialise the RAG module.

        Reads vectorstore and knowledge paths from the central config.
        Does not load the vector store eagerly; the index is loaded/built
        lazily on the first :meth:`retrieve` call.
        """
        self._vectorstore_path: Path = config.vectorstore_path
        self._knowledge_dir: Path = config.knowledge_dir
        self._embedding_model_name: str = config.embedding_model_name
        self._store = None  # populated lazily by load_or_create()
        logger.debug(
            "RAGModule initialised. vectorstore=%s knowledge=%s embedding=%s",
            self._vectorstore_path,
            self._knowledge_dir,
            self._embedding_model_name,
        )

    # ------------------------------------------------------------------
    # Meta / version stamp helpers
    # ------------------------------------------------------------------

    def _meta_path(self) -> Path:
        """Return the path to the vectorstore metadata file.

        Returns:
            Path: ``<vectorstore_path>/meta.json``
        """
        return self._vectorstore_path / META_FILENAME

    def _read_meta(self) -> dict:
        """Read and return the vectorstore metadata dict.

        Returns:
            dict with at least ``{"embedding_model": str}``, or empty dict
            if the file does not exist.
        """
        meta_path = self._meta_path()
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Could not read vectorstore meta.json: %s", exc)
        return {}

    def _write_meta(self) -> None:
        """Persist the current embedding model name to meta.json.

        Writes ``{"embedding_model": "<model_name>"}`` to the vectorstore
        directory for version-stamp checking on subsequent loads.
        """
        self._vectorstore_path.mkdir(parents=True, exist_ok=True)
        meta = {"embedding_model": self._embedding_model_name}
        self._meta_path().write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        logger.debug("Wrote vectorstore meta.json: %s", meta)

    def _check_meta(self) -> bool:
        """Check whether the persisted index was built with the current model.

        Returns:
            ``True`` if the embedding model matches (or no meta exists yet).
            ``False`` on a mismatch — caller should warn and consider rebuild.
        """
        meta = self._read_meta()
        stored_model = meta.get("embedding_model")
        if stored_model and stored_model != self._embedding_model_name:
            logger.warning(
                "Embedding model mismatch: index was built with '%s' but "
                "current config is '%s'. Retrieval quality may be degraded. "
                "Run ingest(force_rebuild=True) to rebuild.",
                stored_model,
                self._embedding_model_name,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Index persistence helpers
    # ------------------------------------------------------------------

    def _clear_index(self) -> None:
        """Remove any persisted index files and drop the in-memory store.

        Called on forced rebuilds that end with zero indexable content so a
        stale index can never keep serving deleted documents.
        """
        self._store = None
        for fname in (INDEX_FILENAME, PKL_FILENAME, META_FILENAME):
            path = self._vectorstore_path / fname
            try:
                if path.exists():
                    path.unlink()
                    logger.info("Removed stale vectorstore file: %s", path)
            except OSError as exc:
                logger.warning("Could not remove %s: %s", path, exc)

    def _index_exists(self) -> bool:
        """Return True if a valid FAISS index is present on disk.

        Returns:
            bool: True when both ``index.faiss`` and ``index.pkl`` exist.
        """
        return (
            (self._vectorstore_path / INDEX_FILENAME).exists()
            and (self._vectorstore_path / PKL_FILENAME).exists()
        )

    def _get_embeddings(self):
        """Construct and return the FastEmbed embeddings object.

        Uses ONNX-based FastEmbed (no torch dependency — torch cannot be
        installed in this environment). Downloads the model on first call
        (~65 MB for bge-small-en-v1.5); subsequent calls use the local cache.

        The returned instance is cached in ``_EMBEDDINGS_CACHE`` keyed by
        model name so the ONNX runtime is loaded only once per process,
        regardless of how many RAGModule instances are created.

        Returns:
            FastEmbedEmbeddings instance.
        """
        if self._embedding_model_name in _EMBEDDINGS_CACHE:
            return _EMBEDDINGS_CACHE[self._embedding_model_name]

        from langchain_community.embeddings.fastembed import (  # lazy import
            FastEmbedEmbeddings,
        )
        logger.info(
            "Loading embedding model '%s' (may download on first run)…",
            self._embedding_model_name,
        )
        # Use an explicit, absolute cache directory (not the library default,
        # which is a *relative* "local_cache" resolved against whatever the
        # current working directory happens to be at call time). This
        # guarantees the build step and the running app process always
        # agree on where the cached model lives, regardless of CWD
        # differences between build and runtime on the host.
        cache_dir = str(Path(__file__).resolve().parent / "model_cache")
        emb = FastEmbedEmbeddings(
            model_name=self._embedding_model_name, cache_dir=cache_dir
        )
        _EMBEDDINGS_CACHE[self._embedding_model_name] = emb
        return emb

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_or_create(self, force_rebuild: bool = False) -> None:
        """Load the FAISS index from disk, or ingest documents to create it.

        If ``force_rebuild`` is True, the existing index is discarded and
        re-built from the knowledge directory regardless.

        Args:
            force_rebuild: When ``True``, rebuild the index unconditionally.
        """
        if force_rebuild:
            logger.info("force_rebuild=True — discarding existing index.")
            self.ingest(force_rebuild=True)
            return

        if self._store is not None:
            logger.debug("Index already loaded in memory, skipping reload.")
            return

        if self._index_exists():
            if not self._check_meta():
                # Index was built with a different embedding model — loading
                # it would give semantically wrong similarity scores.
                logger.info(
                    "Embedding model changed — rebuilding index automatically."
                )
                self.ingest(force_rebuild=True)
                return
            try:
                from langchain_community.vectorstores import FAISS  # lazy
                embeddings = self._get_embeddings()
                self._store = FAISS.load_local(
                    str(self._vectorstore_path),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info(
                    "FAISS index loaded from disk: %s",
                    self._vectorstore_path,
                )
                return
            except Exception as exc:
                logger.warning(
                    "Failed to load existing index (%s); rebuilding.", exc
                )
                self._store = None

        # No index on disk (or load failed) — ingest from scratch.
        self.ingest(force_rebuild=False)

    def ingest(self, force_rebuild: bool = False) -> int:
        """Ingest all documents from the knowledge directory into FAISS.

        Scans ``knowledge/`` for PDF, DOCX, and TXT files, chunks them,
        embeds them with the configured sentence-transformer model, and
        writes the FAISS index to ``vectorstore/``.

        When ``force_rebuild=True`` the new index is written to a temporary
        directory first, then atomically swapped into place so the existing
        index is never left in a corrupt state mid-write.

        Args:
            force_rebuild: When ``True``, rebuild the index even if it
                already exists on disk.

        Returns:
            int: Number of document chunks indexed. Returns 0 gracefully
            when the knowledge directory is missing or empty.
        """
        # ── Guard: skip if index already exists and no rebuild requested ──
        if not force_rebuild and self._index_exists() and self._store is not None:
            logger.info("Index already in memory and on disk; skipping ingest.")
            return 0

        # ── Scan knowledge directory ──────────────────────────────────────
        if not self._knowledge_dir.exists():
            logger.warning(
                "Knowledge directory does not exist: %s. "
                "Create it and add documents, then call ingest() again.",
                self._knowledge_dir,
            )
            if force_rebuild:
                self._clear_index()
            return 0

        doc_paths = sorted(
            p
            for p in self._knowledge_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not doc_paths:
            logger.warning(
                "Knowledge directory '%s' contains no supported documents "
                "(%s). Skipping index build.",
                self._knowledge_dir,
                ", ".join(SUPPORTED_EXTENSIONS),
            )
            if force_rebuild:
                # A forced rebuild with zero documents must not leave a stale
                # index behind — retrieval would silently serve old content.
                self._clear_index()
            return 0

        logger.info(
            "Ingesting %d document(s) from %s …",
            len(doc_paths),
            self._knowledge_dir,
        )

        # ── Load and chunk all documents ─────────────────────────────────
        all_chunks: List[dict] = []
        for path in doc_paths:
            try:
                pages = _load_document(path)
            except RAGIngestionError as exc:
                logger.error("Skipping '%s': %s", path.name, exc)
                continue

            for text, page_num in pages:
                chunks = _chunk_text(
                    text,
                    source_name=path.name,
                    page_number=page_num,
                )
                all_chunks.extend(chunks)
                logger.debug(
                    "  %s page %d → %d chunks", path.name, page_num, len(chunks)
                )

        if not all_chunks:
            logger.warning(
                "All documents loaded but produced zero chunks. "
                "Check that files are not empty or image-only PDFs."
            )
            if force_rebuild:
                self._clear_index()
            return 0

        logger.info("Total chunks to embed: %d", len(all_chunks))

        # ── Build FAISS index ─────────────────────────────────────────────
        from langchain_community.vectorstores import FAISS  # lazy
        from langchain_core.documents import Document  # lazy

        lc_docs = [
            Document(
                page_content=c["text"],
                metadata={
                    "source": c["source"],
                    "page": c["page"],
                    "chunk_index": c["chunk_index"],
                },
            )
            for c in all_chunks
        ]

        embeddings = self._get_embeddings()
        logger.info("Embedding %d chunks — this may take a moment …", len(lc_docs))

        # Embed in small batches rather than one FAISS.from_documents() call
        # over the entire document set. Embedding all chunks at once holds
        # every chunk's text AND its resulting vector in memory
        # simultaneously (on top of the already-loaded ONNX runtime,
        # Streamlit, and LangChain baseline) — on memory-constrained hosts
        # (e.g. Render's 512MB starter plan) this was observed to exceed
        # the limit and crash the process mid-ingest. Batching keeps peak
        # memory roughly constant regardless of knowledge-base size, at
        # the cost of a few extra FAISS merge calls (index build still
        # only touches disk once, at the end, via the atomic swap below).
        batch_size = EMBEDDING_BATCH_SIZE
        new_store = None
        for start in range(0, len(lc_docs), batch_size):
            batch = lc_docs[start: start + batch_size]
            if new_store is None:
                new_store = FAISS.from_documents(batch, embeddings)
            else:
                new_store.add_documents(batch)
            logger.debug(
                "  embedded %d/%d chunks",
                min(start + batch_size, len(lc_docs)),
                len(lc_docs),
            )
            gc.collect()

        # ── Atomic write (temp-dir then swap) ─────────────────────────────
        self._vectorstore_path.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(
            tempfile.mkdtemp(
                prefix="edupilot_vs_tmp_",
                dir=self._vectorstore_path.parent,
            )
        )
        try:
            new_store.save_local(str(tmp_dir))
            self._write_meta()  # write meta.json into final location

            # Move each index file from tmp into place
            for filename in (INDEX_FILENAME, PKL_FILENAME):
                src = tmp_dir / filename
                dst = self._vectorstore_path / filename
                if src.exists():
                    shutil.move(str(src), str(dst))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self._store = new_store
        logger.info(
            "FAISS index built and saved to '%s' (%d chunks).",
            self._vectorstore_path,
            len(all_chunks),
        )
        return len(all_chunks)

    def retrieve(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[str, SourceAttribution]]:
        """Retrieve the top-k most relevant chunks for *query*.

        Lazily loads or builds the FAISS index on the first call so callers
        never need to call :meth:`load_or_create` manually.

        Args:
            query: The search query string.
            top_k: Number of chunks to return (default 5).

        Returns:
            List of ``(chunk_text, SourceAttribution)`` tuples ordered by
            relevance score descending. Returns an empty list (never raises)
            when the knowledge base is empty or not yet indexed.
        """
        if not query or not query.strip():
            logger.warning("retrieve() called with empty query; returning [].")
            return []

        # Lazy load / build
        if self._store is None:
            if not self._index_exists() and not self._knowledge_dir_has_docs():
                logger.warning(
                    "Knowledge base is empty (no documents in '%s' and no "
                    "persisted index). Returning empty retrieval results.",
                    self._knowledge_dir,
                )
                return []
            self.load_or_create()

        if self._store is None:
            # load_or_create() found nothing to index
            logger.warning(
                "Index unavailable after load attempt; returning []."
            )
            return []

        try:
            docs_scores = self._store.similarity_search_with_score(
                query, k=top_k
            )
        except Exception as exc:
            logger.error("FAISS similarity search failed: %s", exc)
            return []

        results: List[Tuple[str, SourceAttribution]] = []
        for doc, raw_score in docs_scores:
            meta = doc.metadata
            # FAISS returns L2 distance; convert to a 0-1 similarity proxy
            similarity = float(1.0 / (1.0 + raw_score))
            attribution = SourceAttribution(
                document_name=meta.get("source", "unknown"),
                page_number=meta.get("page") or None,
                relevance_score=round(similarity, 4),
                excerpt=doc.page_content[:200],
            )
            results.append((doc.page_content, attribution))

        logger.info(
            "Retrieved %d chunks for query: '%s…'",
            len(results),
            query[:60],
        )
        return results

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _knowledge_dir_has_docs(self) -> bool:
        """Return True if the knowledge directory contains at least one supported file.

        Returns:
            bool: True when a supported document exists in ``knowledge/``.
        """
        if not self._knowledge_dir.exists():
            return False
        return any(
            p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            for p in self._knowledge_dir.iterdir()
        )


# ---------------------------------------------------------------------------
# CLI entry point — for local verification only
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    """Command-line verification helper.

    Usage::

        cd artifacts/edupilot
        python rag.py               # normal ingest + retrieve
        python rag.py --rebuild     # force full rebuild
        python rag.py --empty       # test empty-KB path
    """
    import argparse

    parser = argparse.ArgumentParser(description="EduPilot RAG CLI verifier")
    parser.add_argument(
        "--rebuild", action="store_true", help="Force index rebuild"
    )
    parser.add_argument(
        "--empty", action="store_true",
        help="Test empty knowledge base path (temporarily renames knowledge/)"
    )
    args = parser.parse_args()

    rag = RAGModule()

    if args.empty:
        print("\n=== Empty KB path test ===")
        # Temporarily point to a non-existent dir
        rag._knowledge_dir = Path("/tmp/edupilot_empty_test_kb")
        rag._vectorstore_path = Path("/tmp/edupilot_empty_test_vs")
        count = rag.ingest()
        print(f"ingest() returned: {count}  (expected 0)")
        results = rag.retrieve("any query")
        print(f"retrieve() returned: {results}  (expected [])")
        return

    print(f"\n=== EduPilot RAG CLI Verifier ===")
    print(f"Knowledge dir : {rag._knowledge_dir}")
    print(f"Vectorstore   : {rag._vectorstore_path}")
    print(f"Embedding     : {rag._embedding_model_name}")
    print()

    # Ingest
    count = rag.ingest(force_rebuild=args.rebuild)
    print(f"\n✓ ingest() → {count} chunks indexed")

    if count == 0:
        print("  (No documents found — add files to knowledge/ and retry)")
        return

    # Retrieve
    test_query = "data structures and algorithms"
    print(f"\n=== Retrieval test: '{test_query}' (top_k=3) ===")
    results = rag.retrieve(test_query, top_k=3)
    print(f"✓ retrieve() → {len(results)} results\n")
    for i, (text, attr) in enumerate(results, 1):
        print(f"  [{i}] source={attr.document_name!r}  page={attr.page_number}"
              f"  score={attr.relevance_score:.4f}")
        print(f"      excerpt: {text[:120].replace(chr(10), ' ')!r}")
        print()

    # format_rag_context
    ctx = format_rag_context(results)
    print("=== format_rag_context() output ===")
    print(ctx)

    # Reload test — simulates a fresh process
    print("\n=== Reload from disk test ===")
    rag2 = RAGModule()
    results2 = rag2.retrieve(test_query, top_k=2)
    print(f"✓ Fresh RAGModule.retrieve() → {len(results2)} results (from disk)")


if __name__ == "__main__":
    _cli_main()