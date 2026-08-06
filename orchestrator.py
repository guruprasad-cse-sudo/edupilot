"""
EduPilot AI Faculty Assistant
Module: orchestrator.py
Version: 4.3.0
Author: EduPilot Team
Purpose: End-to-end multi-agent pipeline orchestrator.  A single
         PipelineOrchestrator.run() call sequences all six stages —
         Planning → RAG → Generation → Analytics → Reviewer → Export Prep —
         passing typed objects between stages, capturing per-stage status,
         and persisting a run record to runs/ for dashboard history.

         Design decisions implemented:
           - Progress callback: Callable[[str, int], None]; no Streamlit coupling.
           - RAG failure is non-fatal; pipeline continues with empty context.
           - Analytics, Reviewer, and Export Prep failures are non-fatal;
             partial results are preserved in PipelineResult.
           - Planning and Generation failures abort the pipeline early.
           - success = (assessment is not None).
           - Run persistence: JSON flat files in config.runs_dir;
             runs/index.json for history listing; last-write-wins (no lock).
           - Persistence failures are logged but do not affect the returned result.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple, Union

from config import config
from logging_utils import get_logger
from models import (
    AnalyticsReport,
    Assessment,
    AssessmentPlan,
    PipelineResult,
    ReviewerResult,
    RunRecord,
    SourceAttribution,
    StageStatus,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Process-wide RAGModule singleton — avoids recreating (and potentially
# re-loading embeddings / FAISS) on every pipeline run.
# ---------------------------------------------------------------------------
_RAG_MODULE_SINGLETON = None


def _get_rag_module():
    """Return the process-wide RAGModule singleton, creating it on first call."""
    global _RAG_MODULE_SINGLETON
    if _RAG_MODULE_SINGLETON is None:
        from rag import RAGModule  # lazy import
        _RAG_MODULE_SINGLETON = RAGModule()
        logger.info("RAGModule singleton created (process-wide cache)")
    return _RAG_MODULE_SINGLETON


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _now_ms() -> int:
    """Return monotonic clock time in milliseconds.

    Returns:
        int: Current monotonic time as integer milliseconds.
    """
    return int(time.monotonic() * 1000)


def _elapsed_ms(start_ms: int) -> int:
    """Return milliseconds elapsed since *start_ms*.

    Args:
        start_ms: Start time from :func:`_now_ms`.

    Returns:
        int: Elapsed milliseconds.
    """
    return int(time.monotonic() * 1000) - start_ms


def _utcnow() -> str:
    """Return current UTC time as an ISO-8601 string.

    Returns:
        str: e.g. ``"2025-01-15T12:34:56.789012+00:00"``
    """
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# PipelineOrchestrator
# ---------------------------------------------------------------------------


class PipelineOrchestrator:
    """Sequences the EduPilot multi-agent pipeline end-to-end.

    Each run is identified by a UUID4, timed per stage, and persisted to
    ``runs/<run_id>.json`` and ``runs/index.json`` on the filesystem.

    Stages (in order):
        1. **planning** — :class:`~agent.PlanningAgent` parses the faculty
           request into a typed :class:`~models.AssessmentPlan`.
        2. **rag** — :class:`~rag.RAGModule` retrieves relevant knowledge
           chunks and formats a context string.  Non-fatal.
        3. **generation** — :class:`~agent.AssessmentAgent` generates the
           :class:`~models.Assessment` via Groq LLM.  Fatal on failure.
        4. **analytics** — :class:`~agent.AnalyticsAgent` computes
           deterministic metrics into an :class:`~models.AnalyticsReport`.
           Non-fatal.
        5. **reviewer** — :class:`~agent.ReviewerAgent` runs an LLM quality
           audit and returns a :class:`~models.ReviewerResult`.  Non-fatal;
           result is attached to ``analytics.reviewer_result``.
        6. **export_prep** — :class:`~agent.DownloadAgent` renders Markdown,
           Word, and PDF bytes.  Each format is independent; one format's
           failure does not abort the others.  Non-fatal.

    Usage::

        result = PipelineOrchestrator().run(
            "Create a 5-question Quiz on Binary Search Trees…",
            progress_callback=lambda stage, pct: print(f"{stage}: {pct}%"),
        )
        if result.success:
            print(result.assessment.question_count, "questions generated")
    """

    def __init__(self, runs_dir: Optional[Path] = None) -> None:
        """Initialise the PipelineOrchestrator.

        Args:
            runs_dir: Optional per-user directory for run persistence.
                Defaults to the global ``config.runs_dir`` when omitted.
        """
        self._runs_dir = runs_dir
        logger.info("PipelineOrchestrator initialised (runs_dir=%s)", runs_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        faculty_request: Union[str, dict],
        progress_callback: Optional[Callable[[str, int], None]] = None,
        batch_status_callback: Optional[Callable[[str], None]] = None,
    ) -> PipelineResult:
        """Execute the full six-stage pipeline and return a consolidated result.

        Args:
            faculty_request: Either a natural-language faculty requirement
                string, or an already-structured dict (e.g. from the
                Streamlit generation form).  Dict input skips the Planning
                Agent's LLM call.
            progress_callback: Optional callable invoked after key
                milestones with ``(stage_name: str, percent: int)``.
                The stage name is one of the six stage identifiers; percent
                is 0–100.  Exceptions inside the callback are silently
                swallowed so a broken UI callback cannot kill the pipeline.
            batch_status_callback: Optional callable invoked after each
                LLM batch completes during batched generation.  Receives a
                human-readable status string such as
                "Generating questions 9–16 of 25… (batch 2 of 3)".
                Swallowed on exception.  Ignored on single-call generation.

        Returns:
            PipelineResult: Consolidated run result.  ``success`` is
                ``True`` iff an ``Assessment`` was produced.  Non-fatal
                stage failures are recorded in ``stages`` but do not flip
                ``success``.
        """
        run_id = str(uuid.uuid4())
        started_at = _utcnow()
        stages: List[StageStatus] = []

        # Accumulators — populated stage by stage
        plan: Optional[AssessmentPlan] = None
        assessment: Optional[Assessment] = None
        analytics: Optional[AnalyticsReport] = None
        reviewer_result: Optional[ReviewerResult] = None
        export_bytes: Dict[str, bytes] = {}
        rag_sources: List[SourceAttribution] = []
        rag_context: str = ""
        fatal_error: Optional[str] = None

        logger.info(
            "PipelineOrchestrator.run(): run_id=%s input_type=%s",
            run_id,
            "dict" if isinstance(faculty_request, dict) else "str",
        )

        def _cb(stage: str, pct: int) -> None:
            """Fire the progress callback, swallowing any exceptions."""
            if progress_callback:
                try:
                    progress_callback(stage, pct)
                except Exception as cb_exc:  # noqa: BLE001
                    logger.debug(
                        "Progress callback raised (ignored): %s", cb_exc
                    )

        # ==================================================================
        # Stage 1 — Planning
        # ==================================================================
        _cb("planning", 5)
        t = _now_ms()
        try:
            from agent import PlanningAgent  # lazy import
            planner = PlanningAgent()
            plan = planner.plan(faculty_request)
            stages.append(
                StageStatus(
                    stage="planning",
                    status="ok",
                    duration_ms=_elapsed_ms(t),
                )
            )
            logger.info(
                "Stage planning OK: type=%s questions=%d",
                plan.assessment_type,
                plan.question_count,
            )
            _cb("planning", 15)
        except Exception as exc:  # fatal
            stages.append(
                StageStatus(
                    stage="planning",
                    status="error",
                    duration_ms=_elapsed_ms(t),
                    error=str(exc),
                )
            )
            fatal_error = f"Planning Agent failed: {exc}"
            logger.error("Stage planning FATAL: %s", exc)
            return self._build_and_persist(
                run_id, started_at, plan, assessment, analytics,
                reviewer_result, export_bytes, rag_sources, stages, fatal_error,
            )

        # ==================================================================
        # Stage 2 — RAG Retrieval  (non-fatal)
        # ==================================================================
        _cb("rag", 18)
        t = _now_ms()
        try:
            from rag import format_rag_context  # lazy import
            rag_module = _get_rag_module()
            query = (
                f"{plan.course_name} {plan.topics} {plan.bloom_targets}"
            ).strip()
            rag_results: List[Tuple[str, SourceAttribution]] = (
                rag_module.retrieve(query, top_k=5)
            )
            rag_sources = [attr for _, attr in rag_results]
            rag_context = format_rag_context(rag_results) if rag_results else ""

            if rag_results:
                stages.append(
                    StageStatus(
                        stage="rag",
                        status="ok",
                        duration_ms=_elapsed_ms(t),
                    )
                )
                logger.info(
                    "Stage rag OK: %d chunks retrieved", len(rag_results)
                )
            else:
                stages.append(
                    StageStatus(
                        stage="rag",
                        status="skipped",
                        duration_ms=_elapsed_ms(t),
                        warning=(
                            "Knowledge base is empty or no matching chunks "
                            "were found; proceeding without RAG context."
                        ),
                    )
                )
                logger.warning(
                    "Stage rag SKIPPED: no knowledge-base content"
                )
        except Exception as exc:  # non-fatal
            stages.append(
                StageStatus(
                    stage="rag",
                    status="error",
                    duration_ms=_elapsed_ms(t),
                    error=str(exc),
                    warning=(
                        "RAG retrieval failed; proceeding without "
                        "knowledge context."
                    ),
                )
            )
            logger.warning(
                "Stage rag ERROR (non-fatal, proceeding without context): %s",
                exc,
            )
        _cb("rag", 35)

        # ==================================================================
        # Stage 3 — Assessment Generation  (fatal on failure)
        # ==================================================================
        _cb("generation", 38)
        t = _now_ms()
        try:
            from agent import AssessmentAgent  # lazy import
            gen_agent = AssessmentAgent()

            # Build a batch-progress callback that updates both the numeric
            # progress signal and the human-readable status string.
            def _batch_progress_cb(
                batch_idx: int,
                total_batches: int,
                qs_start: int,
                qs_end: int,
                total_qs: int,
            ) -> None:
                """Relay per-batch progress to the UI callbacks."""
                msg = (
                    f"Generating questions {qs_start}–{qs_end} of {total_qs}… "
                    f"(batch {batch_idx} of {total_batches})"
                )
                logger.info("Generation batch progress: %s", msg)
                # Interpolate generation stage from 38 % → 64 % across batches
                gen_pct = 38 + int(26 * batch_idx / total_batches)
                _cb("generation", gen_pct)
                if batch_status_callback:
                    try:
                        batch_status_callback(msg)
                    except Exception as bsc_exc:  # noqa: BLE001
                        logger.debug(
                            "batch_status_callback raised (ignored): %s", bsc_exc
                        )

            assessment = gen_agent.generate(
                plan=plan,
                rag_context=rag_context,
                sources=rag_sources,
                batch_progress_callback=_batch_progress_cb,
            )
            stages.append(
                StageStatus(
                    stage="generation",
                    status="ok",
                    duration_ms=_elapsed_ms(t),
                )
            )
            logger.info(
                "Stage generation OK: %d questions, %d total marks",
                assessment.question_count,
                assessment.total_marks,
            )
            _cb("generation", 65)
        except Exception as exc:  # fatal
            stages.append(
                StageStatus(
                    stage="generation",
                    status="error",
                    duration_ms=_elapsed_ms(t),
                    error=str(exc),
                )
            )
            fatal_error = f"Assessment Agent failed: {exc}"
            logger.error("Stage generation FATAL: %s", exc)
            return self._build_and_persist(
                run_id, started_at, plan, assessment, analytics,
                reviewer_result, export_bytes, rag_sources, stages, fatal_error,
            )

        # ==================================================================
        # Stage 4 — Analytics  (non-fatal)
        # ==================================================================
        _cb("analytics", 67)
        t = _now_ms()
        try:
            from agent import AnalyticsAgent  # lazy import
            analytics_agent = AnalyticsAgent()
            analytics = analytics_agent.analyse(assessment)
            stages.append(
                StageStatus(
                    stage="analytics",
                    status="ok",
                    duration_ms=_elapsed_ms(t),
                )
            )
            logger.info(
                "Stage analytics OK: quality_score=%.1f time_saved=%dm",
                analytics.quality_score,
                analytics.estimated_time_saved_minutes,
            )
        except Exception as exc:  # non-fatal
            stages.append(
                StageStatus(
                    stage="analytics",
                    status="error",
                    duration_ms=_elapsed_ms(t),
                    error=str(exc),
                    warning="Analytics computation failed; partial result preserved.",
                )
            )
            logger.error("Stage analytics ERROR (non-fatal): %s", exc)
        _cb("analytics", 75)

        # ==================================================================
        # Stage 5 — Reviewer  (non-fatal; ReviewerAgent never raises)
        # ==================================================================
        _cb("reviewer", 77)
        t = _now_ms()
        try:
            from agent import ReviewerAgent  # lazy import
            reviewer_agent = ReviewerAgent()
            reviewer_result = reviewer_agent.review(assessment)

            # Attach to analytics so AnalyticsReport.reviewer_result is populated
            if analytics is not None:
                analytics.reviewer_result = reviewer_result

            reviewer_status = "error" if reviewer_result.error else "ok"
            stages.append(
                StageStatus(
                    stage="reviewer",
                    status=reviewer_status,
                    duration_ms=_elapsed_ms(t),
                    error=reviewer_result.error if reviewer_result.error else None,
                    warning=(
                        f"Reviewer LLM degraded: {reviewer_result.error}"
                        if reviewer_result.error
                        else None
                    ),
                )
            )
            logger.info(
                "Stage reviewer %s: llm_quality_score=%.1f",
                reviewer_status.upper(),
                reviewer_result.quality_score,
            )
        except Exception as exc:  # non-fatal (unexpected; ReviewerAgent is designed not to raise)
            stages.append(
                StageStatus(
                    stage="reviewer",
                    status="error",
                    duration_ms=_elapsed_ms(t),
                    error=str(exc),
                )
            )
            logger.error("Stage reviewer ERROR (non-fatal): %s", exc)
        _cb("reviewer", 90)

        # ==================================================================
        # Stage 6 — Export Prep  (non-fatal; per-format errors are isolated)
        # ==================================================================
        _cb("export_prep", 91)
        t = _now_ms()
        export_errors: Dict[str, str] = {}

        try:
            from agent import DownloadAgent  # lazy import
            dl_agent = DownloadAgent()

            for fmt in ("markdown", "docx", "pdf"):
                try:
                    export_bytes[fmt] = dl_agent.export(assessment, fmt)
                    logger.info(
                        "Export format '%s' OK: %d bytes",
                        fmt,
                        len(export_bytes[fmt]),
                    )
                except Exception as fmt_exc:  # noqa: BLE001
                    export_errors[fmt] = str(fmt_exc)
                    logger.warning(
                        "Export format '%s' FAILED (non-fatal): %s",
                        fmt,
                        fmt_exc,
                    )

            export_ok = len(export_bytes)
            export_failed = len(export_errors)
            export_warning = (
                f"Format(s) failed: {list(export_errors.keys())}" if export_errors else None
            )
            export_status = "ok" if not export_errors else (
                "error" if not export_bytes else "ok"
            )
            stages.append(
                StageStatus(
                    stage="export_prep",
                    status=export_status,
                    duration_ms=_elapsed_ms(t),
                    error=str(export_errors) if export_errors and not export_bytes else None,
                    warning=export_warning,
                )
            )
            logger.info(
                "Stage export_prep done: %d format(s) OK, %d failed",
                export_ok,
                export_failed,
            )
        except Exception as exc:  # non-fatal
            stages.append(
                StageStatus(
                    stage="export_prep",
                    status="error",
                    duration_ms=_elapsed_ms(t),
                    error=f"DownloadAgent unavailable: {exc}",
                )
            )
            logger.error(
                "Stage export_prep ERROR (non-fatal): %s", exc
            )
        _cb("export_prep", 100)

        # ==================================================================
        # Build result, persist, return
        # ==================================================================
        return self._build_and_persist(
            run_id, started_at, plan, assessment, analytics,
            reviewer_result, export_bytes, rag_sources, stages, fatal_error,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_and_persist(
        self,
        run_id: str,
        started_at: str,
        plan: Optional[AssessmentPlan],
        assessment: Optional[Assessment],
        analytics: Optional[AnalyticsReport],
        reviewer: Optional[ReviewerResult],
        export_bytes: Dict[str, bytes],
        rag_sources: List[SourceAttribution],
        stages: List[StageStatus],
        fatal_error: Optional[str],
    ) -> PipelineResult:
        """Build a :class:`PipelineResult`, persist it, and return it.

        Separated from ``run()`` so early-abort paths and the happy-path
        share the same finalisation logic.

        Args:
            run_id: UUID4 run identifier.
            started_at: ISO-8601 start timestamp.
            plan: Plan produced by PlanningAgent (may be None on planning failure).
            assessment: Generated Assessment (may be None on generation failure).
            analytics: AnalyticsReport (may be None on analytics failure).
            reviewer: ReviewerResult (may be None when reviewer failed/skipped).
            export_bytes: Format → bytes map for successfully exported formats.
            rag_sources: Retrieved SourceAttribution objects.
            stages: Ordered stage status records.
            fatal_error: Early-abort description, or None.

        Returns:
            PipelineResult: Fully populated result object.
        """
        result = PipelineResult(
            run_id=run_id,
            started_at=started_at,
            completed_at=_utcnow(),
            plan=plan,
            assessment=assessment,
            analytics=analytics,
            reviewer=reviewer,
            export_bytes=export_bytes,
            rag_sources=rag_sources,
            stages=stages,
            success=(assessment is not None),
            fatal_error=fatal_error,
        )
        self._persist(result)
        return result

    def _persist(self, result: PipelineResult) -> None:
        """Write run JSON and update the history index.

        Strategy:
          - ``runs/<run_id>.json``: full serialised result (no export bytes).
          - ``runs/index.json``: JSON array of :class:`~models.RunRecord`
            dicts, newest first.  Last-write-wins; no file locking.

        Failures are logged at ERROR level but do not propagate — a
        persistence failure must never prevent the pipeline result from
        being returned to the caller.

        Args:
            result: The completed :class:`PipelineResult` to persist.
        """
        try:
            runs_dir = self._runs_dir or config.runs_dir
            runs_dir.mkdir(parents=True, exist_ok=True)

            # ── Full run record ────────────────────────────────────────────
            run_path = runs_dir / f"{result.run_id}.json"
            run_path.write_text(
                json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Run record persisted: %s", run_path.name)

            # ── History index (append newest first, last-write-wins) ───────
            index_path = runs_dir / "index.json"
            existing: list = []
            if index_path.exists():
                try:
                    existing = json.loads(
                        index_path.read_text(encoding="utf-8")
                    )
                    if not isinstance(existing, list):
                        existing = []
                except Exception as parse_exc:
                    logger.warning(
                        "Could not parse runs/index.json (%s); "
                        "starting fresh index.",
                        parse_exc,
                    )
                    existing = []

            record = RunRecord.from_pipeline_result(result)
            existing.insert(0, record.to_dict())  # newest first

            index_path.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(
                "Run index updated: %d total entries", len(existing)
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to persist run record run_id=%s (non-fatal): %s",
                result.run_id,
                exc,
            )
