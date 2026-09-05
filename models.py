"""
EduPilot AI Faculty Assistant
Module: models.py
Version: 4.3.0
Author: EduPilot Team
Purpose: Shared typed data models (dataclasses) representing assessment
         metadata, individual questions, analytics containers, the
         ReviewerResult returned by the Reviewer Agent, and the pipeline
         orchestration types (AssessmentPlan, StageStatus, PipelineResult,
         RunRecord) introduced in v4.3.  All models are importable by every
         other module and carry full type annotations.  Every dataclass
         exposes a to_dict() helper so downstream consumers (dashboard,
         download engine) can serialise without depending on
         dataclasses.asdict() recursion quirks.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class BloomLevel(str, Enum):
    """Bloom's Taxonomy cognitive levels (revised, 2001)."""

    REMEMBER = "Remember"
    UNDERSTAND = "Understand"
    APPLY = "Apply"
    ANALYZE = "Analyze"
    EVALUATE = "Evaluate"
    CREATE = "Create"


class DifficultyLevel(str, Enum):
    """Question difficulty classification."""

    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class AssessmentType(str, Enum):
    """Supported OBE assessment categories."""

    INTERNAL = "Internal Assessment"
    QUIZ = "Quiz"
    ASSIGNMENT = "Assignment"
    SEMESTER_EXAM = "Semester Examination"
    VIVA = "Viva"
    ROLE_PLAY = "Role Play"
    QUESTION_BANK = "Question Bank"
    CONSULTANCY_CASE = "Consultancy Case Study"


# ---------------------------------------------------------------------------
# VTU Semester Examination marks structure
# ---------------------------------------------------------------------------
# Standard VTU semester-end exam convention: each Module contributes exactly
# one full question worth a fixed number of marks (the student answers ONE
# of its two OR-alternatives), and the paper covers a fixed number of
# Modules totalling the university's standard maximum. These are used both
# to auto-compute sub-question marks when no custom blueprint is given, and
# to validate faculty-entered custom blueprints and topic counts.
VTU_MARKS_PER_FULL_QUESTION = 20
VTU_MAX_TOTAL_MARKS = 100
VTU_MAX_MODULES = VTU_MAX_TOTAL_MARKS // VTU_MARKS_PER_FULL_QUESTION  # 5


def split_sizes_for_pairing(n: int, max_subparts: int = 3) -> List[int]:
    """Split *n* questions into exactly one OR-alternative pair of groups.

    Shared by the generation layer (agent.py, to auto-derive a marks
    blueprint per topic) and the export layer (downloads.py, to group a
    flat question list into VTU-style Modules) — kept here in models.py
    so neither has to depend on the other for this pure, data-only logic.

    Real VTU exam papers give each Module exactly one choice: two
    alternative full questions (e.g. "Q1 OR Q2"), each broken into as
    many lettered sub-parts as needed — not multiple separate pairs
    within one module. Groups are split as evenly as possible between
    the two alternatives.

    Args:
        n: Total number of questions available for this module.
        max_subparts: Unused; kept for signature stability — real papers
            do not cap sub-parts per question, they flex to fit.

    Returns:
        ``[n]`` when there's only one question (nothing to pair against),
        otherwise a 2-element list of near-equal sizes summing to *n*
        (e.g. ``[5, 4]`` for ``n=9``).
    """
    if n <= 0:
        return []
    if n == 1:
        return [1]
    first = (n + 1) // 2
    second = n - first
    return [first, second]


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------

@dataclass
class SourceAttribution:
    """Represents a knowledge-base chunk used to generate a question.

    Attributes:
        document_name: Human-readable filename or title of the source document.
        page_number: Page (or chunk index) within the document. ``None`` when
            not applicable.
        relevance_score: Cosine similarity or retrieval score (0.0–1.0).
        excerpt: Short text excerpt for provenance display.
    """

    document_name: str
    page_number: Optional[int] = None
    relevance_score: float = 0.0
    excerpt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation.

        Returns:
            Dict with all fields; enum values are converted to their
            string primitives.
        """
        return {
            "document_name": self.document_name,
            "page_number": self.page_number,
            "relevance_score": self.relevance_score,
            "excerpt": self.excerpt,
        }


@dataclass
class Question:
    """A single OBE-aligned assessment question with full metadata.

    Attributes:
        question_id: Unique identifier within the assessment (e.g. "Q1").
        question_text: The question body as a string (may contain Markdown).
        bloom_level: Bloom's Taxonomy level targeted by this question.
        co_mapping: List of Course Outcome codes this question addresses
            (e.g. ["CO1", "CO3"]).
        difficulty: Difficulty level classification.
        marks: Marks allocated to this question.
        answer_key: Model answer or marking scheme as a string.
        question_type: Free-form descriptor, e.g. "MCQ", "Short Answer",
            "Essay", "Numerical".
        sources: Knowledge-base chunks used to generate this question.
        notes: Optional faculty notes or special instructions.
    """

    question_id: str
    question_text: str
    bloom_level: BloomLevel
    co_mapping: List[str]
    difficulty: DifficultyLevel
    marks: int
    answer_key: str
    question_type: str = "Short Answer"
    sources: List[SourceAttribution] = field(default_factory=list)
    notes: str = ""
    options: List[str] = field(default_factory=list)
    topic: str = ""
    """Syllabus topic/module this question was generated for — used to
    group questions into VTU-style exam paper Modules on export. Empty
    string for older saved runs predating this field."""
    blueprint_group: str = ""
    """"A" or "B" when this question was generated against a faculty-
    supplied custom marks blueprint (see AssessmentPlan.vtu_marks_blueprint)
    — identifies which of the two OR-alternative full questions this
    sub-part belongs to, so export can group precisely instead of
    auto-splitting. Empty string when no blueprint was used for this
    question's topic."""
    case_background: str = ""
    """For Consultancy Case Study assessments only: the realistic
    (fictional) company/scenario background and context the case is
    built on — kept separate from question_text (which holds the actual
    task/ask) so export can render them as distinct "Case Background" /
    "Your Task" sections. Empty string for every other assessment type."""

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation.

        Returns:
            Dict with enum values converted to their string primitives.
        """
        return {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "bloom_level": self.bloom_level.value,
            "co_mapping": self.co_mapping,
            "difficulty": self.difficulty.value,
            "marks": self.marks,
            "answer_key": self.answer_key,
            "question_type": self.question_type,
            "sources": [s.to_dict() for s in self.sources],
            "notes": self.notes,
            "options": list(self.options),
            "topic": self.topic,
            "blueprint_group": self.blueprint_group,
            "case_background": self.case_background,
        }


@dataclass
class AssessmentMetadata:
    """Metadata envelope for a generated assessment.

    Attributes:
        title: Assessment title (e.g. "Mid-Semester Quiz – Unit 3").
        course_code: Institutional course code (e.g. "CS3001").
        course_name: Full course name.
        assessment_type: OBE assessment category.
        semester: Semester label (e.g. "Semester 5, 2024-25").
        duration_minutes: Allotted time in minutes.
        total_marks: Sum of all question marks.
        department: Offering department name.
        faculty_name: Name of the faculty member.
        instructions: General instructions for students.
    """

    title: str
    course_code: str
    course_name: str
    assessment_type: AssessmentType
    semester: str = ""
    duration_minutes: int = 60
    total_marks: int = 0
    department: str = ""
    faculty_name: str = ""
    instructions: str = ""
    test_date: str = ""
    batch: str = ""
    """Student batch/admission year label (e.g. "2024"), shown on the
    institutional IAT header. Empty string when not specified."""
    teaching_department: str = ""
    """Abbreviated teaching department (e.g. "CSE"), shown on the
    institutional IAT header's metadata grid. Falls back to
    ``department`` when empty — see ParsedAssessment/AssessmentParser."""
    academic_year: str = ""
    """Academic year / term label (e.g. "2025-26 (Even Sem)"), appended
    to the exam title line on the institutional IAT header. Empty
    string when not specified."""
    iat_number: str = ""
    """Which Internal Assessment Test this is ("1" or "2"). Drives the
    fixed Syllabus Coverage table (see downloads.py's
    _SYLLABUS_COVERAGE_BY_IAT) — a constant, department-wide module
    coverage convention, not something derived per-question. Empty
    string when not applicable (e.g. non-IAT assessment types) or not
    specified, in which case the Syllabus Coverage table is omitted."""

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation.

        Returns:
            Dict with enum values converted to their string primitives.
        """
        return {
            "title": self.title,
            "course_code": self.course_code,
            "course_name": self.course_name,
            "assessment_type": self.assessment_type.value,
            "semester": self.semester,
            "duration_minutes": self.duration_minutes,
            "total_marks": self.total_marks,
            "department": self.department,
            "faculty_name": self.faculty_name,
            "instructions": self.instructions,
            "test_date": self.test_date,
            "batch": self.batch,
            "teaching_department": self.teaching_department,
            "academic_year": self.academic_year,
            "iat_number": self.iat_number,
        }


@dataclass
class Assessment:
    """Complete generated assessment: metadata + questions.

    Attributes:
        metadata: Descriptive envelope for the assessment.
        questions: Ordered list of :class:`Question` instances.
        generation_notes: Free-text notes from the generation agent.
    """

    metadata: AssessmentMetadata
    questions: List[Question] = field(default_factory=list)
    generation_notes: str = ""

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def total_marks(self) -> int:
        """Compute total marks by summing individual question marks.

        Returns:
            int: Sum of marks across all questions.
        """
        return sum(q.marks for q in self.questions)

    @property
    def question_count(self) -> int:
        """Return the number of questions in this assessment.

        Returns:
            int: Number of :class:`Question` objects.
        """
        return len(self.questions)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation.

        Returns:
            Dict with nested to_dict() calls for all child objects.
        """
        return {
            "metadata": self.metadata.to_dict(),
            "questions": [q.to_dict() for q in self.questions],
            "generation_notes": self.generation_notes,
        }


# ---------------------------------------------------------------------------
# Reviewer result model
# ---------------------------------------------------------------------------

@dataclass
class ReviewerResult:
    """Typed result returned by the Reviewer Agent (LLM-backed).

    Holds the AI-reviewed quality assessment of a generated paper.
    All fields have safe defaults so that partial/failed reviews can
    still be deserialised and displayed without crashing.

    Attributes:
        quality_score: LLM-assigned quality score (0.0–100.0).
        strengths: List of identified strengths in the assessment.
        weaknesses: List of identified weaknesses or gaps.
        suggestions: Concrete improvement recommendations.
        duplicate_question_ids: Question IDs suspected to be duplicates.
        bloom_coverage: Mapping of BloomLevel value → bool (True = covered).
        difficulty_balance_ok: True when Easy/Medium/Hard is reasonably spread.
        reviewer_notes: Free-text notes from the reviewer LLM.
        error: Set to a non-empty string when the LLM call or JSON parse
            failed; callers should check this before trusting quality_score.
    """

    quality_score: float = 0.0
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    duplicate_question_ids: List[str] = field(default_factory=list)
    bloom_coverage: Dict[str, bool] = field(default_factory=dict)
    difficulty_balance_ok: bool = True
    reviewer_notes: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation.

        Returns:
            Dict suitable for json.dumps() — no enum values to convert.
        """
        return {
            "quality_score": self.quality_score,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "suggestions": self.suggestions,
            "duplicate_question_ids": self.duplicate_question_ids,
            "bloom_coverage": self.bloom_coverage,
            "difficulty_balance_ok": self.difficulty_balance_ok,
            "reviewer_notes": self.reviewer_notes,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Analytics container
# ---------------------------------------------------------------------------

@dataclass
class BloomDistribution:
    """Per-level question count for Bloom's Taxonomy coverage.

    Attributes:
        counts: Mapping of BloomLevel → number of questions at that level.
    """

    counts: Dict[str, int] = field(default_factory=dict)

    def coverage_percent(self) -> Dict[str, float]:
        """Return percentage coverage per Bloom level.

        Returns:
            Dict mapping level name to percentage (0–100).
        """
        total = sum(self.counts.values()) or 1
        return {k: round(v / total * 100, 1) for k, v in self.counts.items()}

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation.

        Returns:
            Dict with ``counts`` and ``coverage_percent`` sub-dicts.
        """
        return {
            "counts": self.counts,
            "coverage_percent": self.coverage_percent(),
        }


@dataclass
class AnalyticsReport:
    """Analytics container for an :class:`Assessment`.

    Automatically calculated by the Analytics Agent (deterministic).
    The :attr:`reviewer_result` field is optionally populated by the
    Reviewer Agent and attached by the orchestration layer.

    Attributes:
        question_count: Total number of questions.
        total_marks: Aggregated marks.
        bloom_distribution: Bloom-level question counts.
        co_coverage: Mapping of CO code → number of questions covering it.
        difficulty_distribution: Mapping of DifficultyLevel → count.
        knowledge_sources_used: Distinct document names referenced.
        estimated_time_saved_minutes: Estimated faculty prep time saved.
        quality_score: Deterministic analytics quality score (0–100).
            Computed from four 25-point sub-scores: Bloom breadth,
            CO coverage, difficulty balance, and answer-key completeness.
        marks_distribution: Mapping of question_id → marks.
        reviewer_result: Optional result from the LLM Reviewer Agent.
            ``None`` when the reviewer has not been run or failed.
    """

    question_count: int = 0
    total_marks: int = 0
    bloom_distribution: BloomDistribution = field(
        default_factory=BloomDistribution
    )
    co_coverage: Dict[str, int] = field(default_factory=dict)
    difficulty_distribution: Dict[str, int] = field(default_factory=dict)
    knowledge_sources_used: List[str] = field(default_factory=list)
    estimated_time_saved_minutes: int = 0
    quality_score: float = 0.0
    marks_distribution: Dict[str, int] = field(default_factory=dict)
    reviewer_result: Optional[ReviewerResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation.

        Returns:
            Dict with nested to_dict() calls for child objects.
        """
        return {
            "question_count": self.question_count,
            "total_marks": self.total_marks,
            "bloom_distribution": self.bloom_distribution.to_dict(),
            "co_coverage": self.co_coverage,
            "difficulty_distribution": self.difficulty_distribution,
            "knowledge_sources_used": self.knowledge_sources_used,
            "estimated_time_saved_minutes": self.estimated_time_saved_minutes,
            "quality_score": self.quality_score,
            "marks_distribution": self.marks_distribution,
            "reviewer_result": (
                self.reviewer_result.to_dict()
                if self.reviewer_result is not None
                else None
            ),
        }


# ---------------------------------------------------------------------------
# Pipeline orchestration models (v4.3)
# ---------------------------------------------------------------------------


@dataclass
class AssessmentPlan:
    """Structured assessment plan produced by the Planning Agent.

    Replaces the anonymous ``dict`` that previously flowed between
    ``PlanningAgent`` and ``AssessmentAgent``.  Every field has an explicit
    type so callers receive clear AttributeError rather than silent
    ``dict.get()`` misses.

    Attributes:
        assessment_type: OBE category string (e.g. "Quiz").
        course_name: Full course name.
        course_code: Institutional course code (empty string if absent).
        topics: Comma-separated list of topics to cover.
        bloom_targets: Comma-separated target Bloom levels.
        co_mapping: Comma-separated Course Outcome codes.
        question_count: Number of questions to generate.
        marks_per_question: Marks allocated per question.
        difficulty: Difficulty target string ("Easy"|"Medium"|"Hard"|"Mixed").
        duration_minutes: Allotted time in minutes (0 = flexible).
        department: Offering department name (empty string if absent).
        semester: Semester label (empty string if absent).
        faculty_name: Faculty member name (empty string if absent).
        extra_instructions: Additional faculty-supplied instructions.
    """

    assessment_type: str
    course_name: str
    course_code: str
    topics: str
    bloom_targets: str
    co_mapping: str
    question_count: int
    marks_per_question: int
    difficulty: str
    duration_minutes: int
    department: str
    semester: str
    faculty_name: str
    extra_instructions: str
    test_date: str = ""
    vtu_marks_blueprint: str = ""
    """Optional faculty-authored custom marks pattern for VTU-style
    Semester Examinations. One line per topic:
    "<topic name>: <Q-A marks csv> | <Q-B marks csv>", e.g.
    "Autoencoders: 5,5,10 | 5,5,10". Topics not mentioned here fall back
    to the normal uniform question_count/marks_per_question generation.
    Empty string (the default) means no custom blueprint — fully
    backward compatible with existing behaviour."""
    batch: str = ""
    teaching_department: str = ""
    academic_year: str = ""
    iat_number: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation.

        Returns:
            Dict with all fields as primitives.
        """
        return {
            "assessment_type": self.assessment_type,
            "course_name": self.course_name,
            "course_code": self.course_code,
            "topics": self.topics,
            "bloom_targets": self.bloom_targets,
            "co_mapping": self.co_mapping,
            "question_count": self.question_count,
            "marks_per_question": self.marks_per_question,
            "difficulty": self.difficulty,
            "duration_minutes": self.duration_minutes,
            "department": self.department,
            "semester": self.semester,
            "faculty_name": self.faculty_name,
            "extra_instructions": self.extra_instructions,
            "test_date": self.test_date,
            "vtu_marks_blueprint": self.vtu_marks_blueprint,
            "batch": self.batch,
            "teaching_department": self.teaching_department,
            "academic_year": self.academic_year,
            "iat_number": self.iat_number,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssessmentPlan":
        """Construct an :class:`AssessmentPlan` from a raw dict.

        Applies the same defaults and type coercions as
        ``PlanningAgent._normalise_plan()`` so that both the LLM path and
        the direct-dict (form) path produce identical objects.

        Args:
            data: Raw dict, e.g. from the Streamlit generation form or an
                LLM JSON response.

        Returns:
            AssessmentPlan: Normalised, fully-populated plan object.
        """
        def _int(val: Any, default: int) -> int:
            try:
                return int(val)
            except (TypeError, ValueError):
                return default

        return cls(
            assessment_type=str(
                data.get("assessment_type", "Internal Assessment")
            ).strip(),
            course_name=str(data.get("course_name", "")).strip(),
            course_code=str(data.get("course_code", "")).strip(),
            topics=str(data.get("topics", "")).strip(),
            bloom_targets=str(
                data.get("bloom_targets", "Remember, Understand, Apply")
            ).strip(),
            co_mapping=str(data.get("co_mapping", "CO1")).strip(),
            question_count=_int(data.get("question_count"), 5),
            marks_per_question=_int(data.get("marks_per_question"), 5),
            difficulty=str(data.get("difficulty", "Mixed")).strip(),
            duration_minutes=_int(data.get("duration_minutes"), 0),
            department=str(data.get("department", "")).strip(),
            semester=str(data.get("semester", "")).strip(),
            faculty_name=str(data.get("faculty_name", "")).strip(),
            extra_instructions=str(data.get("extra_instructions", "")).strip(),
            test_date=str(data.get("test_date", "")).strip(),
            vtu_marks_blueprint=str(
                data.get("vtu_marks_blueprint", "")
            ).strip(),
            batch=str(data.get("batch", "")).strip(),
            teaching_department=str(
                data.get("teaching_department", "")
            ).strip(),
            academic_year=str(data.get("academic_year", "")).strip(),
            iat_number=str(data.get("iat_number", "")).strip(),
        )


@dataclass
class StageStatus:
    """Per-stage execution status for the orchestration pipeline.

    Captured for every stage whether it succeeds, is skipped, or fails.
    Partial progress is preserved: a failed non-fatal stage does not
    prevent downstream stages from recording their own status.

    Attributes:
        stage: Stage identifier — one of ``"planning"``, ``"rag"``,
            ``"generation"``, ``"analytics"``, ``"reviewer"``,
            ``"export_prep"``.
        status: Execution outcome — ``"ok"``, ``"skipped"``, or ``"error"``.
        duration_ms: Wall-clock duration of the stage in milliseconds.
            ``None`` when timing was not captured (e.g. early abort).
        error: Non-empty string describing a hard error that occurred in
            this stage.  ``None`` when status is ``"ok"`` or ``"skipped"``.
        warning: Non-empty string describing a degraded-but-continued
            condition (e.g. RAG returning zero chunks).  ``None`` when
            there is nothing to warn about.
    """

    stage: str
    status: str
    duration_ms: Optional[int] = None
    error: Optional[str] = None
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation.

        Returns:
            Dict with all fields as JSON primitives.
        """
        return {
            "stage": self.stage,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "warning": self.warning,
        }


@dataclass
class PipelineResult:
    """Consolidated result object returned by :class:`orchestrator.PipelineOrchestrator`.

    Carries every artefact produced by the pipeline: the plan, the
    generated assessment, analytics, reviewer report, export bytes, and
    per-stage status records.  ``export_bytes`` holds raw binary content
    and is excluded from :meth:`to_dict` (use ``export_formats`` instead).

    Attributes:
        run_id: UUID4 string uniquely identifying this pipeline run.
        started_at: ISO-8601 UTC timestamp of pipeline start.
        completed_at: ISO-8601 UTC timestamp of pipeline end.  ``None``
            only when the result is constructed before completion (rare).
        plan: The :class:`AssessmentPlan` produced by the Planning Agent.
            ``None`` only when planning itself fails fatally.
        assessment: The generated :class:`Assessment`.  ``None`` when
            the generation stage failed.
        analytics: The :class:`AnalyticsReport` computed by the Analytics
            Agent.  ``None`` when analytics failed.
        reviewer: The :class:`ReviewerResult` from the Reviewer Agent.
            ``None`` when the reviewer stage was skipped or failed.
        export_bytes: Mapping of format name → raw bytes for each
            successfully exported format (``"markdown"``, ``"docx"``,
            ``"pdf"``).  Empty dict when export_prep failed entirely.
        rag_sources: :class:`SourceAttribution` objects retrieved by the
            RAG stage.  Empty list when the knowledge base is empty or
            retrieval failed.
        stages: Ordered list of :class:`StageStatus` records, one per
            pipeline stage.
        success: ``True`` when an ``Assessment`` was produced (i.e.
            ``assessment is not None``).  Non-fatal stage failures
            (analytics, reviewer, export) do not flip this flag.
        fatal_error: Human-readable description of the error that caused
            an early pipeline abort.  ``None`` on success.
    """

    run_id: str
    started_at: str
    completed_at: Optional[str]
    plan: Optional["AssessmentPlan"]
    assessment: Optional[Assessment]
    analytics: Optional[AnalyticsReport]
    reviewer: Optional[ReviewerResult]
    export_bytes: Dict[str, bytes]
    rag_sources: List[SourceAttribution]
    stages: List[StageStatus]
    success: bool
    fatal_error: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation.

        ``export_bytes`` is excluded; ``export_formats`` lists the keys and
        ``export_sizes`` records each format's byte length so callers can
        report what was produced without serialising binary data.

        Returns:
            Dict suitable for ``json.dumps()``.
        """
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "assessment": (
                self.assessment.to_dict() if self.assessment is not None else None
            ),
            "analytics": (
                self.analytics.to_dict() if self.analytics is not None else None
            ),
            "reviewer": (
                self.reviewer.to_dict() if self.reviewer is not None else None
            ),
            "export_formats": list(self.export_bytes.keys()),
            "export_sizes": {
                fmt: len(data) for fmt, data in self.export_bytes.items()
            },
            "rag_sources": [s.to_dict() for s in self.rag_sources],
            "stages": [s.to_dict() for s in self.stages],
            "success": self.success,
            "fatal_error": self.fatal_error,
        }


@dataclass
class RunRecord:
    """Lightweight summary of a pipeline run for the history index.

    Stored in ``runs/index.json`` (one entry per run, newest-first).
    Omits question texts, answer keys, and export bytes so the index
    stays compact for dashboard list views.

    Attributes:
        run_id: UUID4 string uniquely identifying the run.
        started_at: ISO-8601 UTC timestamp of pipeline start.
        completed_at: ISO-8601 UTC timestamp of pipeline end.
        course_name: Course name from the plan.
        course_code: Course code from the plan.
        assessment_type: Assessment type string from the plan.
        question_count: Number of questions generated (0 on failure).
        total_marks: Sum of question marks (0 on failure).
        quality_score: Deterministic analytics quality score (0.0 on failure).
        reviewer_score: LLM reviewer quality score.  ``None`` when the
            reviewer did not run or returned an error.
        stages: Per-stage status records.
        success: ``True`` when an Assessment was produced.
        fatal_error: Early-abort error description.  ``None`` on success.
    """

    run_id: str
    started_at: str
    completed_at: Optional[str]
    course_name: str
    course_code: str
    assessment_type: str
    question_count: int
    total_marks: int
    quality_score: float
    reviewer_score: Optional[float]
    stages: List[StageStatus]
    success: bool
    fatal_error: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation.

        Returns:
            Dict suitable for ``json.dumps()``.
        """
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "course_name": self.course_name,
            "course_code": self.course_code,
            "assessment_type": self.assessment_type,
            "question_count": self.question_count,
            "total_marks": self.total_marks,
            "quality_score": self.quality_score,
            "reviewer_score": self.reviewer_score,
            "stages": [s.to_dict() for s in self.stages],
            "success": self.success,
            "fatal_error": self.fatal_error,
        }

    @classmethod
    def from_pipeline_result(cls, result: "PipelineResult") -> "RunRecord":
        """Build a :class:`RunRecord` from a completed :class:`PipelineResult`.

        Extracts only the summary fields needed for the history index;
        large fields (questions, answer keys, exports) are omitted.

        Args:
            result: The completed pipeline result to summarise.

        Returns:
            RunRecord: Populated lightweight summary.
        """
        plan = result.plan
        assessment = result.assessment
        analytics = result.analytics
        reviewer = result.reviewer

        return cls(
            run_id=result.run_id,
            started_at=result.started_at,
            completed_at=result.completed_at,
            course_name=plan.course_name if plan else "",
            course_code=plan.course_code if plan else "",
            assessment_type=plan.assessment_type if plan else "",
            question_count=assessment.question_count if assessment else 0,
            total_marks=assessment.total_marks if assessment else 0,
            quality_score=analytics.quality_score if analytics else 0.0,
            reviewer_score=(
                reviewer.quality_score
                if reviewer is not None and reviewer.error is None
                else None
            ),
            stages=result.stages,
            success=result.success,
            fatal_error=result.fatal_error,
        )
