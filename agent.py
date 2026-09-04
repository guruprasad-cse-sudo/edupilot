"""
EduPilot AI Faculty Assistant
Module: agent.py
Version: 4.3.0
Author: EduPilot Team
Purpose: Multi-agent orchestration layer.

         Pipeline agents (v4.3):
           PlanningAgent.plan()        — parse faculty requirement → AssessmentPlan
           AssessmentAgent.generate()  — RAG context + plan → Assessment dataclass
           AnalyticsAgent.analyse()    — deterministic metrics → AnalyticsReport
           ReviewerAgent.review()      — LLM quality audit → ReviewerResult
           DownloadAgent.export()      — delegates to downloads.DownloadEngine

         v4.3 changes:
           - PlanningAgent.plan() now returns AssessmentPlan (typed dataclass)
             instead of a plain dict.  _normalise_plan() updated accordingly.
           - AssessmentAgent.generate() now accepts AssessmentPlan instead of
             dict; all plan.get() accesses replaced with typed attribute access.
           - DownloadAgent.export() fully wired to downloads.DownloadEngine;
             no longer raises NotImplementedError.
           - All results are JSON-serialisable via to_dict().
"""

from __future__ import annotations

import difflib
import json
import math
import re
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import config, ConfigurationError
from logging_utils import get_logger
from models import (
    Assessment,
    AssessmentMetadata,
    AssessmentPlan,
    AssessmentType,
    AnalyticsReport,
    BloomDistribution,
    BloomLevel,
    DifficultyLevel,
    Question,
    ReviewerResult,
    SourceAttribution,
    VTU_MARKS_PER_FULL_QUESTION,
    VTU_MAX_MODULES,
    VTU_MAX_TOTAL_MARKS,
    split_sizes_for_pairing,
)
from prompts import (
    ASSESSMENT_SYSTEM_PROMPT,
    PLANNING_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    build_assessment_prompt,
    build_planning_prompt,
    build_reviewer_prompt,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers — string / enum parsing
# ---------------------------------------------------------------------------

# Mapping from raw LLM strings → enum values (case-insensitive)
_BLOOM_MAP: Dict[str, BloomLevel] = {
    "remember": BloomLevel.REMEMBER,
    "understand": BloomLevel.UNDERSTAND,
    "apply": BloomLevel.APPLY,
    "analyse": BloomLevel.ANALYZE,
    "analyze": BloomLevel.ANALYZE,
    "evaluate": BloomLevel.EVALUATE,
    "create": BloomLevel.CREATE,
}

_DIFFICULTY_MAP: Dict[str, DifficultyLevel] = {
    "easy": DifficultyLevel.EASY,
    "medium": DifficultyLevel.MEDIUM,
    "hard": DifficultyLevel.HARD,
}

_ASSESSMENT_TYPE_MAP: Dict[str, AssessmentType] = {
    "internal assessment": AssessmentType.INTERNAL,
    "internal": AssessmentType.INTERNAL,
    "quiz": AssessmentType.QUIZ,
    "assignment": AssessmentType.ASSIGNMENT,
    "semester examination": AssessmentType.SEMESTER_EXAM,
    "semester exam": AssessmentType.SEMESTER_EXAM,
    "viva": AssessmentType.VIVA,
    "role play": AssessmentType.ROLE_PLAY,
    "roleplay": AssessmentType.ROLE_PLAY,
    "role-play": AssessmentType.ROLE_PLAY,
    "question bank": AssessmentType.QUESTION_BANK,
    "question-bank": AssessmentType.QUESTION_BANK,
    "qb": AssessmentType.QUESTION_BANK,
}


def _parse_bloom(raw: str) -> BloomLevel:
    """Convert a raw Bloom-level string to a :class:`BloomLevel` enum.

    Args:
        raw: Case-insensitive Bloom level string from LLM output.

    Returns:
        BloomLevel: Matched enum value, defaulting to UNDERSTAND if unknown.
    """
    return _BLOOM_MAP.get(raw.strip().lower(), BloomLevel.UNDERSTAND)


def _parse_difficulty(raw: str) -> DifficultyLevel:
    """Convert a raw difficulty string to a :class:`DifficultyLevel` enum.

    Args:
        raw: Case-insensitive difficulty string from LLM output.

    Returns:
        DifficultyLevel: Matched enum value, defaulting to MEDIUM if unknown.
    """
    return _DIFFICULTY_MAP.get(raw.strip().lower(), DifficultyLevel.MEDIUM)


def _parse_assessment_type(raw: str) -> AssessmentType:
    """Convert a raw assessment-type string to an :class:`AssessmentType` enum.

    Args:
        raw: Case-insensitive assessment type string.

    Returns:
        AssessmentType: Matched enum value, defaulting to INTERNAL if unknown.
    """
    return _ASSESSMENT_TYPE_MAP.get(raw.strip().lower(), AssessmentType.INTERNAL)


def _extract_json(text: str) -> str:
    """Extract a JSON object or array from a potentially noisy LLM response.

    Strips markdown code fences and any leading/trailing prose so the caller
    can safely call ``json.loads()`` on the result.

    Args:
        text: Raw LLM response text.

    Returns:
        str: Extracted JSON string (may still be invalid JSON).
    """
    # Remove ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    # Find the outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return text[start : end + 1]

    # Fallback: return the stripped text and let json.loads() fail naturally
    return text.strip()


def _get_llm_client(temperature: float = 0.3) -> Any:
    """Instantiate a ChatGroq client using the configured API key and model.

    Args:
        temperature: Sampling temperature for the LLM (default 0.3).

    Returns:
        ChatGroq: Configured LangChain Groq chat client.

    Raises:
        ConfigurationError: When GROQ_API_KEY is absent.
        ImportError: When langchain-groq is not installed.
    """
    api_key = config.require_groq_api_key()
    try:
        from langchain_groq import ChatGroq  # lazy import
    except ImportError as exc:
        raise ImportError(
            "langchain-groq is not installed.  "
            "Run: pip install langchain-groq"
        ) from exc

    return ChatGroq(
        api_key=api_key,
        model_name=config.groq_model_name,
        temperature=temperature,
        # Output budget is configurable (GROQ_MAX_TOKENS). Too low truncates
        # long assessments into malformed JSON; too high trips Groq's
        # free-tier TPM limit, which counts the requested budget up front.
        max_tokens=config.groq_max_tokens,
    )


def _invoke_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
) -> str:
    """Call the Groq LLM with a system + user message pair.

    Args:
        system_prompt: System-role instruction text.
        user_prompt: User-turn prompt text.
        temperature: Sampling temperature (default 0.3).

    Returns:
        str: Raw model response content.

    Raises:
        ConfigurationError: When GROQ_API_KEY is absent.
        RuntimeError: On any LLM API or network failure.
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage  # lazy import
    except ImportError as exc:
        raise ImportError(
            "langchain-core is not installed.  "
            "Run: pip install langchain-core"
        ) from exc

    llm = _get_llm_client(temperature=temperature)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    logger.debug(
        "Invoking LLM model=%s temperature=%.2f", config.groq_model_name, temperature
    )
    try:
        response = llm.invoke(messages)
        content: str = response.content
        logger.debug("LLM response length=%d chars", len(content))
        return content
    except Exception as exc:
        logger.error("LLM invocation failed: %s", exc)
        raise RuntimeError(f"Groq LLM call failed: {exc}") from exc


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True when *exc* looks like a Groq TPM / rate-limit error.

    Groq surfaces these as HTTP 413 or HTTP 429 inside a RuntimeError whose
    message contains "413", "429", "rate_limit", or "too many" (case-
    insensitive).  We match broadly so future Groq SDK wording changes are
    caught without a code update.

    Args:
        exc: The exception to inspect.

    Returns:
        bool: True when the error is a retryable rate-limit condition.
    """
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("413", "429", "rate_limit", "rate limit", "too many")
    )


def _invoke_llm_with_rate_retry(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_rate_retries: int = 2,
    rate_retry_delay_s: float = 65.0,
    on_call_start: Optional[Callable[[], None]] = None,
) -> str:
    """Call the Groq LLM, retrying on rate-limit (413/429) errors.

    Non-rate-limit RuntimeErrors and ConfigurationErrors propagate
    immediately without retrying.

    Args:
        system_prompt: System-role instruction text.
        user_prompt: User-turn prompt text.
        temperature: Sampling temperature (default 0.3).
        max_rate_retries: Maximum number of retry attempts on rate-limit
            errors (default 2, giving 3 total attempts).
        rate_retry_delay_s: Seconds to sleep before each retry (linear
            delay × attempt number, default 65 s so the Groq 60 s rolling
            TPM window has fully reset).
        on_call_start: Optional zero-arg callable invoked immediately before
            each physical LLM request (including retries).  Used by batched
            generation to timestamp the true start of the most recent call
            for TPM cooldown accounting.  Exceptions it raises are swallowed.

    Returns:
        str: Raw model response content.

    Raises:
        ConfigurationError: When GROQ_API_KEY is absent.
        RuntimeError: When all attempts are exhausted or a non-rate-limit
            error occurs.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_rate_retries + 1):
        try:
            if on_call_start is not None:
                try:
                    on_call_start()
                except Exception:  # noqa: BLE001 — never let tracking break a call
                    pass
            return _invoke_llm(system_prompt, user_prompt, temperature)
        except RuntimeError as exc:
            if _is_rate_limit_error(exc) and attempt < max_rate_retries:
                wait = rate_retry_delay_s * (attempt + 1)
                logger.warning(
                    "Rate-limit hit on LLM call (attempt %d/%d); "
                    "sleeping %.0f s before retry.",
                    attempt + 1,
                    max_rate_retries + 1,
                    wait,
                )
                time.sleep(wait)
                last_exc = exc
            else:
                raise  # non-rate-limit error or retries exhausted → propagate
        except ConfigurationError:
            raise  # key errors are never retriable

    # Should be unreachable, but satisfies type checkers.
    raise RuntimeError(
        f"Rate-limit retries exhausted after {max_rate_retries + 1} attempts. "
        f"Last error: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Per-type token-budget estimates for batching decisions
# ---------------------------------------------------------------------------

# Estimated output tokens consumed per question for each assessment type.
# Higher values reflect longer answer keys / marking schemes.
_TOKENS_PER_QUESTION: Dict[str, int] = {
    "Question Bank":        350,
    "Assignment":           300,
    "Semester Examination": 250,
    "Role Play":            400,
    "Viva":                 220,
    "Internal Assessment":  220,
    "Quiz":                 150,
}

# Approximate prompt overhead (system + user prompt, excluding RAG context)
# added to each LLM call.  Conservative estimate; real usage varies.
_PROMPT_OVERHEAD_TOKENS: int = 600

# Fraction of groq_max_tokens reserved for output; the rest is treated as
# a safety margin so we never push right up to the limit.
_OUTPUT_BUDGET_FRACTION: float = 0.85


# ---------------------------------------------------------------------------
# Shared pure-function helpers (analytics + reviewer)
# ---------------------------------------------------------------------------


def parse_vtu_marks_blueprint(
    blueprint_text: str,
) -> Dict[str, Tuple[List[int], List[int]]]:
    """Parse a faculty-authored custom marks blueprint into a lookup dict.

    Expected format, one line per topic::

        <topic name>: <Q-A sub-part marks, comma-separated> | <Q-B sub-part marks>

    e.g. ``"Autoencoders: 5,5,10 | 5,5,10"``. Blank lines, lines without a
    colon, or lines that fail to parse cleanly are skipped rather than
    raising — this is faculty-typed free text, not a strict config file,
    so a typo in one line should not break every other topic's blueprint
    or fall back to fully automatic generation for the whole assessment.

    Args:
        blueprint_text: Raw textarea contents from the Generate form.
            Empty string (the default / no blueprint given) yields ``{}``.

    Returns:
        Dict mapping topic name (as typed by the faculty) to a
        ``(marks_for_q_a, marks_for_q_b)`` tuple of integer lists.
    """
    result: Dict[str, Tuple[List[int], List[int]]] = {}
    if not blueprint_text or not blueprint_text.strip():
        return result

    for raw_line in blueprint_text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        topic_part, _, marks_part = line.partition(":")
        topic = topic_part.strip()
        if not topic:
            continue
        sides = marks_part.split("|")
        if len(sides) != 2:
            logger.warning(
                "parse_vtu_marks_blueprint(): skipping malformed line "
                "(expected one '|' separating Q-A/Q-B marks): %r",
                line,
            )
            continue
        try:
            marks_a = [
                int(m.strip()) for m in sides[0].split(",") if m.strip()
            ]
            marks_b = [
                int(m.strip()) for m in sides[1].split(",") if m.strip()
            ]
        except ValueError:
            logger.warning(
                "parse_vtu_marks_blueprint(): skipping line with "
                "non-integer marks: %r", line,
            )
            continue
        if not marks_a or not marks_b:
            logger.warning(
                "parse_vtu_marks_blueprint(): skipping line with an "
                "empty marks list: %r", line,
            )
            continue
        result[topic] = (marks_a, marks_b)

    return result


def _depth_guidance_for_marks(marks: int) -> str:
    """Return depth/scope guidance text for a sub-part worth *marks* marks.

    Used to tell the LLM how substantial an answer should be for each
    individually marked sub-part in a custom VTU marks blueprint, so a
    10-mark sub-part reads as visibly more developed than a 3-mark one
    rather than both getting generic, similarly-sized answers.

    Args:
        marks: Marks allocated to this specific sub-part.

    Returns:
        str: A short scope-guidance phrase for the prompt.
    """
    if marks <= 3:
        return (
            "a very brief, direct answer — a definition, a one-line fact, "
            "or a short list (roughly 2-3 sentences)"
        )
    if marks <= 6:
        return (
            "a short-to-moderate explanation — a few sentences, or one "
            "simple worked example"
        )
    if marks <= 10:
        return (
            "a detailed explanation covering multiple points, a full "
            "worked example, or a multi-step derivation"
        )
    return (
        "an in-depth, comprehensive answer covering multiple sub-aspects "
        "in detail — e.g. an extended derivation, multi-part case "
        "analysis, or thorough design discussion"
    )


def _auto_marks_split(n: int, total: int) -> List[int]:
    """Split *total* marks across *n* sub-parts as evenly as possible.

    Used to auto-derive a marks blueprint for a VTU Semester Examination
    topic that the faculty did NOT manually customize — every full
    question still ends up totalling exactly ``total`` marks (per
    ``models.VTU_MARKS_PER_FULL_QUESTION``), just with an even split
    instead of a faculty-chosen one.

    Args:
        n: Number of sub-parts to split marks across.
        total: Total marks to distribute (must sum exactly across parts).

    Returns:
        List of *n* positive integers summing to *total*, e.g.
        ``_auto_marks_split(3, 20)`` → ``[7, 7, 6]``.
    """
    if n <= 0:
        return []
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def _normalise_text(text: str) -> str:
    """Normalise a question text for duplicate comparison.

    Lowercases, collapses whitespace, and strips punctuation so that
    paraphrase-only differences do not affect the similarity ratio.

    Args:
        text: Raw question text.

    Returns:
        str: Normalised string suitable for SequenceMatcher comparison.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # strip punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _find_duplicate_questions(
    questions: List[Question],
    threshold: float = 0.70,
) -> List[str]:
    """Detect syntactically similar question pairs using SequenceMatcher.

    Compares all O(n²) pairs of normalised question texts.  Pairs whose
    ratio meets or exceeds *threshold* are flagged.  Returns the question
    IDs involved in at least one duplicate pair (deduped).

    Args:
        questions: List of :class:`~models.Question` objects to scan.
        threshold: Similarity ratio (0.0–1.0) above which questions are
            considered duplicates.  Default 0.70.

    Returns:
        List[str]: Question IDs that are part of a duplicate pair.
            Empty list when no duplicates are found.
    """
    if len(questions) < 2:
        return []

    duplicate_ids: set[str] = set()
    normalised = [_normalise_text(q.question_text) for q in questions]

    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            ratio = difflib.SequenceMatcher(
                None, normalised[i], normalised[j]
            ).ratio()
            if ratio >= threshold:
                duplicate_ids.add(questions[i].question_id)
                duplicate_ids.add(questions[j].question_id)
                logger.debug(
                    "Duplicate pair detected: %s ↔ %s (ratio=%.3f)",
                    questions[i].question_id,
                    questions[j].question_id,
                    ratio,
                )

    return sorted(duplicate_ids)


def _compute_time_saved(assessment: Assessment) -> int:
    """Estimate faculty preparation time saved (in minutes) by EduPilot.

    Heuristic formula (per question):
      - Base:                      3 minutes
      - Non-empty answer key:     +1 minute
      - Multiple CO mappings:     +0.5 minutes (len(co_mapping) > 1)
      - Higher-order Bloom level: +0.5 minutes (Evaluate or Create)

    Minimum return value is 10 minutes regardless of question count.

    Args:
        assessment: The :class:`~models.Assessment` whose questions are
            being analysed.

    Returns:
        int: Estimated minutes saved, rounded to the nearest integer.
            Returns 0 for an empty assessment.
    """
    if not assessment.questions:
        return 0

    total: float = 0.0
    for q in assessment.questions:
        minutes = 3.0
        if q.answer_key and q.answer_key.strip():
            minutes += 1.0
        if len(q.co_mapping) > 1:
            minutes += 0.5
        if q.bloom_level in (BloomLevel.EVALUATE, BloomLevel.CREATE):
            minutes += 0.5
        total += minutes

    return max(10, round(total))


def _compute_analytics_quality_score(assessment: Assessment) -> float:
    """Compute a deterministic analytics quality score (0–100).

    Four equally-weighted sub-scores (25 points each):

    1. Bloom breadth:
       ``(distinct Bloom levels used / 6) × 25``

    2. CO coverage:
       ``min(distinct CO codes covered × 5, 25)``
       Full marks at 5 or more distinct COs.

    3. Difficulty balance:
       Standard deviation of [easy%, medium%, hard%] normalised to 0–25.
       Perfect 33.3/33.3/33.3 distribution → 25 pts.
       All questions at one difficulty → 0 pts.
       Formula: ``25 × (1 − std_dev / 33.3)`` clamped to [0, 25].

    4. Answer-key completeness:
       ``(questions with non-empty answer_key / total_questions) × 25``

    Args:
        assessment: The :class:`~models.Assessment` to score.

    Returns:
        float: Overall quality score (0.0–100.0), rounded to one decimal
            place.  Returns 0.0 for an empty assessment.
    """
    questions = assessment.questions
    if not questions:
        return 0.0

    n = len(questions)

    # --- Sub-score 1: Bloom breadth ---
    distinct_bloom = len({q.bloom_level for q in questions})
    bloom_score = (distinct_bloom / 6) * 25.0

    # --- Sub-score 2: CO coverage ---
    all_cos: set[str] = set()
    for q in questions:
        all_cos.update(q.co_mapping)
    co_score = min(len(all_cos) * 5.0, 25.0)

    # --- Sub-score 3: Difficulty balance ---
    diff_counts: Dict[str, int] = {
        DifficultyLevel.EASY.value: 0,
        DifficultyLevel.MEDIUM.value: 0,
        DifficultyLevel.HARD.value: 0,
    }
    for q in questions:
        diff_counts[q.difficulty.value] = diff_counts.get(q.difficulty.value, 0) + 1

    percentages = [count / n * 100.0 for count in diff_counts.values()]
    mean_pct = sum(percentages) / len(percentages)  # always 33.33…
    variance = sum((p - mean_pct) ** 2 for p in percentages) / len(percentages)
    std_dev = math.sqrt(variance)
    # Perfect balance → std_dev = 0 → diff_score = 25; worst case → 33.3
    diff_score = max(0.0, 25.0 * (1.0 - std_dev / 33.3))

    # --- Sub-score 4: Answer-key completeness ---
    answered = sum(1 for q in questions if q.answer_key and q.answer_key.strip())
    key_score = (answered / n) * 25.0

    total = bloom_score + co_score + diff_score + key_score
    return round(min(total, 100.0), 1)


# ---------------------------------------------------------------------------
# Planning Agent
# ---------------------------------------------------------------------------


class PlanningAgent:
    """Parses faculty requirements and produces a structured assessment plan.

    The Planning Agent is the entry point of the pipeline.  It interprets a
    natural-language faculty request (or a pre-structured dict) and emits a
    normalised plan dict consumed by :class:`AssessmentAgent`.

    Attributes:
        use_llm: When True (default), uses Groq to parse free-text requests.
            When False, the input must already be a structured dict.
    """

    def __init__(self, use_llm: bool = True) -> None:
        """Initialise the PlanningAgent.

        Args:
            use_llm: Whether to call the LLM to parse natural-language input.
        """
        self.use_llm = use_llm
        logger.info("PlanningAgent initialised (use_llm=%s)", use_llm)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, faculty_request: str | dict) -> AssessmentPlan:
        """Convert a faculty requirement into a structured assessment plan.

        When *faculty_request* is already a dict (e.g. from the Streamlit
        generation form) the LLM call is skipped entirely — the dict is
        normalised directly into an :class:`~models.AssessmentPlan`.  When
        it is a natural-language string the Groq LLM is invoked to extract
        the structured fields.

        Args:
            faculty_request: Either a natural-language string describing the
                desired assessment, or an already-structured dict with keys
                matching the planning schema.

        Returns:
            AssessmentPlan: Typed, normalised plan consumed by
                :class:`AssessmentAgent`.

        Raises:
            ConfigurationError: When Groq API key is absent and use_llm=True.
            RuntimeError: On LLM failure.
            ValueError: When the input cannot be parsed into a valid plan.
        """
        if isinstance(faculty_request, dict):
            logger.info(
                "PlanningAgent.plan(): using pre-structured dict input — "
                "skipping LLM call"
            )
            return self._normalise_plan(faculty_request)

        logger.info(
            "PlanningAgent.plan(): parsing natural-language request (%d chars)",
            len(faculty_request),
        )

        if not self.use_llm:
            raise ValueError(
                "use_llm=False but a string request was provided.  "
                "Pass a structured dict or set use_llm=True."
            )

        raw_response = _invoke_llm(
            system_prompt=PLANNING_SYSTEM_PROMPT,
            user_prompt=build_planning_prompt(faculty_request),
        )
        return self._parse_plan_response(raw_response)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_plan_response(self, raw: str) -> AssessmentPlan:
        """Parse the LLM planning response into a normalised :class:`AssessmentPlan`.

        Args:
            raw: Raw LLM response text.

        Returns:
            AssessmentPlan: Normalised plan.

        Raises:
            ValueError: When JSON cannot be extracted or required keys are missing.
        """
        json_str = _extract_json(raw)
        try:
            data: dict = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.error(
                "PlanningAgent: failed to parse JSON from LLM response: %s\n%s",
                exc,
                raw[:500],
            )
            raise ValueError(
                f"Planning Agent returned malformed JSON: {exc}"
            ) from exc

        return self._normalise_plan(data)

    def _normalise_plan(self, data: dict) -> AssessmentPlan:
        """Apply defaults and type coercions to a raw plan dict.

        Delegates to :meth:`~models.AssessmentPlan.from_dict` so that the
        normalisation logic lives in exactly one place (the model).

        Args:
            data: Raw plan dict (from LLM JSON or direct structured input).

        Returns:
            AssessmentPlan: Typed, normalised plan with all fields populated.
        """
        plan = AssessmentPlan.from_dict(data)
        logger.debug("Normalised plan: %s", plan.to_dict())
        return plan


# ---------------------------------------------------------------------------
# Assessment Agent
# ---------------------------------------------------------------------------


class AssessmentAgent:
    """Generates a complete :class:`~models.Assessment` from a plan + context.

    Uses the Groq LLM via LangChain to produce a structured JSON assessment
    given a plan from :class:`PlanningAgent` and retrieved context from the
    RAG layer.  Parses the LLM response into the shared data model.

    For large question counts that would exceed the Groq output-token budget
    (``config.groq_max_tokens``, default 6000), generation is automatically
    split into batches of ``config.batch_size`` questions each.  Batch calls
    are spaced by ``config.batch_inter_call_delay_s`` seconds to avoid hitting
    Groq's free-tier 8000 TPM rolling-window limit.  Rate-limit errors (HTTP
    413/429) trigger exponential backoff retries.

    Attributes:
        max_retries: Number of JSON-parse retries per LLM call on malformed
            responses.  Does not apply to rate-limit errors (handled
            separately with sleeping retries).
    """

    def __init__(self, max_retries: int = 2) -> None:
        """Initialise the AssessmentAgent.

        Args:
            max_retries: How many times to retry the LLM call on a JSON parse
                failure before raising.
        """
        self.max_retries = max_retries
        self._vtu_blueprint: Dict[str, Tuple[List[int], List[int]]] = {}
        logger.info("AssessmentAgent initialised (max_retries=%d)", max_retries)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        plan: AssessmentPlan,
        rag_context: str,
        sources: Optional[List[SourceAttribution]] = None,
        batch_progress_callback: Optional[
            Callable[[int, int, int, int, int], None]
        ] = None,
    ) -> Assessment:
        """Generate a full assessment from the plan and RAG context.

        Dispatches to :meth:`_generate_batched` when the requested question
        count would likely exceed the Groq output-token budget, or to
        :meth:`_generate_single` otherwise.

        Args:
            plan: Typed :class:`~models.AssessmentPlan` from
                :class:`PlanningAgent`.
            rag_context: Concatenated retrieved text chunks from the RAG layer.
            sources: Source attribution objects from the RAG module.  Used to
                attach provenance to questions.
            batch_progress_callback: Optional callable invoked after each
                batch completes.  Signature:
                ``(batch_idx, total_batches, qs_start, qs_end, total_qs)``
                where all values are 1-based integers.  Exceptions inside
                the callback are swallowed.

        Returns:
            Assessment: Populated :class:`~models.Assessment` dataclass.

        Raises:
            ConfigurationError: When Groq API key is absent.
            RuntimeError: On LLM failure after all retries.
            ValueError: When the LLM response cannot be parsed into an
                Assessment, or when any batch fails after all retries.
        """
        logger.info(
            "AssessmentAgent.generate(): type=%s questions=%d",
            plan.assessment_type,
            plan.question_count,
        )

        # Semester Examinations always use the per-topic marks-blueprint
        # path now (custom or auto-derived — see _resolve_vtu_blueprint),
        # which is only wired into the batched generation path. Force
        # batching for every Semester Exam, and whenever any other
        # assessment type has a custom blueprint typed in, so it's never
        # silently ignored by the single-call path on a small assessment.
        is_semester_exam = (
            plan.assessment_type == AssessmentType.SEMESTER_EXAM.value
        )
        has_custom_blueprint = bool(
            parse_vtu_marks_blueprint(plan.vtu_marks_blueprint)
        )
        if is_semester_exam or has_custom_blueprint or self._needs_batching(plan):
            return self._generate_batched(
                plan=plan,
                rag_context=rag_context,
                sources=sources or [],
                batch_progress_callback=batch_progress_callback,
            )
        return self._generate_single(
            plan=plan,
            rag_context=rag_context,
            sources=sources or [],
        )

    # ------------------------------------------------------------------
    # Batching decision helpers
    # ------------------------------------------------------------------

    def _needs_batching(self, plan: AssessmentPlan) -> bool:
        """Return True when *plan* requires batched LLM generation.

        Uses a per-type token estimate to compute how many questions fit
        safely within the configured output budget, then compares against
        ``plan.question_count``.  ``config.batch_threshold`` acts as a hard
        upper cap: even if the token estimate would allow more questions in
        a single call, a question count above the threshold always batches.

        Args:
            plan: The assessment plan containing type and question count.

        Returns:
            bool: True when batching should be used.
        """
        tokens_per_q = _TOKENS_PER_QUESTION.get(
            plan.assessment_type, _TOKENS_PER_QUESTION["Internal Assessment"]
        )
        output_budget = int(config.groq_max_tokens * _OUTPUT_BUDGET_FRACTION)
        # Questions that fit comfortably in one call
        estimate_threshold = max(
            1,
            int((output_budget - _PROMPT_OVERHEAD_TOKENS) / tokens_per_q),
        )
        # config.batch_threshold is a hard cap: never allow more than this
        # in a single call regardless of the token estimate.
        effective_threshold = min(estimate_threshold, config.batch_threshold)
        needs = plan.question_count > effective_threshold
        logger.info(
            "AssessmentAgent._needs_batching(): type=%r tokens_per_q=%d "
            "estimate_threshold=%d config_threshold=%d effective=%d "
            "question_count=%d → %s",
            plan.assessment_type,
            tokens_per_q,
            estimate_threshold,
            config.batch_threshold,
            effective_threshold,
            plan.question_count,
            "BATCH" if needs else "SINGLE",
        )
        return needs

    def _allocate_per_topic(
        self, topics_str: str, total: int
    ) -> List[Tuple[str, int]]:
        """Proportionally allocate *total* questions across comma-separated topics.

        Uses the largest-remainder (Hamilton) method so that:
        - Every topic gets at least ``floor(total / n_topics)`` questions.
        - The first ``total % n_topics`` topics (by list order) each get one
          extra question to account for the remainder.
        - Topics that would receive 0 questions (when total < n_topics) are
          omitted from the result.

        This guarantees ``max_count − min_count ≤ 1`` for any set of topics
        and that the returned counts sum to exactly *total*.

        Args:
            topics_str: Comma-separated topic string from the plan.
            total: Total number of questions to distribute.

        Returns:
            List[Tuple[str, int]]: ``(topic_label, question_count)`` pairs
                for every topic that receives at least one question, in the
                original topic order.
        """
        raw = [t.strip() for t in topics_str.split(",") if t.strip()]
        if not raw:
            raw = ["General"]

        n = len(raw)
        base, remainder = divmod(total, n)
        allocations: List[Tuple[str, int]] = []
        for i, topic in enumerate(raw):
            count = base + (1 if i < remainder else 0)
            if count > 0:
                allocations.append((topic, count))

        logger.debug(
            "AssessmentAgent._allocate_per_topic(): total=%d topics=%d "
            "base=%d remainder=%d → %s",
            total,
            n,
            base,
            remainder,
            [(t[:30], c) for t, c in allocations],
        )
        return allocations

    def _resolve_vtu_blueprint(
        self, plan: AssessmentPlan
    ) -> Dict[str, Tuple[List[int], List[int]]]:
        """Resolve the complete per-topic marks blueprint for *plan*.

        Combines any faculty-typed custom blueprint entries
        (``plan.vtu_marks_blueprint``) with auto-derived ones. Auto-
        derivation only applies to Semester Examinations: every topic
        NOT covered by a custom entry still gets an automatically
        computed marks split, evenly dividing
        ``models.VTU_MARKS_PER_FULL_QUESTION`` (20) across however many
        sub-parts that topic's allocated question quota works out to —
        so every module in a Semester Exam paper ends up with a full
        question worth exactly 20 marks, whether the faculty customized
        that topic or not.

        Non-Semester-Exam assessment types only ever use faculty-typed
        custom entries (if any are present) and are otherwise returned
        with no auto-derived entries — this preserves the original,
        uniform marks_per_question generation behaviour for every other
        assessment type exactly as it was before this feature existed.

        Args:
            plan: The assessment plan.

        Returns:
            Dict mapping topic name -> ``(marks_for_q_a, marks_for_q_b)``.
        """
        custom = parse_vtu_marks_blueprint(plan.vtu_marks_blueprint)
        is_semester_exam = (
            plan.assessment_type == AssessmentType.SEMESTER_EXAM.value
        )
        if not is_semester_exam:
            return custom

        all_topics = [t.strip() for t in plan.topics.split(",") if t.strip()]
        if not all_topics:
            all_topics = ["General"]

        if len(all_topics) > VTU_MAX_MODULES:
            logger.warning(
                "AssessmentAgent._resolve_vtu_blueprint(): %d topics given "
                "but a %d-mark Semester Exam supports at most %d Modules "
                "(%d marks each) — the paper will exceed %d marks. "
                "Consider reducing the Topics field to %d entries.",
                len(all_topics), VTU_MAX_TOTAL_MARKS, VTU_MAX_MODULES,
                VTU_MARKS_PER_FULL_QUESTION, VTU_MAX_TOTAL_MARKS,
                VTU_MAX_MODULES,
            )

        resolved: Dict[str, Tuple[List[int], List[int]]] = dict(custom)
        auto_topics = [t for t in all_topics if t not in custom]
        if auto_topics:
            quota_by_topic = dict(
                self._allocate_per_topic(
                    ",".join(auto_topics), plan.question_count
                )
            )
            for topic in auto_topics:
                # Every Module needs an OR-pair, so floor the quota at 2
                # even if the proportional allocation would have given
                # this topic 0 or 1 — a small, documented deviation from
                # plan.question_count in exchange for a structurally
                # valid VTU paper (see class docstring note on totals).
                quota = max(quota_by_topic.get(topic, 2), 2)
                sizes = split_sizes_for_pairing(quota)
                marks_a = _auto_marks_split(
                    sizes[0], VTU_MARKS_PER_FULL_QUESTION
                )
                marks_b = (
                    _auto_marks_split(sizes[1], VTU_MARKS_PER_FULL_QUESTION)
                    if len(sizes) > 1
                    else list(marks_a)
                )
                resolved[topic] = (marks_a, marks_b)

        return resolved

    def _build_batch_specs(
        self, plan: AssessmentPlan
    ) -> List[Tuple[int, str]]:
        """Compute per-batch (question_count, topics) specifications.

        Every topic with a resolved VTU marks blueprint entry (custom or
        auto-derived — see :meth:`_resolve_vtu_blueprint`) gets an exact
        single batch sized to that blueprint's total sub-part count —
        never split across batches and never subject to proportional
        allocation, since its question count is fixed by the blueprint,
        not derived from ``plan.question_count``.

        Any remaining topics (only possible for non-Semester-Exam types,
        since Semester Exams get a blueprint entry for every topic) use
        the original proportional allocation: first allocated via
        :meth:`_allocate_per_topic` against ``plan.question_count``, then
        split into batches of at most ``config.batch_size`` questions.

        As a side effect, stashes the resolved blueprint on
        ``self._vtu_blueprint`` for the generation loop to consult.

        Args:
            plan: The assessment plan to split.

        Returns:
            List[Tuple[int, str]]: One ``(count, topic_string)`` pair per
                batch, blueprint topics first, then any remaining
                non-blueprint topics in original order.
        """
        batch_size = config.batch_size
        blueprint = self._resolve_vtu_blueprint(plan)
        self._vtu_blueprint = blueprint
        all_topics = [t.strip() for t in plan.topics.split(",") if t.strip()]
        if not all_topics:
            all_topics = ["General"]

        specs: List[Tuple[int, str]] = []
        for topic in all_topics:
            if topic in blueprint:
                marks_a, marks_b = blueprint[topic]
                specs.append((len(marks_a) + len(marks_b), topic))

        remaining_topics_str = ",".join(
            t for t in all_topics if t not in blueprint
        )
        if remaining_topics_str:
            per_topic = self._allocate_per_topic(
                remaining_topics_str, plan.question_count
            )
            for topic, quota in per_topic:
                # Split this topic's quota into ≤batch_size sub-batches
                remaining = quota
                while remaining > 0:
                    count = min(batch_size, remaining)
                    specs.append((count, topic))
                    remaining -= count

        num_batches = len(specs)
        logger.info(
            "AssessmentAgent._build_batch_specs(): plan_total=%d "
            "batch_size=%d blueprint_topics=%d num_batches=%d specs=%s",
            plan.question_count,
            batch_size,
            len(blueprint),
            num_batches,
            [(c, t[:40]) for c, t in specs],
        )
        return specs

    # ------------------------------------------------------------------
    # Single-call generation (original logic, refactored out)
    # ------------------------------------------------------------------

    def _generate_single(
        self,
        plan: AssessmentPlan,
        rag_context: str,
        sources: List[SourceAttribution],
    ) -> Assessment:
        """Generate all questions in one LLM call.

        This is the original generation path, used when the question count
        fits within a single output budget window.

        Args:
            plan: Typed assessment plan.
            rag_context: Retrieved context string.
            sources: RAG source attributions.

        Returns:
            Assessment: Fully parsed assessment.

        Raises:
            ValueError: After ``max_retries + 1`` consecutive JSON parse
                failures.
            RuntimeError: On LLM/network failure.
            ConfigurationError: When GROQ_API_KEY is absent.
        """
        logger.info(
            "AssessmentAgent._generate_single(): %d questions",
            plan.question_count,
        )
        user_prompt = build_assessment_prompt(
            assessment_type=plan.assessment_type,
            course_name=plan.course_name,
            course_code=plan.course_code,
            topics=plan.topics,
            bloom_targets=plan.bloom_targets,
            co_mapping=plan.co_mapping,
            rag_context=rag_context,
            question_count=plan.question_count,
            marks_per_question=plan.marks_per_question,
            difficulty=plan.difficulty,
            extra_instructions=plan.extra_instructions,
        )

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 2):
            logger.info(
                "AssessmentAgent._generate_single(): attempt %d/%d",
                attempt,
                self.max_retries + 1,
            )
            try:
                raw = _invoke_llm(
                    system_prompt=ASSESSMENT_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                )
                assessment = self._parse_assessment_response(
                    raw=raw,
                    plan=plan,
                    sources=sources,
                )
                logger.info(
                    "AssessmentAgent._generate_single(): %d questions OK",
                    assessment.question_count,
                )
                # Best-effort topic tagging: the single-call path requests
                # all topics in one LLM call, so there's no natural
                # per-question topic boundary the way batching provides.
                # Assign topics in the same proportional order used for
                # batch allocation so VTU-style Module grouping on export
                # still works reasonably for small, non-batched runs.
                self._tag_topics_best_effort(assessment.questions, plan.topics)
                return assessment
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_error = exc
                logger.warning(
                    "AssessmentAgent._generate_single(): parse error "
                    "attempt %d: %s",
                    attempt,
                    exc,
                )
            except (RuntimeError, ConfigurationError):
                raise

        raise ValueError(
            f"AssessmentAgent failed to produce valid JSON after "
            f"{self.max_retries + 1} attempts.  Last error: {last_error}"
        )

    def _tag_topics_best_effort(
        self, questions: List[Question], topics_str: str
    ) -> None:
        """Assign a topic to each question in proportional order.

        Used only by the single-call generation path, where all topics are
        requested together in one LLM call and there is no natural
        per-question topic boundary the way batching provides. Questions
        are tagged in the same proportional allocation order used for
        batching (see :meth:`_allocate_per_topic`), so VTU-style Module
        grouping on export still works reasonably for small, non-batched
        assessments. This is a best-effort heuristic, not a guarantee that
        question N is actually about the Nth allocated topic.

        Args:
            questions: The generated questions, in original order, to tag
                in place.
            topics_str: Comma-separated topic string from the plan.
        """
        if not questions:
            return
        allocations = self._allocate_per_topic(topics_str, len(questions))
        if not allocations:
            return
        idx = 0
        for topic, count in allocations:
            for _ in range(count):
                if idx >= len(questions):
                    return
                questions[idx].topic = topic
                idx += 1
        # Any leftover questions (shouldn't normally happen) fall back to
        # the last allocated topic rather than staying untagged.
        while idx < len(questions):
            questions[idx].topic = allocations[-1][0]
            idx += 1

    # ------------------------------------------------------------------
    # Batched generation
    # ------------------------------------------------------------------

    def _apply_blueprint_marks(
        self,
        questions: List[Question],
        blueprint_entry: Tuple[List[int], List[int]],
        topic: str,
    ) -> None:
        """Force each question's marks/group to match the faculty blueprint.

        The LLM is asked to honour the exact per-question marks list, but
        prompt compliance is never guaranteed — this makes the paper
        correct regardless, by overwriting whatever the LLM put in the
        ``marks`` field with the faculty-authored value at that position,
        and tagging ``blueprint_group`` ("A"/"B") so export can split the
        two OR-alternative questions precisely instead of guessing.

        If the LLM returned a different number of questions than
        requested (rare, but possible), this pads or truncates gracefully
        rather than raising — a slightly-off blueprint match is far
        better than crashing the whole generation run over it.

        Args:
            questions: This batch's generated questions, in order.
            blueprint_entry: ``(marks_for_q_a, marks_for_q_b)`` from
                :func:`parse_vtu_marks_blueprint`.
            topic: Topic name, used only for the log message.
        """
        marks_a, marks_b = blueprint_entry
        expected = marks_a + marks_b
        groups = ["A"] * len(marks_a) + ["B"] * len(marks_b)

        if len(questions) != len(expected):
            logger.warning(
                "AssessmentAgent._apply_blueprint_marks(): topic %r "
                "expected %d questions per blueprint but LLM returned "
                "%d — applying blueprint marks to as many as match, "
                "leaving any extras with their original marks.",
                topic[:60], len(expected), len(questions),
            )

        for i, q in enumerate(questions):
            if i < len(expected):
                q.marks = expected[i]
                q.blueprint_group = groups[i]
            # Extra questions beyond the blueprint (shouldn't normally
            # happen) keep whatever marks the LLM gave them and are left
            # with an empty blueprint_group, so export's automatic
            # even-split fallback picks them up rather than mis-grouping.

    def _generate_batched(
        self,
        plan: AssessmentPlan,
        rag_context: str,
        sources: List[SourceAttribution],
        batch_progress_callback: Optional[
            Callable[[int, int, int, int, int], None]
        ],
    ) -> Assessment:
        """Generate questions in sequential LLM batches and merge results.

        Each batch gets a proportionally allocated topic slice (see
        :meth:`_build_batch_specs`).  Batches are separated by a TPM-aware
        cooldown that subtracts the time already spent on the previous LLM
        call from the configured window, minimising idle wait.  Rate-limit
        errors trigger sleeping retries; other errors abort immediately.

        After all batches succeed the individual question lists are merged,
        ``question_id`` values are renumbered globally (Q1…Qn), and the
        merged count is checked against the requested total.

        Args:
            plan: Typed assessment plan.
            rag_context: Retrieved context string.
            sources: RAG source attributions.
            batch_progress_callback: Optional callback fired after each
                batch: ``(batch_idx, total_batches, qs_start, qs_end,
                total_qs)`` — all 1-based.

        Returns:
            Assessment: Merged, renumbered assessment.

        Raises:
            ValueError: When any batch fails after all retries, or when
                the merged result is empty.
        """
        batch_specs = self._build_batch_specs(plan)
        total_batches = len(batch_specs)
        total_qs = plan.question_count
        # self._vtu_blueprint was populated as a side effect of
        # _build_batch_specs() above (custom + auto-derived entries — see
        # _resolve_vtu_blueprint), ready for the per-batch loop below.
        logger.info(
            "AssessmentAgent._generate_batched(): %d questions → %d batches "
            "(size ≤ %d, delay %.0fs)",
            total_qs,
            total_batches,
            config.batch_size,
            config.batch_inter_call_delay_s,
        )

        batch_assessments: List[Assessment] = []
        questions_done = 0
        prev_call_start: Optional[float] = None
        # Flat list of every question_text generated so far in THIS run,
        # across all batches — passed to each subsequent batch's prompt so
        # the model can avoid repeating an earlier batch's question. Batches
        # are independent LLM calls with no other visibility into sibling
        # batches' output, so without this, thematically close topics
        # readily produce near-duplicate questions (observed in practice:
        # multiple batches independently asking "list the four phases of
        # Simon's decision-making process" when several topics all touch
        # decision-making phases).
        already_asked: List[str] = []

        for batch_idx, (batch_count, batch_topics) in enumerate(batch_specs):
            batch_num = batch_idx + 1  # 1-based for display

            # ── TPM-aware inter-call cooldown (all batches except first) ──
            # Groq's rolling 60 s window counts the previous call's requested
            # max_tokens from when that call STARTED, so we only need to
            # sleep the remainder of the window — the time the call itself
            # took (often 20-40 s) already counts toward it.
            if prev_call_start is not None:
                elapsed = time.monotonic() - prev_call_start
                delay = max(0.0, config.batch_inter_call_delay_s - elapsed)
                if delay > 0:
                    logger.info(
                        "AssessmentAgent._generate_batched(): sleeping %.0fs "
                        "before batch %d/%d (TPM cooldown; %.0fs of the "
                        "%.0fs window already elapsed).",
                        delay,
                        batch_num,
                        total_batches,
                        elapsed,
                        config.batch_inter_call_delay_s,
                    )
                    time.sleep(delay)
                else:
                    logger.info(
                        "AssessmentAgent._generate_batched(): no cooldown "
                        "needed before batch %d/%d (window already elapsed).",
                        batch_num,
                        total_batches,
                    )

            logger.info(
                "AssessmentAgent._generate_batched(): batch %d/%d — "
                "%d questions, topics=%r",
                batch_num,
                total_batches,
                batch_count,
                batch_topics[:60],
            )

            # ── Build this batch's plan fragment ───────────────────────────
            blueprint_entry = self._vtu_blueprint.get(batch_topics)
            per_question_marks = None
            if blueprint_entry is not None:
                marks_a, marks_b = blueprint_entry
                all_marks = marks_a + marks_b
                per_question_marks = [
                    (m, _depth_guidance_for_marks(m)) for m in all_marks
                ]
                batch_hint = (
                    f"BATCH {batch_num} OF {total_batches} — topic: "
                    f"{batch_topics}. This topic uses a FIXED faculty-"
                    f"authored marks blueprint (see PER-QUESTION MARKS "
                    f"below) — follow it exactly, do not add or drop "
                    f"questions. Other batches will cover the remaining "
                    f"topics separately."
                )
            else:
                batch_hint = (
                    f"BATCH {batch_num} OF {total_batches} — "
                    f"Generate exactly {batch_count} question(s) covering these "
                    f"topics: {batch_topics}. "
                    f"Other batches will cover the remaining topics separately."
                )
            user_prompt = build_assessment_prompt(
                assessment_type=plan.assessment_type,
                course_name=plan.course_name,
                course_code=plan.course_code,
                topics=batch_topics,
                bloom_targets=plan.bloom_targets,
                co_mapping=plan.co_mapping,
                rag_context=rag_context,
                question_count=batch_count,
                marks_per_question=plan.marks_per_question,
                difficulty=plan.difficulty,
                extra_instructions=plan.extra_instructions,
                batch_hint=batch_hint,
                per_question_marks=per_question_marks,
                avoid_repeating=already_asked,
            )

            # ── Invoke LLM: rate-limit retries + parse retries ────────────
            # call_start_tracker is updated before EVERY physical LLM request
            # (including rate-limit and parse retries), so the next batch's
            # cooldown is measured from the most recent request, not from an
            # obsolete first-attempt timestamp.
            call_start_tracker: List[float] = []
            batch_assessment = self._invoke_batch_with_parse_retry(
                user_prompt=user_prompt,
                plan=plan,
                sources=sources,
                batch_num=batch_num,
                total_batches=total_batches,
                call_start_tracker=call_start_tracker,
            )
            if call_start_tracker:
                prev_call_start = call_start_tracker[-1]

            # Tag every question in this batch with the topic string it was
            # generated for. This is what lets the export layer group
            # questions into VTU-style exam paper Modules later — the
            # association only exists here, at batch-assignment time, and
            # is otherwise lost once questions are merged into a flat list.
            for q in batch_assessment.questions:
                q.topic = batch_topics

            if blueprint_entry is not None:
                self._apply_blueprint_marks(
                    batch_assessment.questions, blueprint_entry, batch_topics
                )

            batch_assessments.append(batch_assessment)
            already_asked.extend(
                q.question_text for q in batch_assessment.questions
            )

            qs_start = questions_done + 1
            qs_end = questions_done + batch_assessment.question_count
            questions_done = qs_end

            logger.info(
                "AssessmentAgent._generate_batched(): batch %d/%d done — "
                "%d questions (running total %d/%d)",
                batch_num,
                total_batches,
                batch_assessment.question_count,
                questions_done,
                total_qs,
            )

            # ── Fire progress callback ─────────────────────────────────────
            if batch_progress_callback:
                try:
                    batch_progress_callback(
                        batch_num, total_batches, qs_start, qs_end, total_qs
                    )
                except Exception as cb_exc:  # noqa: BLE001
                    logger.debug("batch_progress_callback raised (ignored): %s", cb_exc)

        # ── Merge and renumber ────────────────────────────────────────────
        merged = self._merge_batches(batch_assessments, plan)
        return merged

    def _invoke_batch_with_parse_retry(
        self,
        user_prompt: str,
        plan: AssessmentPlan,
        sources: List[SourceAttribution],
        batch_num: int,
        total_batches: int,
        call_start_tracker: Optional[List[float]] = None,
    ) -> Assessment:
        """Attempt one batch LLM call with JSON-parse retries and rate-limit retries.

        Rate-limit errors (413/429) are handled by
        :func:`_invoke_llm_with_rate_retry`.  JSON parse/validation errors
        are retried up to ``self.max_retries`` additional times (no sleep
        between parse retries because the batch is already sized to fit the
        token budget — the truncation cause is transient model variance, not
        systematic overflow).

        Args:
            user_prompt: Fully-built user prompt for this batch.
            plan: The original assessment plan (used for metadata fallback).
            sources: RAG source attributions.
            batch_num: 1-based batch number (for log messages).
            total_batches: Total number of batches (for log messages).
            call_start_tracker: Optional list that receives a
                ``time.monotonic()`` timestamp immediately before every
                physical LLM request (including all retries).  The caller
                uses the last entry for TPM cooldown accounting.

        Returns:
            Assessment: Parsed assessment for this batch.

        Raises:
            ValueError: After ``max_retries + 1`` parse failures.
            RuntimeError: On persistent rate-limit or network errors.
            ConfigurationError: When GROQ_API_KEY is absent.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 2):
            logger.info(
                "AssessmentAgent batch %d/%d: LLM attempt %d/%d",
                batch_num,
                total_batches,
                attempt,
                self.max_retries + 1,
            )
            try:
                raw = _invoke_llm_with_rate_retry(
                    system_prompt=ASSESSMENT_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    max_rate_retries=2,
                    rate_retry_delay_s=config.batch_inter_call_delay_s,
                    on_call_start=(
                        (lambda: call_start_tracker.append(time.monotonic()))
                        if call_start_tracker is not None
                        else None
                    ),
                )
                batch_assessment = self._parse_assessment_response(
                    raw=raw,
                    plan=plan,
                    sources=sources,
                )
                return batch_assessment
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_error = exc
                logger.warning(
                    "AssessmentAgent batch %d/%d: parse error attempt %d: %s",
                    batch_num,
                    total_batches,
                    attempt,
                    exc,
                )
            except (RuntimeError, ConfigurationError):
                raise

        raise ValueError(
            f"AssessmentAgent batch {batch_num}/{total_batches} failed to "
            f"produce valid JSON after {self.max_retries + 1} attempts.  "
            f"Last error: {last_error}"
        )

    def _merge_batches(
        self, batches: List[Assessment], plan: AssessmentPlan
    ) -> Assessment:
        """Merge a list of per-batch assessments into a single Assessment.

        Renumbers all question IDs globally (Q1, Q2, …, Qn) in batch order
        so exports and analytics see sequential, unique IDs.  The metadata
        envelope from the first batch is kept; total_marks and question_count
        are :class:`~models.Assessment` properties that recompute from the
        merged question list automatically.

        Logs a warning and adds a note to ``generation_notes`` when the
        merged question count is less than ``plan.question_count``.

        Args:
            batches: Ordered list of per-batch :class:`Assessment` objects.
            plan: The original plan (used for the count-check warning).

        Returns:
            Assessment: Merged, globally-renumbered assessment.

        Raises:
            ValueError: When ``batches`` is empty or no questions could be
                parsed from any batch.
        """
        if not batches:
            raise ValueError(
                "AssessmentAgent._merge_batches(): no batches to merge."
            )

        all_questions: List[Question] = []
        notes_parts: List[str] = []
        global_idx = 1

        for b_idx, batch in enumerate(batches):
            for q in batch.questions:
                # Renumber: never trust per-batch LLM IDs.
                q.question_id = f"Q{global_idx}"
                global_idx += 1
                all_questions.append(q)
            if batch.generation_notes and batch.generation_notes.strip():
                notes_parts.append(
                    f"[Batch {b_idx + 1}/{len(batches)}] {batch.generation_notes}"
                )

        if not all_questions:
            raise ValueError(
                "AssessmentAgent._merge_batches(): all batches produced zero "
                "valid questions."
            )

        merged_count = len(all_questions)
        generation_notes = "; ".join(notes_parts) if notes_parts else ""

        if merged_count < plan.question_count:
            shortfall = plan.question_count - merged_count
            warn_msg = (
                f"WARNING: LLM returned {merged_count} questions across "
                f"{len(batches)} batches but {plan.question_count} were "
                f"requested (shortfall: {shortfall}).  "
                f"Consider re-generating or reducing the question count."
            )
            logger.warning(
                "AssessmentAgent._merge_batches(): %s", warn_msg
            )
            generation_notes = (
                (generation_notes + "; " if generation_notes else "") + warn_msg
            )

        # Reuse first batch's metadata envelope; total_marks and
        # question_count are live properties on Assessment, so they
        # automatically reflect the merged question list.
        first_meta = batches[0].metadata

        merged = Assessment(
            metadata=first_meta,
            questions=all_questions,
            generation_notes=generation_notes,
        )

        logger.info(
            "AssessmentAgent._merge_batches(): merged %d questions "
            "from %d batches (Q1–Q%d)",
            merged_count,
            len(batches),
            merged_count,
        )
        return merged

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_assessment_response(
        self,
        raw: str,
        plan: AssessmentPlan,
        sources: List[SourceAttribution],
    ) -> Assessment:
        """Parse the LLM assessment response into an :class:`Assessment`.

        Args:
            raw: Raw LLM response text.
            plan: Typed :class:`~models.AssessmentPlan` used for fallback metadata.
            sources: RAG source attribution objects.

        Returns:
            Assessment: Populated dataclass.

        Raises:
            ValueError: When JSON extraction or question parsing fails.
        """
        json_str = _extract_json(raw)
        try:
            data: dict = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.error(
                "AssessmentAgent: JSON parse error: %s\nRaw (first 600):\n%s",
                exc,
                raw[:600],
            )
            raise ValueError(
                f"AssessmentAgent: malformed JSON from LLM: {exc}"
            ) from exc

        questions_raw: List[dict] = data.get("questions", [])
        if not questions_raw:
            raise ValueError(
                "AssessmentAgent: LLM response contained zero questions."
            )

        questions = self._parse_questions(questions_raw, sources)

        assessment_type_str = data.get("assessment_type", plan.assessment_type)
        assessment_type_enum = _parse_assessment_type(assessment_type_str)

        metadata = AssessmentMetadata(
            title=data.get(
                "assessment_title",
                f"{assessment_type_str} – {plan.course_name or 'Course'}",
            ),
            course_code=data.get("course_code", plan.course_code),
            course_name=data.get("course_name", plan.course_name),
            assessment_type=assessment_type_enum,
            semester=plan.semester,
            duration_minutes=int(
                data.get("duration_minutes", plan.duration_minutes) or 60
            ),
            total_marks=sum(q.marks for q in questions),
            department=plan.department,
            faculty_name=plan.faculty_name,
            instructions=data.get("instructions", ""),
            test_date=plan.test_date,
        )

        return Assessment(
            metadata=metadata,
            questions=questions,
            generation_notes=data.get("generation_notes", ""),
        )

    def _parse_questions(
        self,
        questions_raw: List[dict],
        sources: List[SourceAttribution],
    ) -> List[Question]:
        """Convert raw question dicts from LLM JSON into :class:`Question` objects.

        Args:
            questions_raw: List of raw question dicts from LLM JSON.
            sources: RAG-retrieved source attribution objects to attach.

        Returns:
            List[Question]: Parsed and validated Question objects.

        Raises:
            ValueError: When question_text is missing on any question.
        """
        questions: List[Question] = []
        for i, raw_q in enumerate(questions_raw, start=1):
            q_text = str(raw_q.get("question_text", "")).strip()
            if not q_text:
                logger.warning(
                    "Question %d has no question_text — skipping", i
                )
                continue

            # Parse CO mapping: accept list or comma-separated string
            co_raw = raw_q.get("co_mapping", ["CO1"])
            if isinstance(co_raw, str):
                co_list = [c.strip() for c in co_raw.split(",") if c.strip()]
            elif isinstance(co_raw, list):
                co_list = [str(c).strip() for c in co_raw if str(c).strip()]
            else:
                co_list = ["CO1"]
            if not co_list:
                co_list = ["CO1"]

            # Attach sources proportionally — spread RAG sources across questions
            q_sources: List[SourceAttribution] = []
            if sources:
                src = sources[(i - 1) % len(sources)]
                q_sources = [src]

            question = Question(
                question_id=str(raw_q.get("question_id", f"Q{i}")),
                question_text=q_text,
                bloom_level=_parse_bloom(
                    str(raw_q.get("bloom_level", "Understand"))
                ),
                co_mapping=co_list,
                difficulty=_parse_difficulty(
                    str(raw_q.get("difficulty", "Medium"))
                ),
                marks=int(raw_q.get("marks", 5) or 5),
                answer_key=str(raw_q.get("answer_key", "")).strip(),
                question_type=str(
                    raw_q.get("question_type", "Short Answer")
                ).strip(),
                sources=q_sources,
                notes=str(raw_q.get("notes", "")),
                options=[
                    str(o).strip()
                    for o in (raw_q.get("options") or [])
                    if str(o).strip()
                ] if isinstance(raw_q.get("options"), (list, tuple)) else [],
                case_background=str(
                    raw_q.get("case_background", "")
                ).strip(),
            )
            questions.append(question)

        if not questions:
            raise ValueError(
                "AssessmentAgent: no valid questions could be parsed from LLM output."
            )

        logger.debug("Parsed %d questions from LLM response", len(questions))
        return questions


# ---------------------------------------------------------------------------
# Analytics Agent
# ---------------------------------------------------------------------------


class AnalyticsAgent:
    """Calculates quantitative analytics for a generated assessment.

    Fully deterministic — no LLM calls.  Computes Bloom distribution,
    CO coverage, difficulty balance, marks distribution, knowledge sources
    used, estimated faculty time saved, and an overall analytics quality
    score.

    Handles missing / partial data gracefully:
      - Empty questions list → returns zeroed :class:`~models.AnalyticsReport`
        with a logged warning.
      - Missing answer keys → reflected in quality score sub-score 4.
      - Missing sources → ``knowledge_sources_used`` is an empty list.
    """

    def analyse(self, assessment: Assessment) -> AnalyticsReport:
        """Compute analytics for the given assessment.

        Args:
            assessment: The :class:`~models.Assessment` to analyse.

        Returns:
            AnalyticsReport: Fully populated analytics container.
                ``reviewer_result`` is always ``None`` — the orchestration
                layer attaches it separately after calling
                :meth:`ReviewerAgent.review`.
        """
        logger.info(
            "AnalyticsAgent.analyse(): question_count=%d",
            assessment.question_count,
        )

        # --- Guard: empty assessment ---
        if not assessment.questions:
            logger.warning(
                "AnalyticsAgent.analyse(): assessment has no questions — "
                "returning zeroed report."
            )
            return AnalyticsReport()

        questions = assessment.questions
        n = len(questions)

        # --- Marks distribution ---
        marks_distribution: Dict[str, int] = {
            q.question_id: q.marks for q in questions
        }
        total_marks = sum(q.marks for q in questions)

        # --- Bloom distribution ---
        bloom_counts: Dict[str, int] = {}
        for level in BloomLevel:
            bloom_counts[level.value] = 0
        for q in questions:
            bloom_counts[q.bloom_level.value] = (
                bloom_counts.get(q.bloom_level.value, 0) + 1
            )
        # Remove zero-count levels from the counts dict for cleaner display,
        # but keep all six keys in coverage_percent via BloomDistribution.
        bloom_distribution = BloomDistribution(counts=bloom_counts)

        # --- CO coverage ---
        co_coverage: Dict[str, int] = {}
        for q in questions:
            for co in q.co_mapping:
                co_coverage[co] = co_coverage.get(co, 0) + 1

        # --- Difficulty distribution ---
        difficulty_distribution: Dict[str, int] = {
            DifficultyLevel.EASY.value: 0,
            DifficultyLevel.MEDIUM.value: 0,
            DifficultyLevel.HARD.value: 0,
        }
        for q in questions:
            difficulty_distribution[q.difficulty.value] = (
                difficulty_distribution.get(q.difficulty.value, 0) + 1
            )

        # --- Knowledge sources ---
        seen_docs: set[str] = set()
        for q in questions:
            for src in q.sources:
                seen_docs.add(src.document_name)
        knowledge_sources_used = sorted(seen_docs)

        # --- Time saved ---
        estimated_time_saved = _compute_time_saved(assessment)

        # --- Analytics quality score ---
        quality_score = _compute_analytics_quality_score(assessment)

        report = AnalyticsReport(
            question_count=n,
            total_marks=total_marks,
            bloom_distribution=bloom_distribution,
            co_coverage=co_coverage,
            difficulty_distribution=difficulty_distribution,
            knowledge_sources_used=knowledge_sources_used,
            estimated_time_saved_minutes=estimated_time_saved,
            quality_score=quality_score,
            marks_distribution=marks_distribution,
            reviewer_result=None,
        )

        logger.info(
            "AnalyticsAgent.analyse(): quality_score=%.1f time_saved=%dm "
            "bloom_levels=%d distinct_cos=%d",
            quality_score,
            estimated_time_saved,
            len({q.bloom_level for q in questions}),
            len(co_coverage),
        )
        return report


# ---------------------------------------------------------------------------
# Reviewer Agent
# ---------------------------------------------------------------------------


class ReviewerAgent:
    """AI-powered quality reviewer: reviews generated assessments via Groq LLM.

    Evaluates the generated assessment for quality, Bloom coverage, duplicate
    questions, and difficulty balance.  Returns a :class:`~models.ReviewerResult`.

    Behaviour on failure:
      - LLM call error or malformed JSON → retries once, then returns a
        :class:`~models.ReviewerResult` with ``error`` set and
        ``quality_score=0.0``.  Never raises after construction.

    Attributes:
        temperature: LLM sampling temperature (default 0.2 for consistency).
    """

    # LLM temperature for the reviewer — lower than generation for repeatability
    _TEMPERATURE: float = 0.2

    def __init__(self) -> None:
        """Initialise the ReviewerAgent."""
        logger.info("ReviewerAgent initialised (temperature=%.2f)", self._TEMPERATURE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def review(self, assessment: Assessment) -> ReviewerResult:
        """Review an assessment and return a quality report.

        Sends a compact assessment summary (no answer key bodies, only
        presence flags) to the Groq LLM and parses its structured JSON
        response into a :class:`~models.ReviewerResult`.

        Also runs deterministic duplicate detection via
        :func:`_find_duplicate_questions` and merges those IDs with any
        duplicates the LLM reports.

        Args:
            assessment: The :class:`~models.Assessment` to review.

        Returns:
            ReviewerResult: Populated quality report.  If the LLM call or
                JSON parse fails after one retry, returns a default result
                with ``error`` set.
        """
        logger.info(
            "ReviewerAgent.review(): question_count=%d",
            assessment.question_count,
        )

        # --- Guard: empty assessment ---
        if not assessment.questions:
            logger.warning(
                "ReviewerAgent.review(): assessment has no questions — "
                "returning empty ReviewerResult."
            )
            return ReviewerResult(
                error="Assessment contains no questions; review skipped."
            )

        # --- Deterministic duplicate detection (always runs) ---
        deterministic_dupes = _find_duplicate_questions(assessment.questions)
        logger.debug(
            "Deterministic duplicates found: %s", deterministic_dupes
        )

        # --- Build compact summary for LLM (omit answer key bodies) ---
        summary = self._build_assessment_summary(assessment)

        # --- LLM call with one retry ---
        llm_result = self._invoke_reviewer_with_retry(summary)

        # --- Merge deterministic duplicates with LLM-reported duplicates ---
        if llm_result.error is None:
            merged_dupes = sorted(
                set(llm_result.duplicate_question_ids) | set(deterministic_dupes)
            )
            llm_result.duplicate_question_ids = merged_dupes
        else:
            # LLM failed; surface at least the deterministic duplicates
            llm_result.duplicate_question_ids = deterministic_dupes

        logger.info(
            "ReviewerAgent.review(): quality_score=%.1f "
            "strengths=%d weaknesses=%d duplicates=%d error=%s",
            llm_result.quality_score,
            len(llm_result.strengths),
            len(llm_result.weaknesses),
            len(llm_result.duplicate_question_ids),
            llm_result.error,
        )
        return llm_result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_assessment_summary(self, assessment: Assessment) -> str:
        """Serialise the assessment into a compact JSON string for the reviewer.

        Answer key bodies are intentionally omitted to conserve tokens; a
        boolean ``answer_key_present`` flag is included per question so the
        reviewer can penalise missing keys.

        Args:
            assessment: The assessment to summarise.

        Returns:
            str: JSON string of the compact assessment summary.
        """
        meta = assessment.metadata
        summary: Dict[str, Any] = {
            "assessment_type": meta.assessment_type.value,
            "course_name": meta.course_name,
            "course_code": meta.course_code,
            "total_marks": assessment.total_marks,
            "duration_minutes": meta.duration_minutes,
            "question_count": assessment.question_count,
            "questions": [
                {
                    "question_id": q.question_id,
                    "question_text": q.question_text,
                    "question_type": q.question_type,
                    "bloom_level": q.bloom_level.value,
                    "co_mapping": q.co_mapping,
                    "difficulty": q.difficulty.value,
                    "marks": q.marks,
                    "answer_key_present": bool(
                        q.answer_key and q.answer_key.strip()
                    ),
                }
                for q in assessment.questions
            ],
        }
        return json.dumps(summary, ensure_ascii=False, indent=2)

    def _invoke_reviewer_with_retry(self, summary: str) -> ReviewerResult:
        """Call the LLM reviewer with one automatic retry on parse failure.

        Retry logic:
          - LLM / config hard-errors (RuntimeError, ConfigurationError): break
            immediately — retrying a decommissioned model or missing API key
            will never succeed.
          - All other exceptions (JSONDecodeError, ValueError, KeyError,
            AttributeError, TypeError, and any other unexpected parse-time
            error): retry once, then return a graceful ReviewerResult(error=…).

        Args:
            summary: Compact JSON assessment summary string.

        Returns:
            ReviewerResult: Parsed result, or a default result with
                ``error`` set if both attempts fail.
        """
        user_prompt = build_reviewer_prompt(summary)
        last_error: Optional[str] = None

        for attempt in range(1, 3):  # attempts 1 and 2
            logger.info("ReviewerAgent: LLM call attempt %d/2", attempt)
            try:
                raw = _invoke_llm(
                    system_prompt=REVIEWER_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=self._TEMPERATURE,
                )
                return self._parse_reviewer_response(raw)
            except (RuntimeError, ConfigurationError) as exc:
                # Hard infrastructure error — do not retry
                last_error = str(exc)
                logger.error(
                    "ReviewerAgent: LLM/config error on attempt %d: %s",
                    attempt,
                    exc,
                )
                break
            except Exception as exc:  # noqa: BLE001
                # Catches JSONDecodeError, ValueError, KeyError, AttributeError,
                # TypeError, and any other unexpected parse-time exception so
                # the agent always returns ReviewerResult rather than raising.
                last_error = str(exc)
                logger.warning(
                    "ReviewerAgent: parse/unexpected error on attempt %d "
                    "(%s): %s",
                    attempt,
                    type(exc).__name__,
                    exc,
                )

        logger.error(
            "ReviewerAgent: all attempts failed.  Last error: %s", last_error
        )
        return ReviewerResult(
            quality_score=0.0,
            error=f"Reviewer LLM failed after 2 attempts: {last_error}",
        )

    def _parse_reviewer_response(self, raw: str) -> ReviewerResult:
        """Parse the LLM reviewer JSON response into a :class:`ReviewerResult`.

        All fields are parsed with explicit type guards: unexpected shapes
        (wrong type, null, missing) are coerced to safe defaults rather than
        propagating AttributeError or TypeError to the caller.

        Args:
            raw: Raw LLM response text (may contain markdown fences or prose).

        Returns:
            ReviewerResult: Populated result.

        Raises:
            ValueError: When JSON extraction fails, the top-level value is not
                a JSON object, or json.loads() raises JSONDecodeError (normalised
                to ValueError so the retry loop catches it uniformly).
        """
        json_str = _extract_json(raw)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            # Normalise to ValueError so _invoke_reviewer_with_retry retries it
            raise ValueError(f"ReviewerAgent: malformed JSON: {exc}") from exc

        # Guard: top-level must be a JSON object.  Arrays, scalars, and null
        # are not valid responses; raise ValueError to trigger a retry.
        if not isinstance(data, dict):
            raise ValueError(
                f"ReviewerAgent: expected a JSON object at top level, "
                f"got {type(data).__name__!r}: {str(data)[:120]}"
            )

        # --- quality_score: accept int, float, numeric-string; default 0.0 ---
        raw_score = data.get("quality_score", 0)
        try:
            quality_score = float(raw_score)
        except (TypeError, ValueError):
            quality_score = 0.0
        quality_score = max(0.0, min(100.0, quality_score))

        # --- List fields: accept list-of-anything; coerce items to str;
        #     ignore null, dict, string, and other non-list shapes. ---
        def _str_list(key: str) -> List[str]:
            """Return a clean List[str] for *key*, ignoring non-list values."""
            val = data.get(key)
            if not isinstance(val, list):
                return []
            return [str(item) for item in val if item is not None]

        strengths = _str_list("strengths")
        weaknesses = _str_list("weaknesses")
        suggestions = _str_list("suggestions")
        duplicate_ids = _str_list("duplicate_question_ids")

        # --- bloom_coverage: three accepted shapes from the LLM ---
        #   1. dict {"Remember": true, ...}         — canonical
        #   2. list ["Remember", "Apply", ...]      — list of covered level names
        #   3. null / string / other                — treat all levels as False
        bloom_raw = data.get("bloom_coverage")
        bloom_coverage: Dict[str, bool] = {}

        if isinstance(bloom_raw, dict):
            for level in BloomLevel:
                raw_val = bloom_raw.get(level.value)   # may be bool, str, int, None
                if isinstance(raw_val, bool):
                    bloom_coverage[level.value] = raw_val
                elif isinstance(raw_val, str):
                    bloom_coverage[level.value] = raw_val.lower() in (
                        "true", "1", "yes"
                    )
                elif raw_val is None:
                    bloom_coverage[level.value] = False
                else:
                    # int or other numeric — treat non-zero as True
                    try:
                        bloom_coverage[level.value] = bool(raw_val)
                    except Exception:  # noqa: BLE001
                        bloom_coverage[level.value] = False

        elif isinstance(bloom_raw, list):
            # LLM returned a list of covered Bloom level names
            covered = {str(item).strip() for item in bloom_raw if item is not None}
            for level in BloomLevel:
                bloom_coverage[level.value] = level.value in covered

        else:
            # null, string, missing, or any other unexpected type → all False
            logger.debug(
                "ReviewerAgent: bloom_coverage has unexpected type %s — "
                "defaulting all levels to False",
                type(bloom_raw).__name__,
            )
            for level in BloomLevel:
                bloom_coverage[level.value] = False

        # --- difficulty_balance_ok: bool/str/int/null; default True on null ---
        balance_raw = data.get("difficulty_balance_ok")
        if isinstance(balance_raw, bool):
            difficulty_balance_ok = balance_raw
        elif isinstance(balance_raw, str):
            difficulty_balance_ok = balance_raw.lower() in ("true", "1", "yes")
        elif balance_raw is None:
            # Missing or null — do not penalise; use safe default
            difficulty_balance_ok = True
        else:
            try:
                difficulty_balance_ok = bool(balance_raw)
            except Exception:  # noqa: BLE001
                difficulty_balance_ok = True

        # --- reviewer_notes: coerce to string; null → empty string ---
        notes_raw = data.get("reviewer_notes")
        reviewer_notes = str(notes_raw) if notes_raw is not None else ""

        logger.debug(
            "ReviewerAgent: parsed response quality_score=%.1f "
            "strengths=%d weaknesses=%d bloom_keys=%d",
            quality_score,
            len(strengths),
            len(weaknesses),
            len(bloom_coverage),
        )

        return ReviewerResult(
            quality_score=quality_score,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            duplicate_question_ids=duplicate_ids,
            bloom_coverage=bloom_coverage,
            difficulty_balance_ok=difficulty_balance_ok,
            reviewer_notes=reviewer_notes,
            error=None,
        )


# ---------------------------------------------------------------------------
# Download Agent (stub — later task)
# ---------------------------------------------------------------------------


class DownloadAgent:
    """Orchestrates document generation in Markdown, Word, and PDF formats.

    Delegates to :mod:`downloads` :class:`~downloads.DownloadEngine` for
    the actual rendering logic.  Each format is generated independently so
    a failure in one format does not prevent the others.

    Attributes:
        _engine: Lazily-instantiated :class:`~downloads.DownloadEngine`.
    """

    def __init__(self) -> None:
        """Initialise the DownloadAgent."""
        self._engine = None
        logger.info("DownloadAgent initialised")

    def _get_engine(self):
        """Lazily import and instantiate :class:`~downloads.DownloadEngine`.

        Returns:
            DownloadEngine: Configured export engine.

        Raises:
            ImportError: When the downloads module cannot be imported.
        """
        if self._engine is None:
            try:
                from downloads import DownloadEngine  # lazy import
            except ImportError as exc:
                raise ImportError(
                    "downloads module is not available.  "
                    "Ensure all required packages are installed."
                ) from exc
            self._engine = DownloadEngine()
        return self._engine

    def export(self, assessment: Assessment, fmt: str) -> bytes:
        """Export an assessment to the requested format.

        Delegates to :class:`~downloads.DownloadEngine`.  Markdown output
        is encoded to UTF-8 bytes; Word and PDF are returned as-is.

        Args:
            assessment: The :class:`~models.Assessment` to export.
            fmt: Target format — one of ``"markdown"``, ``"docx"``,
                ``"pdf"`` (case-insensitive).

        Returns:
            bytes: Rendered document bytes.

        Raises:
            ValueError: When *fmt* is not a recognised export format.
            RuntimeError: When the underlying exporter raises.
            ImportError: When the downloads module is unavailable.
        """
        fmt_lower = fmt.lower().strip()
        logger.info("DownloadAgent.export(): fmt=%s", fmt_lower)

        engine = self._get_engine()

        if fmt_lower == "markdown":
            text: str = engine.export_markdown(assessment)
            return text.encode("utf-8")
        elif fmt_lower == "docx":
            return engine.export_word(assessment)
        elif fmt_lower == "pdf":
            return engine.export_pdf(assessment)
        else:
            raise ValueError(
                f"DownloadAgent: unsupported format {fmt!r}.  "
                "Use 'markdown', 'docx', or 'pdf'."
            )
