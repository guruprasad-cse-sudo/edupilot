"""
EduPilot AI Faculty Assistant
Module: prompts.py
Version: 4.2.0
Author: EduPilot Team
Purpose: LangChain prompt template library.  Defines reusable, per-assessment-
         type ChatPromptTemplate instances and factory functions used by the
         Planning, Assessment, and Reviewer agents.  Every template instructs
         the model to return strict JSON so the parser in agent.py has a stable
         contract.

         v4.2 additions:
           - REVIEWER_SYSTEM_PROMPT: full identity + rules for the Reviewer Agent.
           - build_reviewer_prompt(): expanded with complete JSON output schema,
             Bloom coverage check, difficulty balance analysis, and duplicate
             detection instructions.
"""

from __future__ import annotations

from typing import List

from logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# JSON output schema (shared across all assessment types)
# ---------------------------------------------------------------------------

ASSESSMENT_JSON_SCHEMA = """\
Return ONLY a single JSON object — no markdown fences, no prose before or after.

Schema:
{
  "assessment_title": "<string — descriptive title for this assessment>",
  "course_code": "<string>",
  "course_name": "<string>",
  "duration_minutes": <integer>,
  "instructions": "<string — student-facing general instructions>",
  "generation_notes": "<string — optional notes from the generation agent>",
  "questions": [
    {
      "question_id": "Q1",
      "question_text": "<full question text>",
      "question_type": "<MCQ | Short Answer | Long Answer | Numerical | Oral | Essay | Case Study>",
      "bloom_level": "<Remember | Understand | Apply | Analyze | Evaluate | Create>",
      "co_mapping": ["CO1"],
      "difficulty": "<Easy | Medium | Hard>",
      "marks": <integer>,
      "answer_key": "<model answer or marking scheme>",
      "options": ["<option 1>", "<option 2>", "<option 3>", "<option 4>"],
      "notes": "<optional notes for the faculty>"
    }
  ]
}
"""

# ---------------------------------------------------------------------------
# System prompt — shared identity for all generation calls
# ---------------------------------------------------------------------------

ASSESSMENT_SYSTEM_PROMPT = """\
You are EduPilot, an expert AI assistant specialising in Outcome-Based Education
(OBE) assessment design for higher education.  You generate rigorous,
curriculum-aligned assessments following Bloom's Taxonomy (Revised, 2001) and
the institution's OBE framework.

Rules you MUST follow:
1. Map every question to the correct Bloom level and the specified Course Outcomes.
2. Spread difficulty (Easy/Medium/Hard) across questions unless the faculty
   explicitly requests a single difficulty.
3. Provide a complete, accurate answer key or marking scheme for every question.
4. Base questions on the supplied knowledge-base context whenever possible.
   Cite the knowledge source in the 'notes' field (e.g. "Source: lecture_notes.pdf").
5. Never fabricate course-specific facts; use only information present in the
   context or general academic knowledge.
6. Honour the exact question count and marks per question specified.
"""

# ---------------------------------------------------------------------------
# Reviewer system prompt
# ---------------------------------------------------------------------------

REVIEWER_SYSTEM_PROMPT = """\
You are EduPilot's Reviewer Agent — an expert academic quality auditor for
Outcome-Based Education (OBE) assessments in higher education.

Your role is to critically evaluate a generated assessment and return a
structured JSON quality report.  You do NOT generate questions; you only review.

Evaluation dimensions you MUST address:

1. QUALITY SCORE (0–100 integer or decimal)
   Rate the overall academic rigour, clarity, and OBE alignment.
   90–100 = Excellent / publication-ready
   70–89  = Good, minor improvements needed
   50–69  = Acceptable, significant gaps present
   < 50   = Poor, major rework required

2. STRENGTHS
   List concrete, specific strengths (not vague praise).
   Examples: "All six Bloom levels are represented", "CO3 is well-covered
   with three application-level questions".

3. WEAKNESSES
   List specific gaps or problems.
   Examples: "No Create-level questions despite Assignment type",
   "Q4 and Q6 test identical recall of the same definition".

4. SUGGESTIONS
   Actionable, ranked improvement recommendations.
   Examples: "Replace Q2 (Remember) with an Evaluate-level scenario question
   to improve cognitive depth", "Add an answer key to Q7".

5. DUPLICATE QUESTIONS
   Identify question IDs that test the same concept in the same way.
   Paraphrase alone does NOT constitute a duplicate — the underlying
   cognitive demand and concept must also be identical.
   Return an empty list if no duplicates are found.

6. BLOOM COVERAGE
   For each Bloom level (Remember, Understand, Apply, Analyze, Evaluate,
   Create) state whether it is covered (true) or absent (false) given
   the questions provided.

7. DIFFICULTY BALANCE
   State whether the Easy/Medium/Hard distribution is reasonable for the
   given assessment type.  A single difficulty level is only acceptable
   for targeted drills or Viva oral exams.

Rules:
- Base your review ONLY on the question data supplied; do not invent content.
- Be specific: reference question IDs when citing problems or strengths.
- Never reward an assessment for having an answer key you cannot see;
  flag missing answer keys as a weakness.
- Return ONLY a single JSON object — no markdown fences, no prose outside JSON.
"""

# ---------------------------------------------------------------------------
# Reviewer JSON output schema
# ---------------------------------------------------------------------------

REVIEWER_JSON_SCHEMA = """\
Return ONLY a single JSON object — no markdown fences, no prose before or after.

Schema:
{
  "quality_score": <number 0–100>,
  "strengths": [
    "<specific strength 1>",
    "<specific strength 2>"
  ],
  "weaknesses": [
    "<specific weakness 1>",
    "<specific weakness 2>"
  ],
  "suggestions": [
    "<ranked suggestion 1>",
    "<ranked suggestion 2>"
  ],
  "duplicate_question_ids": [
    "<question_id of first duplicate>",
    "<question_id of its duplicate pair>"
  ],
  "bloom_coverage": {
    "Remember": <true|false>,
    "Understand": <true|false>,
    "Apply": <true|false>,
    "Analyze": <true|false>,
    "Evaluate": <true|false>,
    "Create": <true|false>
  },
  "difficulty_balance_ok": <true|false>,
  "reviewer_notes": "<optional free-text notes for faculty>"
}

Important rules for the JSON:
- quality_score must be a number, not a string.
- All list fields must be arrays (use [] if empty, never null).
- bloom_coverage must always contain all six keys.
- difficulty_balance_ok must be a boolean.
- reviewer_notes may be an empty string but must be present.
"""

# ---------------------------------------------------------------------------
# Per-assessment-type guidance blocks
# ---------------------------------------------------------------------------

_TYPE_GUIDANCE: dict[str, str] = {
    "Internal Assessment": """\
This is a formal mid-semester Internal Assessment.
- Prefer Short Answer and Long Answer question types.
- Target a balanced spread of Bloom levels: at least one Apply/Analyze question.
- Default duration: 90 minutes unless specified.
- Include unit and module references in answer keys where relevant.
""",
    "Quiz": """\
This is a quick knowledge-check Quiz.
- Every question MUST be an MCQ (question_type "MCQ") with exactly 4 answer
  options in the "options" array. Options must be plain answer text without
  "A)"/"B)" prefixes. Exactly one option is correct; the others must be
  plausible distractors.
- The answer_key must state the correct option verbatim, optionally followed
  by a one-sentence justification.
- Favour Remember, Understand, and Apply levels.
- Keep each question concise.
- Default duration: 20–30 minutes.
""",
    "Assignment": """\
This is a take-home Assignment.
- Use Long Answer, Essay, Case Study, or Numerical question types.
- Target higher-order Bloom levels: Analyze, Evaluate, Create.
- Questions should require independent research or application.
- Default duration: 1–2 weeks (express as 0 minutes in the JSON).
- Each question must have a detailed marking rubric as the answer_key.
""",
    "Semester Examination": """\
This is a formal end-semester Semester Examination.
- Mix MCQ, Short Answer, and Long Answer types.
- Cover all specified Course Outcomes with at least one question each.
- Span all six Bloom levels across the paper.
- Default duration: 180 minutes.
- Include section headings in generation_notes if multiple sections exist.
""",
    "Viva": """\
This is a Viva (oral examination).
- All questions must be open-ended and suitable for spoken responses.
- Use question_type "Oral" for all questions.
- Target higher-order thinking: Apply, Analyze, Evaluate, Create.
- Include follow-up probing hints in the answer_key field.
- Default duration: 15–20 minutes per student (express as 0 in the JSON).
""",
    "Role Play": """\
This is a classroom Role Play activity for experiential learning.
- Each "question" is one complete role-play SCENARIO the faculty can run in
  class. Use question_type "Role Play" for every item.
- The question_text must contain, clearly structured with Markdown headings
  or bold labels: (1) **Scenario** — a realistic situation grounded in the
  given topics; (2) **Roles** — 2–5 named roles with a one-line brief each;
  (3) **Setup** — time needed, room arrangement, any props/materials;
  (4) **Task** — what the participants must accomplish or resolve.
- The answer_key is the FACULTY FACILITATION GUIDE: learning objectives,
  discussion/debrief questions, expected behaviours or talking points per
  role, and an observation rubric for awarding the marks.
- Target higher-order Bloom levels: Apply, Analyze, Evaluate, Create.
- Scenarios must be practical, engaging, and directly tied to the specified
  topics and Course Outcomes.
- Default duration: 10–15 minutes per scenario plus debrief (express the
  total as duration_minutes in the JSON).
""",
    "Question Bank": """\
This is a Question Bank — a reference repository of questions faculty draw
from when setting future tests and exams, NOT a single timed paper.
- Provide broad coverage: spread questions evenly across ALL the specified
  topics and Course Outcomes, and across the target Bloom levels.
- Do NOT use MCQ questions. Use only descriptive types: Short Answer,
  Long Answer, Essay, Case Study, and Numerical where the subject allows.
  Leave the "options" array empty for every question.
- Order questions by topic, and within a topic from lower to higher Bloom
  levels (easy recall first, then application/analysis).
- Prefix each question_text with the topic it covers in bold, e.g.
  "**[Binary Search Trees]** Explain…".
- Every question must have a complete answer_key (correct option for MCQs,
  model answer or marking scheme for the rest).
- Set duration_minutes to 0 — a question bank is not a timed assessment.
""",
}

# ---------------------------------------------------------------------------
# Planning agent prompt
# ---------------------------------------------------------------------------

PLANNING_SYSTEM_PROMPT = """\
You are EduPilot's Planning Agent.  Extract a structured assessment plan from
a natural-language faculty requirement.

Return ONLY a single JSON object — no markdown fences, no prose.

Schema:
{
  "assessment_type": "<Internal Assessment | Quiz | Assignment | Semester Examination | Viva>",
  "course_name": "<string>",
  "course_code": "<string — empty string if not mentioned>",
  "topics": "<comma-separated list of topics>",
  "bloom_targets": "<comma-separated Bloom levels, e.g. Apply, Analyze>",
  "co_mapping": "<comma-separated CO codes, e.g. CO1, CO2>",
  "question_count": <integer — default 5 if not specified>,
  "marks_per_question": <integer — default 5 if not specified>,
  "difficulty": "<Easy | Medium | Hard | Mixed — default Mixed>",
  "duration_minutes": <integer — 0 if not specified>,
  "department": "<string — empty if not mentioned>",
  "semester": "<string — empty if not mentioned>",
  "faculty_name": "<string — empty if not mentioned>",
  "extra_instructions": "<any other special instructions from the faculty>"
}
"""

# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------


def build_assessment_prompt(
    assessment_type: str,
    course_name: str,
    course_code: str,
    topics: str,
    bloom_targets: str,
    co_mapping: str,
    rag_context: str,
    question_count: int,
    marks_per_question: int,
    difficulty: str = "Mixed",
    extra_instructions: str = "",
    batch_hint: str = "",
) -> str:
    """Build the user-turn prompt for the Assessment Agent.

    Combines faculty requirements with retrieved RAG context and per-type
    guidance into a single structured prompt string for the LLM.

    Args:
        assessment_type: OBE assessment category (e.g. "Quiz").
        course_name: Full course name.
        course_code: Institutional course code (e.g. "CS3001").
        topics: Comma-separated list of topics to cover.
        bloom_targets: Target Bloom levels (e.g. "Apply, Analyse").
        co_mapping: Target Course Outcome codes (e.g. "CO1, CO2").
        rag_context: Retrieved knowledge-base text for grounding.
        question_count: Number of questions to generate.
        marks_per_question: Marks allocated per question.
        difficulty: Overall difficulty target (Easy/Medium/Hard/Mixed).
        extra_instructions: Any additional faculty instructions.
        batch_hint: Optional batching context injected by the Assessment Agent
            when generating in batches, e.g.
            "BATCH 2 OF 3 — Topics for this batch: Graphs, Sorting".
            Empty string when generating in a single call.

    Returns:
        str: Formatted prompt string ready for the LLM.
    """
    logger.debug(
        "build_assessment_prompt(): type=%s questions=%d marks=%d batch_hint=%r",
        assessment_type,
        question_count,
        marks_per_question,
        batch_hint or "(single call)",
    )

    type_guidance = _TYPE_GUIDANCE.get(
        assessment_type,
        "Generate academically rigorous questions appropriate for higher education.",
    )

    rag_section = (
        f"KNOWLEDGE-BASE CONTEXT (STRICT GROUNDING REQUIRED):\n"
        f"{'=' * 60}\n"
        f"{rag_context}\n"
        f"{'=' * 60}\n"
        f"IMPORTANT: You MUST base every question EXCLUSIVELY on the "
        f"knowledge-base context above. Do NOT introduce concepts, facts, "
        f"terminology, or examples that are not present in the provided "
        f"material. If the context does not cover a requested topic "
        f"sufficiently, generate fewer questions on that topic rather than "
        f"inventing content from general knowledge. Every question and its "
        f"answer must be verifiable from the context above.\n"
        if (rag_context or "").strip()
        else "KNOWLEDGE-BASE CONTEXT: No additional context retrieved — use general academic knowledge.\n"
    )

    extra_section = (
        f"\nADDITIONAL FACULTY INSTRUCTIONS:\n{extra_instructions}\n"
        if extra_instructions.strip()
        else ""
    )

    batch_section = (
        f"\nBATCH CONTEXT (READ CAREFULLY):\n{batch_hint}\n"
        f"Generate ONLY the questions listed above for the topics in this batch.\n"
        f"Use temporary question IDs Q1, Q2, … within this batch only — "
        f"the caller will renumber all questions globally after merging batches.\n"
        if batch_hint.strip()
        else ""
    )

    return (
        f"ASSESSMENT TYPE: {assessment_type}\n"
        f"COURSE: {course_name}" + (f" ({course_code})" if course_code else "") + "\n"
        f"TOPICS: {topics}\n"
        f"BLOOM LEVELS TO TARGET: {bloom_targets}\n"
        f"COURSE OUTCOMES TO MAP: {co_mapping}\n"
        f"NUMBER OF QUESTIONS: {question_count}\n"
        f"MARKS PER QUESTION: {marks_per_question}\n"
        f"DIFFICULTY: {difficulty}\n"
        f"\nTYPE-SPECIFIC GUIDANCE:\n{type_guidance}\n"
        f"\n{rag_section}"
        f"{extra_section}"
        f"{batch_section}\n"
        f"OUTPUT FORMAT:\n{ASSESSMENT_JSON_SCHEMA}"
    )


def build_planning_prompt(faculty_request: str) -> str:
    """Build the prompt for the Planning Agent to parse a faculty requirement.

    Args:
        faculty_request: Natural-language description of the desired assessment.

    Returns:
        str: Formatted planning prompt.
    """
    logger.debug("build_planning_prompt(): request length=%d", len(faculty_request))
    return (
        "Parse the following faculty requirement into a structured assessment plan.\n\n"
        f"FACULTY REQUIREMENT:\n{faculty_request}\n\n"
        "Return the JSON plan as specified."
    )


def build_reviewer_prompt(assessment_summary: str) -> str:
    """Build the user-turn prompt for the Reviewer Agent.

    The summary omits full answer key text to conserve tokens; it includes
    an ``answer_key_present`` boolean flag per question so the reviewer can
    penalise missing answer keys without seeing their content.

    Args:
        assessment_summary: JSON string produced by
            :func:`~agent.ReviewerAgent._build_assessment_summary` — contains
            assessment metadata, question texts, Bloom levels, CO mappings,
            difficulty levels, marks, and answer_key_present flags.

    Returns:
        str: Formatted reviewer prompt string ready for the LLM.
    """
    logger.debug(
        "build_reviewer_prompt(): summary length=%d chars", len(assessment_summary)
    )
    return (
        "Review the following OBE assessment and return your quality report.\n\n"
        "ASSESSMENT DATA:\n"
        f"{assessment_summary}\n\n"
        f"OUTPUT FORMAT:\n{REVIEWER_JSON_SCHEMA}"
    )
