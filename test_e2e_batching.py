"""
EduPilot — End-to-end batching test (requires GROQ_API_KEY in env)
Calls AssessmentAgent.generate() with a real 25-question Question Bank request.
Expected runtime: ~3–5 minutes (3 × 62 s inter-batch delays + LLM time).

Run: cd artifacts/edupilot && python test_e2e_batching.py
"""
from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from models import AssessmentPlan
from agent import AssessmentAgent

# ── Plan ─────────────────────────────────────────────────────────────────────
PLAN = AssessmentPlan(
    assessment_type="Question Bank",
    course_name="Data Structures and Algorithms",
    course_code="CS301",
    topics="Binary Trees, Graphs, Sorting, Hashing, Dynamic Programming",
    bloom_targets="Remember, Understand, Apply, Analyze",
    co_mapping="CO1, CO2, CO3",
    question_count=25,
    marks_per_question=5,
    difficulty="Mixed",
    duration_minutes=0,
    department="Computer Science and Engineering",
    semester="Semester 5",
    faculty_name="Dr. EduPilot Test",
    extra_instructions="",
)

def _progress(batch_idx, total_batches, qs_start, qs_end, total_qs):
    print(f"  ✓ Batch {batch_idx}/{total_batches} complete — "
          f"questions {qs_start}–{qs_end} of {total_qs}")

def main():
    print("=" * 60)
    print("EduPilot E2E — 25-question Question Bank (live Groq API)")
    print("=" * 60)

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set — cannot run E2E test.")
        sys.exit(1)
    print(f"GROQ_API_KEY present ({len(api_key)} chars)")

    agent = AssessmentAgent()

    # Verify batching is triggered
    needs = agent._needs_batching(PLAN)
    print(f"_needs_batching() → {needs}")
    assert needs, "Expected _needs_batching() == True for 25-question QB"

    # Show batch plan
    specs = agent._build_batch_specs(PLAN)
    print(f"Batch plan: {len(specs)} batches")
    for i, (count, topics) in enumerate(specs, 1):
        print(f"  Batch {i}: {count} questions | topics: {topics}")

    print("\nStarting generation (this takes several minutes due to TPM delays)…")
    t0 = time.monotonic()

    assessment = agent.generate(
        plan=PLAN,
        rag_context="",
        sources=[],
        batch_progress_callback=_progress,
    )

    elapsed = time.monotonic() - t0
    print(f"\nGeneration complete in {elapsed:.0f} s")

    # ── Assertions ────────────────────────────────────────────────────────────
    errors = []

    if assessment.question_count != 25:
        errors.append(
            f"FAIL: expected 25 questions, got {assessment.question_count}"
            + (" (shortfall noted in generation_notes)"
               if "WARNING" in (assessment.generation_notes or "")
               else "")
        )
    else:
        print(f"✓ question_count = {assessment.question_count}")

    ids = [q.question_id for q in assessment.questions]
    expected_ids = [f"Q{i}" for i in range(1, assessment.question_count + 1)]
    if ids != expected_ids:
        errors.append(f"FAIL: IDs not sequential. Got {ids[:5]}… (first 5)")
    else:
        print(f"✓ IDs sequential Q1–Q{assessment.question_count}")

    missing_keys = [q.question_id for q in assessment.questions
                    if not (q.answer_key and q.answer_key.strip())]
    if missing_keys:
        errors.append(f"FAIL: missing answer keys on: {missing_keys}")
    else:
        print(f"✓ All {assessment.question_count} questions have answer keys")

    if not assessment.metadata.course_name:
        errors.append("FAIL: metadata.course_name is empty")
    else:
        print(f"✓ metadata.course_name = {assessment.metadata.course_name!r}")

    if assessment.total_marks != assessment.question_count * PLAN.marks_per_question:
        # Not a hard failure — LLM may vary marks slightly
        print(f"⚠ total_marks = {assessment.total_marks} "
              f"(expected {assessment.question_count * PLAN.marks_per_question})")
    else:
        print(f"✓ total_marks = {assessment.total_marks}")

    if assessment.generation_notes and "WARNING" in assessment.generation_notes:
        print(f"⚠ generation_notes warning: {assessment.generation_notes[:200]}")

    print("\n── Question sample (first 3) ──")
    for q in assessment.questions[:3]:
        print(f"  {q.question_id} [{q.bloom_level.value} / {q.difficulty.value}]")
        print(f"    {q.question_text[:100]}…")
        print(f"    Answer: {q.answer_key[:80]}…")

    if errors:
        print("\n" + "=" * 60)
        for e in errors:
            print(e)
        print("=" * 60)
        sys.exit(1)

    print("\n✅ All assertions passed.")

if __name__ == "__main__":
    main()
