"""
EduPilot — Batched generation test suite
Tests run without a live Groq API key by patching _invoke_llm.
Run: cd artifacts/edupilot && python test_batching.py
"""
from __future__ import annotations

import json
import sys
import os
import unittest
from pathlib import Path
from typing import List
from unittest.mock import patch

# Ensure the module directory is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

from models import (
    Assessment, AssessmentMetadata, AssessmentPlan, AssessmentType,
    BloomLevel, DifficultyLevel, Question,
)


# ---------------------------------------------------------------------------
# Config-patching helper
# ---------------------------------------------------------------------------

def _mock_config(**overrides):
    """Return a new EduPilotConfig with given overrides (handles frozen dataclass)."""
    import config as cfg_module
    defaults = dict(
        groq_api_key=None,
        groq_model_name="test-model",
        groq_max_tokens=6000,
        embedding_model_name="test",
        vectorstore_path=Path("/tmp"),
        knowledge_dir=Path("/tmp"),
        log_level="INFO",
        runs_dir=Path("/tmp"),
        batch_threshold=30,
        batch_size=8,
        batch_inter_call_delay_s=1.0,  # use 1 s in tests so they don't actually sleep
    )
    defaults.update(overrides)
    return cfg_module.EduPilotConfig(**defaults)


# ---------------------------------------------------------------------------
# Assessment / plan factories
# ---------------------------------------------------------------------------

def _make_plan(
    question_count: int = 25,
    assessment_type: str = "Question Bank",
    topics: str = "Binary Trees, Graphs, Sorting, Hashing, Dynamic Programming",
) -> AssessmentPlan:
    return AssessmentPlan(
        assessment_type=assessment_type,
        course_name="Data Structures",
        course_code="CS301",
        topics=topics,
        bloom_targets="Remember, Understand, Apply, Analyze",
        co_mapping="CO1, CO2, CO3",
        question_count=question_count,
        marks_per_question=5,
        difficulty="Mixed",
        duration_minutes=0,
        department="CSE",
        semester="Semester 5",
        faculty_name="Dr. Test",
        extra_instructions="",
    )


def _make_batch_json(count: int, start_idx: int = 1) -> str:
    """Build a minimal valid assessment JSON with *count* questions."""
    questions = [
        {
            "question_id": f"Q{i}",          # per-batch IDs; merge will renumber
            "question_text": f"Explain concept {start_idx + i - 1} in detail.",
            "question_type": "Short Answer",
            "bloom_level": "Understand",
            "co_mapping": ["CO1"],
            "difficulty": "Medium",
            "marks": 5,
            "answer_key": f"Model answer for question {start_idx + i - 1}.",
            "options": [],
            "notes": "",
        }
        for i in range(1, count + 1)
    ]
    return json.dumps({
        "assessment_title": "Test Question Bank",
        "course_code": "CS301",
        "course_name": "Data Structures",
        "duration_minutes": 0,
        "instructions": "Answer all questions.",
        "generation_notes": f"Batch of {count} questions.",
        "questions": questions,
    })


def _fake_assessments(counts: List[int]) -> List[Assessment]:
    """Build Assessment objects with the given question counts (Q1..Qn per batch)."""
    result = []
    q_num = 1
    for count in counts:
        qs = [
            Question(
                question_id=f"Q{i}",   # per-batch IDs; merge renumbers globally
                question_text=f"Question text {q_num + i - 1}",
                bloom_level=BloomLevel.UNDERSTAND,
                co_mapping=["CO1"],
                difficulty=DifficultyLevel.MEDIUM,
                marks=5,
                answer_key=f"Answer {q_num + i - 1}",
            )
            for i in range(1, count + 1)
        ]
        meta = AssessmentMetadata(
            title="Test QB",
            course_code="CS301",
            course_name="Data Structures",
            assessment_type=AssessmentType.QUESTION_BANK,
        )
        result.append(Assessment(metadata=meta, questions=qs))
        q_num += count
    return result


# ---------------------------------------------------------------------------
# Helper: batch responses in order matching batch specs
# ---------------------------------------------------------------------------

def _responses_for_specs(specs):
    """Build one fake JSON response per (count, topics) spec."""
    responses = []
    idx = 1
    for count, _ in specs:
        responses.append(_make_batch_json(count, start_idx=idx))
        idx += count
    return responses


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBatchingThreshold(unittest.TestCase):
    """_needs_batching() returns correct decisions."""

    def _agent(self):
        from agent import AssessmentAgent
        return AssessmentAgent()

    def test_small_qb_not_batched(self):
        """8 QB questions fit in one call (estimate threshold ≈ 12)."""
        agent = self._agent()
        plan = _make_plan(question_count=8, assessment_type="Question Bank")
        with patch("agent.config", _mock_config(batch_threshold=30)):
            self.assertFalse(agent._needs_batching(plan))

    def test_quiz_12_not_batched(self):
        """12 Quiz questions don't batch (tokens_per_q=150 → threshold ~30)."""
        agent = self._agent()
        plan = _make_plan(question_count=12, assessment_type="Quiz")
        with patch("agent.config", _mock_config(batch_threshold=30)):
            self.assertFalse(agent._needs_batching(plan))

    def test_large_qb_batched(self):
        """25 QB questions exceed token estimate threshold (~12 for QB)."""
        agent = self._agent()
        plan = _make_plan(question_count=25, assessment_type="Question Bank")
        with patch("agent.config", _mock_config(batch_threshold=30)):
            self.assertTrue(agent._needs_batching(plan))

    def test_config_threshold_as_hard_cap(self):
        """batch_threshold=5 forces batching even for Quiz at 6 questions."""
        agent = self._agent()
        plan = _make_plan(question_count=6, assessment_type="Quiz")
        # Normally 6 Quiz Qs wouldn't batch; threshold=5 forces it.
        with patch("agent.config", _mock_config(batch_threshold=5)):
            self.assertTrue(agent._needs_batching(plan))

    def test_threshold_boundary_below(self):
        """question_count == threshold → no batching (> not >=)."""
        agent = self._agent()
        # QB estimate threshold ≈ 12; at exactly 12 should be SINGLE
        plan = _make_plan(question_count=12, assessment_type="Question Bank")
        with patch("agent.config", _mock_config(batch_threshold=30)):
            self.assertFalse(agent._needs_batching(plan))

    def test_threshold_boundary_above(self):
        """question_count == threshold + 1 → batching triggers."""
        agent = self._agent()
        plan = _make_plan(question_count=13, assessment_type="Question Bank")
        with patch("agent.config", _mock_config(batch_threshold=30)):
            self.assertTrue(agent._needs_batching(plan))


class TestAllocatePerTopic(unittest.TestCase):
    """_allocate_per_topic() distributes questions fairly across topics."""

    def _agent(self):
        from agent import AssessmentAgent
        return AssessmentAgent()

    def test_equal_split_no_remainder(self):
        """25q / 5 topics → every topic gets exactly 5."""
        agent = self._agent()
        allocs = agent._allocate_per_topic("A, B, C, D, E", 25)
        self.assertEqual(len(allocs), 5)
        counts = [c for _, c in allocs]
        self.assertEqual(sum(counts), 25)
        self.assertTrue(all(c == 5 for c in counts))

    def test_remainder_distributed_one_extra(self):
        """10q / 3 topics → [4, 3, 3]; max-min == 1."""
        agent = self._agent()
        allocs = agent._allocate_per_topic("T1, T2, T3", 10)
        self.assertEqual(len(allocs), 3)
        counts = [c for _, c in allocs]
        self.assertEqual(sum(counts), 10)
        self.assertEqual(max(counts) - min(counts), 1)
        # First topic gets the extra question (largest-remainder method)
        self.assertEqual(counts[0], 4)
        self.assertEqual(counts[1], 3)
        self.assertEqual(counts[2], 3)

    def test_7q_5topics_max_min_le_1(self):
        """7q / 5 topics → [2, 2, 1, 1, 1]; balanced (max-min ≤ 1)."""
        agent = self._agent()
        allocs = agent._allocate_per_topic("T1, T2, T3, T4, T5", 7)
        counts = [c for _, c in allocs]
        self.assertEqual(sum(counts), 7)
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_more_topics_than_questions_zero_quota_omitted(self):
        """3q / 5 topics → only 3 topics returned (T4, T5 get 0 and are dropped)."""
        agent = self._agent()
        allocs = agent._allocate_per_topic("T1, T2, T3, T4, T5", 3)
        # Zero-quota topics must be omitted from the result
        self.assertEqual(len(allocs), 3)
        counts = [c for _, c in allocs]
        self.assertEqual(sum(counts), 3)
        self.assertTrue(all(c >= 1 for c in counts))

    def test_single_topic_gets_all(self):
        """Single topic gets every question."""
        agent = self._agent()
        allocs = agent._allocate_per_topic("Sorting", 12)
        self.assertEqual(len(allocs), 1)
        self.assertEqual(allocs[0][0], "Sorting")
        self.assertEqual(allocs[0][1], 12)

    def test_empty_topics_uses_general(self):
        """Empty topic string → 'General' placeholder receives all questions."""
        agent = self._agent()
        allocs = agent._allocate_per_topic("", 5)
        self.assertEqual(len(allocs), 1)
        self.assertEqual(allocs[0][0], "General")
        self.assertEqual(allocs[0][1], 5)

    def test_total_always_matches_requested(self):
        """Sum of allocated counts always equals the requested total."""
        agent = self._agent()
        for total, topics in [
            (1,  "A"),
            (5,  "A, B, C"),
            (10, "A, B, C"),
            (16, "A, B, C, D, E"),
            (30, "A, B, C, D"),
        ]:
            allocs = agent._allocate_per_topic(topics, total)
            self.assertEqual(
                sum(c for _, c in allocs),
                total,
                f"Total mismatch for total={total}, topics={topics!r}",
            )


class TestBuildBatchSpecs(unittest.TestCase):
    """_build_batch_specs() honours per-topic quotas and batch_size cap."""

    def _agent(self):
        from agent import AssessmentAgent
        return AssessmentAgent()

    def test_exact_multiple(self):
        """24q / 3 topics / bs=8 → 3 batches of exactly 8 (one per topic)."""
        agent = self._agent()
        # 3 topics × 8 = 24 exactly — each topic quota fits in one bs=8 batch
        plan = _make_plan(question_count=24, topics="T1, T2, T3")
        with patch("agent.config", _mock_config(batch_size=8)):
            specs = agent._build_batch_specs(plan)
        self.assertEqual(len(specs), 3)
        self.assertEqual(sum(c for c, _ in specs), 24)
        self.assertTrue(all(c == 8 for c, _ in specs))

    def test_25_questions_four_balanced_batches(self):
        """25q / 4 topics / bs=8 → 4 batches [7, 6, 6, 6] (largest-remainder)."""
        agent = self._agent()
        # 4 topics: base=6, remainder=1 → T1 gets 7, T2-T4 get 6 each
        plan = _make_plan(question_count=25, topics="T1, T2, T3, T4")
        with patch("agent.config", _mock_config(batch_size=8)):
            specs = agent._build_batch_specs(plan)
        self.assertEqual(len(specs), 4)
        self.assertEqual(sum(c for c, _ in specs), 25)
        counts = [c for c, _ in specs]
        self.assertEqual(counts, [7, 6, 6, 6])
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_25_questions_batch_size_10_balanced(self):
        """25q / 3 topics / bs=10 → 3 batches [9, 8, 8] (largest-remainder)."""
        agent = self._agent()
        # 3 topics: base=8, remainder=1 → T1 gets 9, T2-T3 get 8 each
        plan = _make_plan(question_count=25, topics="T1, T2, T3")
        with patch("agent.config", _mock_config(batch_size=10)):
            specs = agent._build_batch_specs(plan)
        self.assertEqual([c for c, _ in specs], [9, 8, 8])

    def test_per_topic_quotas_equal_split_default_topics(self):
        """25q / 5 default topics / bs=8 → 5 batches of 5, each topic once."""
        agent = self._agent()
        plan = _make_plan(question_count=25)  # uses 5-topic default
        with patch("agent.config", _mock_config(batch_size=8)):
            specs = agent._build_batch_specs(plan)
        self.assertEqual(len(specs), 5)
        self.assertEqual(sum(c for c, _ in specs), 25)
        self.assertTrue(all(c == 5 for c, _ in specs))
        topics_seen = [t for _, t in specs]
        self.assertEqual(
            sorted(topics_seen),
            sorted(["Binary Trees", "Graphs", "Sorting", "Hashing", "Dynamic Programming"]),
        )

    def test_quota_above_batch_size_splits_into_sub_batches(self):
        """A single topic with quota > batch_size gets split into ≤batch_size sub-batches."""
        agent = self._agent()
        # 20q, 1 topic, bs=6 → ceil(20/6) = 4 sub-batches for that topic
        plan = _make_plan(question_count=20, topics="Solo Topic")
        with patch("agent.config", _mock_config(batch_size=6)):
            specs = agent._build_batch_specs(plan)
        self.assertEqual(sum(c for c, _ in specs), 20)
        for c, t in specs:
            self.assertLessEqual(c, 6)
            self.assertEqual(t, "Solo Topic")

    def test_remainder_distributed_across_topics(self):
        """10q / 3 topics / bs=8 → [4, 3, 3] — max-min ≤ 1, total=10."""
        agent = self._agent()
        plan = _make_plan(question_count=10, topics="T1, T2, T3")
        with patch("agent.config", _mock_config(batch_size=8)):
            specs = agent._build_batch_specs(plan)
        counts = [c for c, _ in specs]
        self.assertEqual(sum(counts), 10)
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_total_always_equals_plan(self):
        for total in [1, 5, 8, 9, 16, 24, 25, 30]:
            agent = self._agent()
            plan = _make_plan(question_count=total)
            with patch("agent.config", _mock_config(batch_size=8)):
                specs = agent._build_batch_specs(plan)
            self.assertEqual(
                sum(c for c, _ in specs),
                total,
                f"Batch total mismatch for question_count={total}",
            )

    def test_no_batch_exceeds_batch_size(self):
        for total in [9, 17, 25]:
            agent = self._agent()
            plan = _make_plan(question_count=total)
            with patch("agent.config", _mock_config(batch_size=8)):
                specs = agent._build_batch_specs(plan)
            for c, _ in specs:
                self.assertLessEqual(c, 8, f"Batch exceeds batch_size for total={total}")


class TestMergeBatches(unittest.TestCase):
    """_merge_batches() renumbers globally and handles edge cases."""

    def _agent(self):
        from agent import AssessmentAgent
        return AssessmentAgent()

    def test_sequential_ids_after_merge(self):
        agent = self._agent()
        plan = _make_plan(question_count=24)
        batches = _fake_assessments([8, 8, 8])
        merged = agent._merge_batches(batches, plan)
        ids = [q.question_id for q in merged.questions]
        self.assertEqual(ids, [f"Q{i}" for i in range(1, 25)])

    def test_total_count_correct(self):
        agent = self._agent()
        plan = _make_plan(question_count=25)
        batches = _fake_assessments([8, 8, 8, 1])
        merged = agent._merge_batches(batches, plan)
        self.assertEqual(merged.question_count, 25)

    def test_unique_ids(self):
        agent = self._agent()
        plan = _make_plan(question_count=16)
        batches = _fake_assessments([8, 8])
        merged = agent._merge_batches(batches, plan)
        ids = [q.question_id for q in merged.questions]
        self.assertEqual(len(set(ids)), 16)

    def test_question_text_preserved(self):
        """Question text and answer keys must survive the merge unchanged."""
        agent = self._agent()
        plan = _make_plan(question_count=3)
        batches = _fake_assessments([3])
        merged = agent._merge_batches(batches, plan)
        for i, q in enumerate(merged.questions):
            self.assertIn("Question text", q.question_text)
            self.assertIn("Answer", q.answer_key)

    def test_shortfall_warning_in_notes(self):
        """When merged count < plan.question_count, generation_notes warns."""
        agent = self._agent()
        plan = _make_plan(question_count=25)
        batches = _fake_assessments([10, 10])   # only 20 returned
        merged = agent._merge_batches(batches, plan)
        self.assertEqual(merged.question_count, 20)
        self.assertIn("WARNING", merged.generation_notes)
        self.assertIn("shortfall", merged.generation_notes)

    def test_empty_batches_raises(self):
        agent = self._agent()
        plan = _make_plan(question_count=8)
        with self.assertRaises(ValueError):
            agent._merge_batches([], plan)

    def test_first_batch_metadata_wins(self):
        agent = self._agent()
        plan = _make_plan(question_count=16)
        batches = _fake_assessments([8, 8])
        batches[1].metadata.title = "Should Not Win"
        merged = agent._merge_batches(batches, plan)
        self.assertEqual(merged.metadata.title, "Test QB")


class TestRateLimitRetry(unittest.TestCase):
    """_invoke_llm_with_rate_retry() sleeps and retries on 413/429."""

    def test_succeeds_on_first_try(self):
        with patch("agent._invoke_llm", return_value="ok") as mock_llm:
            from agent import _invoke_llm_with_rate_retry
            result = _invoke_llm_with_rate_retry("sys", "usr", max_rate_retries=2)
        self.assertEqual(result, "ok")
        self.assertEqual(mock_llm.call_count, 1)

    def test_retries_on_413_sleeps_then_succeeds(self):
        side_effects = [
            RuntimeError("Groq LLM call failed: 413 rate_limit_exceeded"),
            "success_after_retry",
        ]
        with patch("agent._invoke_llm", side_effect=side_effects) as mock_llm, \
             patch("agent.time.sleep") as mock_sleep:
            from agent import _invoke_llm_with_rate_retry
            result = _invoke_llm_with_rate_retry(
                "sys", "usr", max_rate_retries=2, rate_retry_delay_s=1.0
            )
        self.assertEqual(result, "success_after_retry")
        self.assertEqual(mock_llm.call_count, 2)
        mock_sleep.assert_called_once_with(1.0)  # delay × attempt_number(1)

    def test_retries_on_429_too_many_requests(self):
        side_effects = [RuntimeError("429 too many requests"), "ok"]
        with patch("agent._invoke_llm", side_effect=side_effects), \
             patch("agent.time.sleep") as mock_sleep:
            from agent import _invoke_llm_with_rate_retry
            result = _invoke_llm_with_rate_retry(
                "sys", "usr", max_rate_retries=2, rate_retry_delay_s=2.0
            )
        self.assertEqual(result, "ok")
        mock_sleep.assert_called_once()

    def test_retries_on_rate_limit_wording(self):
        """'rate limit' in error text also triggers retry."""
        side_effects = [RuntimeError("rate limit exceeded by model"), "ok"]
        with patch("agent._invoke_llm", side_effect=side_effects), \
             patch("agent.time.sleep"):
            from agent import _invoke_llm_with_rate_retry
            result = _invoke_llm_with_rate_retry(
                "sys", "usr", max_rate_retries=2, rate_retry_delay_s=0.1
            )
        self.assertEqual(result, "ok")

    def test_abort_on_persistent_rate_limit(self):
        side_effects = [RuntimeError("413 rate_limit_exceeded")] * 5
        with patch("agent._invoke_llm", side_effect=side_effects), \
             patch("agent.time.sleep"):
            from agent import _invoke_llm_with_rate_retry
            with self.assertRaises(RuntimeError):
                _invoke_llm_with_rate_retry(
                    "sys", "usr", max_rate_retries=2, rate_retry_delay_s=0.001
                )

    def test_non_rate_limit_propagates_immediately(self):
        """Network timeout must not be silently retried."""
        side_effects = [RuntimeError("Connection timeout"), "should_not_reach"]
        with patch("agent._invoke_llm", side_effect=side_effects) as mock_llm, \
             patch("agent.time.sleep") as mock_sleep:
            from agent import _invoke_llm_with_rate_retry
            with self.assertRaises(RuntimeError):
                _invoke_llm_with_rate_retry("sys", "usr", max_rate_retries=2)
        self.assertEqual(mock_llm.call_count, 1)
        mock_sleep.assert_not_called()


class TestSingleCallPath(unittest.TestCase):
    """_generate_single() works as before; no sleep called."""

    def test_single_call_returns_correct_assessment(self):
        plan = _make_plan(question_count=8, assessment_type="Question Bank")
        fake_json = _make_batch_json(8, start_idx=1)
        with patch("agent._invoke_llm", return_value=fake_json), \
             patch("agent.time.sleep") as mock_sleep, \
             patch("agent.config", _mock_config(batch_threshold=30)):
            from agent import AssessmentAgent
            result = AssessmentAgent()._generate_single(plan, "", [])
        self.assertEqual(result.question_count, 8)
        mock_sleep.assert_not_called()


class TestBatchedGenerationIntegration(unittest.TestCase):
    """AssessmentAgent.generate() end-to-end with mocked _invoke_llm."""

    def test_25_questions_sequential_ids_and_answer_keys(self):
        """25-question QB → merged result has 25 sequential IDs, all with answer keys."""
        plan = _make_plan(question_count=25, assessment_type="Question Bank")
        cfg = _mock_config(batch_size=8, batch_threshold=30, batch_inter_call_delay_s=0.0)

        from agent import AssessmentAgent
        agent = AssessmentAgent()

        # Get the real batch specs so responses line up exactly
        with patch("agent.config", cfg):
            specs = agent._build_batch_specs(plan)

        responses = _responses_for_specs(specs)

        with patch("agent._invoke_llm", side_effect=responses), \
             patch("agent.time.sleep") as mock_sleep, \
             patch("agent.config", cfg):
            result = agent.generate(plan, rag_context="", sources=[])

        self.assertEqual(result.question_count, 25, "Must return exactly 25 questions")
        ids = [q.question_id for q in result.questions]
        self.assertEqual(ids, [f"Q{i}" for i in range(1, 26)], "IDs must be Q1..Q25")
        self.assertTrue(
            all(q.answer_key.strip() for q in result.questions),
            "Every question must have a non-empty answer key",
        )
        # With a 0 s cooldown window the elapsed time always covers the window
        # (elapsed-aware implementation) — no sleep should ever fire.
        mock_sleep.assert_not_called()

    def test_cooldown_sleeps_only_window_remainder(self):
        """Sleep before batch N covers only the remainder of the TPM window."""
        # 2 topics → 2 batches of 8 each; exactly one inter-batch cooldown.
        plan = _make_plan(question_count=16, assessment_type="Question Bank",
                          topics="Topic A, Topic B")
        cfg = _mock_config(batch_size=8, batch_threshold=30, batch_inter_call_delay_s=62.0)

        from agent import AssessmentAgent
        agent = AssessmentAgent()
        with patch("agent.config", cfg):
            specs = agent._build_batch_specs(plan)
        self.assertEqual(len(specs), 2, "Expect exactly 2 batches for this test")
        responses = _responses_for_specs(specs)

        # Monotonic sequence: batch 1 call starts at t=100; when batch 2's
        # cooldown is computed the clock reads t=140 (call took 40 s) →
        # expected sleep = 62 − 40 = 22 s.  Then batch 2 call starts at 162.
        clock = iter([100.0, 140.0, 162.0])
        with patch("agent._invoke_llm", side_effect=responses), \
             patch("agent.time.monotonic", side_effect=lambda: next(clock)), \
             patch("agent.time.sleep") as mock_sleep, \
             patch("agent.config", cfg):
            result = agent.generate(plan, rag_context="", sources=[])

        self.assertEqual(result.question_count, 16)
        mock_sleep.assert_called_once()
        self.assertAlmostEqual(mock_sleep.call_args[0][0], 22.0)

    def test_cooldown_uses_latest_call_start_after_parse_retry(self):
        """A parse retry updates the timestamp; the next batch measures from
        the retry's call start, not the obsolete first attempt."""
        # 2 topics → 2 batches of 8 each; exactly one inter-batch cooldown.
        plan = _make_plan(question_count=16, assessment_type="Question Bank",
                          topics="Topic A, Topic B")
        cfg = _mock_config(batch_size=8, batch_threshold=30, batch_inter_call_delay_s=62.0)

        from agent import AssessmentAgent
        agent = AssessmentAgent()
        with patch("agent.config", cfg):
            specs = agent._build_batch_specs(plan)
        self.assertEqual(len(specs), 2, "Expect exactly 2 batches for this test")
        good = _responses_for_specs(specs)
        # Batch 1 attempt 1 returns garbage (parse error), attempt 2 succeeds.
        responses = ["not json at all {{{"] + good

        # Clock: batch1 attempt1 starts t=0; attempt2 (retry) starts t=50;
        # batch2 cooldown computed at t=80 → elapsed since LATEST start is
        # 30 s → sleep 32 s.  (Measured from the obsolete t=0 it would have
        # been 62−80 → no sleep, recreating the 413 risk.)
        clock = iter([0.0, 50.0, 80.0, 120.0])
        with patch("agent._invoke_llm", side_effect=responses), \
             patch("agent.time.monotonic", side_effect=lambda: next(clock)), \
             patch("agent.time.sleep") as mock_sleep, \
             patch("agent.config", cfg):
            result = agent.generate(plan, rag_context="", sources=[])

        self.assertEqual(result.question_count, 16)
        mock_sleep.assert_called_once()
        self.assertAlmostEqual(mock_sleep.call_args[0][0], 32.0)

    def test_batch_progress_callback_fired_per_batch(self):
        """batch_progress_callback is called once per batch with correct signature."""
        plan = _make_plan(question_count=16, assessment_type="Question Bank")
        cfg = _mock_config(batch_size=8, batch_threshold=30, batch_inter_call_delay_s=0.0)

        from agent import AssessmentAgent
        agent = AssessmentAgent()

        with patch("agent.config", cfg):
            specs = agent._build_batch_specs(plan)

        responses = _responses_for_specs(specs)
        cb_calls = []

        def _cb(batch_idx, total_batches, qs_start, qs_end, total_qs):
            cb_calls.append((batch_idx, total_batches, qs_start, qs_end, total_qs))

        with patch("agent._invoke_llm", side_effect=responses), \
             patch("agent.time.sleep"), \
             patch("agent.config", cfg):
            agent.generate(plan, "", [], batch_progress_callback=_cb)

        self.assertEqual(len(cb_calls), len(specs), "One callback per batch")
        # Last callback: qs_end=16, total_qs=16
        last = cb_calls[-1]
        self.assertEqual(last[3], 16)   # qs_end
        self.assertEqual(last[4], 16)   # total_qs
        # First callback: qs_start=1
        self.assertEqual(cb_calls[0][2], 1)  # qs_start

    def test_abort_on_persistent_parse_failure(self):
        """Persistent bad JSON on first batch → ValueError, not a partial result."""
        plan = _make_plan(question_count=16, assessment_type="Question Bank")
        cfg = _mock_config(batch_size=8, batch_threshold=30, batch_inter_call_delay_s=0.0)

        from agent import AssessmentAgent
        agent = AssessmentAgent(max_retries=1)

        with patch("agent._invoke_llm", return_value="NOT_VALID_JSON {{ "), \
             patch("agent.time.sleep"), \
             patch("agent.config", cfg):
            with self.assertRaises((ValueError, RuntimeError)):
                agent.generate(plan, "", [])

    def test_small_request_uses_single_call_no_sleep(self):
        """8 QB questions → single LLM call, zero sleeps."""
        plan = _make_plan(question_count=8, assessment_type="Question Bank")
        cfg = _mock_config(batch_size=8, batch_threshold=30, batch_inter_call_delay_s=0.0)
        fake_json = _make_batch_json(8, start_idx=1)

        from agent import AssessmentAgent
        with patch("agent._invoke_llm", return_value=fake_json) as mock_llm, \
             patch("agent.time.sleep") as mock_sleep, \
             patch("agent.config", cfg):
            result = AssessmentAgent().generate(plan, "", [])

        self.assertEqual(result.question_count, 8)
        self.assertEqual(mock_llm.call_count, 1, "Single call path must use exactly 1 LLM call")
        mock_sleep.assert_not_called()

    def test_413_retry_during_batch(self):
        """A 413 on batch 1 is retried (with sleep); success on retry gives full result."""
        # 2 topics → 2 batches; easy to construct an exact side-effect list.
        plan = _make_plan(question_count=16, assessment_type="Question Bank",
                          topics="Topic A, Topic B")
        cfg = _mock_config(batch_size=8, batch_threshold=30, batch_inter_call_delay_s=0.1)

        from agent import AssessmentAgent
        agent = AssessmentAgent()

        with patch("agent.config", cfg):
            specs = agent._build_batch_specs(plan)
        self.assertEqual(len(specs), 2, "Expect exactly 2 batches for this test")

        responses = _responses_for_specs(specs)
        # Inject a 413 before the first real response; retry succeeds, then batch 2.
        llm_side_effects = [
            RuntimeError("413 rate_limit_exceeded"),
            responses[0],   # retry for batch 1 succeeds
            responses[1],   # batch 2 succeeds normally
        ]

        with patch("agent._invoke_llm", side_effect=llm_side_effects) as mock_llm, \
             patch("agent.time.sleep") as mock_sleep, \
             patch("agent.config", cfg):
            result = agent.generate(plan, "", [])

        self.assertEqual(result.question_count, 16)
        # LLM called 3 times: 413 + retry for batch 1 + normal for batch 2
        self.assertEqual(mock_llm.call_count, 3)
        # sleep must have been called (at least once for the 413 rate-limit retry)
        self.assertGreaterEqual(mock_sleep.call_count, 1)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("EduPilot Batching Unit Tests")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestBatchingThreshold,
        TestAllocatePerTopic,
        TestBuildBatchSpecs,
        TestMergeBatches,
        TestRateLimitRetry,
        TestSingleCallPath,
        TestBatchedGenerationIntegration,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
