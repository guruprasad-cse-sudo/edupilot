"""
EduPilot — Robustness verification (Task #40 code-review follow-up)
Run with: python verify_robustness.py
"""
from __future__ import annotations
import json, sys, traceback, urllib.request, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from models import (
    Assessment, AssessmentMetadata, AssessmentType,
    BloomLevel, DifficultyLevel, Question, SourceAttribution,
    ReviewerResult, AnalyticsReport,
)
from agent import AnalyticsAgent, ReviewerAgent

PASS, FAIL = "✅ PASS", "❌ FAIL"
results: list[tuple[str, bool, str]] = []

def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {PASS if ok else FAIL}  {name}" + (f"  [{detail}]" if detail else ""))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def full_assessment() -> Assessment:
    src = SourceAttribution("lecture_notes.pdf", 3, 0.92, "Chapter 3")
    qs = [
        Question("Q1","Define an operating system.",BloomLevel.REMEMBER,["CO1"],DifficultyLevel.EASY,5,"An OS manages hardware.",sources=[src]),
        Question("Q2","Explain virtual memory.",BloomLevel.UNDERSTAND,["CO1","CO2"],DifficultyLevel.EASY,5,"VM uses disk as extended RAM.",sources=[src]),
        Question("Q3","Apply Round-Robin scheduling.",BloomLevel.APPLY,["CO2"],DifficultyLevel.MEDIUM,10,"Distribute quanta equally.",sources=[src]),
        Question("Q4","Analyse preemptive vs non-preemptive.",BloomLevel.ANALYZE,["CO2","CO3"],DifficultyLevel.MEDIUM,10,"Preemptive: OS can interrupt.",sources=[src]),
        Question("Q5","Evaluate FCFS vs SJF trade-offs.",BloomLevel.EVALUATE,["CO3"],DifficultyLevel.HARD,10,"FCFS convoy; SJF starvation.",sources=[src]),
        Question("Q6","Design a deadlock prevention strategy.",BloomLevel.CREATE,["CO3","CO4"],DifficultyLevel.HARD,15,"Use Banker's Algorithm.",sources=[src]),
    ]
    meta = AssessmentMetadata("OS Internal","CS3001","Operating Systems",
                              AssessmentType.INTERNAL,semester="Sem 5",duration_minutes=90,department="CS")
    return Assessment(metadata=meta, questions=qs)

def empty_assessment() -> Assessment:
    return Assessment(metadata=AssessmentMetadata("Empty","","Test",AssessmentType.QUIZ), questions=[])

def dup_assessment() -> Assessment:
    qs = [
        Question("Q1","What is an operating system and what are its main functions?",
                 BloomLevel.REMEMBER,["CO1"],DifficultyLevel.EASY,5,"An OS manages hardware."),
        Question("Q2","What is an operating system and what are its primary functions?",
                 BloomLevel.REMEMBER,["CO1"],DifficultyLevel.EASY,5,"An OS manages hardware."),
        Question("Q3","Explain process scheduling.",BloomLevel.UNDERSTAND,["CO2"],DifficultyLevel.MEDIUM,10,"Scheduling allocates CPU."),
    ]
    return Assessment(metadata=AssessmentMetadata("Dup","X","OS",AssessmentType.QUIZ), questions=qs)

reviewer = ReviewerAgent()
analytics = AnalyticsAgent()

def _parse(payload: dict) -> ReviewerResult:
    return reviewer._parse_reviewer_response(json.dumps(payload))

def _parse_raw(s: str) -> ReviewerResult:
    return reviewer._parse_reviewer_response(s)

# ===========================================================================
# A — malformed-but-valid JSON payloads fed directly into _parse_reviewer_response
# ===========================================================================
print("\n" + "="*65)
print("EduPilot Robustness Verification — Task #40 follow-up")
print("="*65)

print("\n[A1] bloom_coverage = null")
try:
    r = _parse({"quality_score":70,"strengths":[],"weaknesses":[],"suggestions":[],
                "duplicate_question_ids":[],"bloom_coverage":None,
                "difficulty_balance_ok":True,"reviewer_notes":""})
    check("no crash", True)
    check("bloom_coverage has 6 keys", len(r.bloom_coverage)==6, str(len(r.bloom_coverage)))
    check("all levels False", all(v is False for v in r.bloom_coverage.values()))
    check("error is None", r.error is None)
except Exception as e:
    check("no crash — EXCEPTION", False, str(e)); traceback.print_exc()

print("\n[A2] bloom_coverage = list of level names")
try:
    r = _parse({"quality_score":75,"strengths":["Good"],"weaknesses":[],
                "suggestions":[],"duplicate_question_ids":[],
                "bloom_coverage":["Remember","Apply","Evaluate"],
                "difficulty_balance_ok":True,"reviewer_notes":""})
    check("no crash", True)
    check("bloom_coverage has 6 keys", len(r.bloom_coverage)==6)
    check("Remember=True",  r.bloom_coverage.get("Remember") is True)
    check("Apply=True",     r.bloom_coverage.get("Apply") is True)
    check("Evaluate=True",  r.bloom_coverage.get("Evaluate") is True)
    check("Understand=False",r.bloom_coverage.get("Understand") is False)
    check("Create=False",   r.bloom_coverage.get("Create") is False)
    check("error is None", r.error is None)
except Exception as e:
    check("no crash — EXCEPTION", False, str(e)); traceback.print_exc()

print("\n[A3] bloom_coverage = plain string")
try:
    r = _parse({"quality_score":60,"strengths":[],"weaknesses":[],"suggestions":[],
                "duplicate_question_ids":[],"bloom_coverage":"All levels covered",
                "difficulty_balance_ok":False,"reviewer_notes":""})
    check("no crash", True)
    check("bloom_coverage has 6 keys", len(r.bloom_coverage)==6)
    check("all levels False", all(v is False for v in r.bloom_coverage.values()))
    check("difficulty_balance_ok=False preserved", r.difficulty_balance_ok is False)
    check("error is None", r.error is None)
except Exception as e:
    check("no crash — EXCEPTION", False, str(e)); traceback.print_exc()

print("\n[A4] bloom_coverage dict with int 0/1 values")
try:
    r = _parse({"quality_score":80,"strengths":[],"weaknesses":[],"suggestions":[],
                "duplicate_question_ids":[],
                "bloom_coverage":{"Remember":1,"Understand":0,"Apply":1,
                                   "Analyze":0,"Evaluate":1,"Create":0},
                "difficulty_balance_ok":True,"reviewer_notes":""})
    check("no crash", True)
    check("bloom_coverage has 6 keys", len(r.bloom_coverage)==6)
    check("Remember=True  (int 1)",   r.bloom_coverage.get("Remember") is True)
    check("Understand=False (int 0)", r.bloom_coverage.get("Understand") is False)
    check("error is None", r.error is None)
except Exception as e:
    check("no crash — EXCEPTION", False, str(e)); traceback.print_exc()

print("\n[A5] bloom_coverage dict with null per-key values")
try:
    r = _parse({"quality_score":55,"strengths":[],"weaknesses":[],"suggestions":[],
                "duplicate_question_ids":[],
                "bloom_coverage":{"Remember":None,"Understand":True,"Apply":None,
                                   "Analyze":False,"Evaluate":None,"Create":True},
                "difficulty_balance_ok":True,"reviewer_notes":""})
    check("no crash", True)
    check("bloom_coverage has 6 keys", len(r.bloom_coverage)==6)
    check("Remember=False (null→False)", r.bloom_coverage.get("Remember") is False)
    check("Understand=True",             r.bloom_coverage.get("Understand") is True)
    check("Create=True",                 r.bloom_coverage.get("Create") is True)
    check("error is None", r.error is None)
except Exception as e:
    check("no crash — EXCEPTION", False, str(e)); traceback.print_exc()

print("\n[A6] bloom_coverage dict with string-boolean values; difficulty_balance_ok = 'true'")
try:
    r = _parse({"quality_score":65,"strengths":[],"weaknesses":[],"suggestions":[],
                "duplicate_question_ids":[],
                "bloom_coverage":{"Remember":"true","Understand":"false","Apply":"yes",
                                   "Analyze":"no","Evaluate":"True","Create":"False"},
                "difficulty_balance_ok":"true","reviewer_notes":"ok"})
    check("no crash", True)
    check("Remember=True (str 'true')",   r.bloom_coverage.get("Remember") is True)
    check("Understand=False (str 'false')",r.bloom_coverage.get("Understand") is False)
    check("Apply=True (str 'yes')",        r.bloom_coverage.get("Apply") is True)
    check("difficulty_balance_ok=True",    r.difficulty_balance_ok is True)
    check("reviewer_notes='ok'",           r.reviewer_notes == "ok")
    check("error is None", r.error is None)
except Exception as e:
    check("no crash — EXCEPTION", False, str(e)); traceback.print_exc()

print("\n[A7] all list fields = null")
try:
    r = _parse({"quality_score":50,"strengths":None,"weaknesses":None,
                "suggestions":None,"duplicate_question_ids":None,
                "bloom_coverage":{},"difficulty_balance_ok":True,"reviewer_notes":None})
    check("no crash", True)
    check("strengths == []",              r.strengths == [])
    check("weaknesses == []",             r.weaknesses == [])
    check("suggestions == []",            r.suggestions == [])
    check("duplicate_question_ids == []", r.duplicate_question_ids == [])
    check("reviewer_notes == ''",         r.reviewer_notes == "", repr(r.reviewer_notes))
    check("error is None", r.error is None)
except Exception as e:
    check("no crash — EXCEPTION", False, str(e)); traceback.print_exc()

print("\n[A8] list fields are wrong types (dict/str/int/bool)")
try:
    r = _parse({"quality_score":45,"strengths":{"a":1},"weaknesses":"bad",
                "suggestions":42,"duplicate_question_ids":True,
                "bloom_coverage":{},"difficulty_balance_ok":True,"reviewer_notes":""})
    check("no crash", True)
    check("strengths == [] (dict→[])",   r.strengths == [])
    check("weaknesses == [] (str→[])",   r.weaknesses == [])
    check("suggestions == [] (int→[])",  r.suggestions == [])
    check("dup_ids == [] (bool→[])",     r.duplicate_question_ids == [])
    check("error is None", r.error is None)
except Exception as e:
    check("no crash — EXCEPTION", False, str(e)); traceback.print_exc()

print("\n[A9] quality_score = string; difficulty_balance_ok = null")
try:
    r = _parse({"quality_score":"87.5","strengths":["Good"],"weaknesses":[],
                "suggestions":[],"duplicate_question_ids":[],
                "bloom_coverage":{},"difficulty_balance_ok":None,"reviewer_notes":""})
    check("no crash", True)
    check("quality_score=87.5 (str→float)", r.quality_score == 87.5, str(r.quality_score))
    check("difficulty_balance_ok=True (null→safe default)", r.difficulty_balance_ok is True)
    check("error is None", r.error is None)
except Exception as e:
    check("no crash — EXCEPTION", False, str(e)); traceback.print_exc()

print("\n[A10] quality_score = non-numeric string 'N/A'")
try:
    r = _parse({"quality_score":"N/A","strengths":[],"weaknesses":[],
                "suggestions":[],"duplicate_question_ids":[],
                "bloom_coverage":{},"difficulty_balance_ok":True,"reviewer_notes":""})
    check("no crash", True)
    check("quality_score=0.0 (unparseable→0.0)", r.quality_score == 0.0, str(r.quality_score))
    check("error is None", r.error is None)
except Exception as e:
    check("no crash — EXCEPTION", False, str(e)); traceback.print_exc()

print("\n[A11] top-level JSON is an array wrapping a valid object")
try:
    # _extract_json() strips the outer array and recovers the inner {...};
    # the function should succeed gracefully rather than raise.
    r = _parse_raw('[{"quality_score":70,"strengths":[],"weaknesses":[],'
                   '"suggestions":[],"duplicate_question_ids":[],'
                   '"bloom_coverage":{},"difficulty_balance_ok":true,'
                   '"reviewer_notes":""}]')
    check("no crash (array wrapper stripped by _extract_json)", True)
    check("returns ReviewerResult", isinstance(r, ReviewerResult))
    check("quality_score=70.0 from inner object", r.quality_score == 70.0, str(r.quality_score))
    check("error is None", r.error is None)
except Exception as e:
    check("no crash — EXCEPTION", False, str(e)); traceback.print_exc()

print("\n[A12] top-level JSON is null (bare 'null')")
try:
    r = _parse_raw("null")
    check("should have raised ValueError — did not", False, "got ReviewerResult unexpectedly")
except ValueError as e:
    check("raises ValueError for null top-level", True, str(e)[:80])
except Exception as e:
    check("wrong exception type", False, f"{type(e).__name__}: {e}")

print("\n[A13] completely empty JSON object {}")
try:
    r = _parse({})
    check("no crash", True)
    check("quality_score=0.0",             r.quality_score == 0.0)
    check("strengths == []",               r.strengths == [])
    check("bloom_coverage has 6 keys",     len(r.bloom_coverage) == 6, str(len(r.bloom_coverage)))
    check("difficulty_balance_ok=True",    r.difficulty_balance_ok is True)
    check("reviewer_notes == ''",          r.reviewer_notes == "")
    check("error is None",                 r.error is None)
except Exception as e:
    check("no crash — EXCEPTION", False, str(e)); traceback.print_exc()

print("\n[A14] bloom_coverage dict has extra unknown keys (ignored)")
try:
    r = _parse({"quality_score":70,"strengths":[],"weaknesses":[],"suggestions":[],
                "duplicate_question_ids":[],
                "bloom_coverage":{"Remember":True,"Understand":True,"Apply":False,
                                   "Analyze":True,"Evaluate":False,"Create":True,
                                   "Synthesis":True,"Knowledge":False},
                "difficulty_balance_ok":True,"reviewer_notes":""})
    check("no crash", True)
    check("bloom_coverage has exactly 6 keys", len(r.bloom_coverage)==6, str(len(r.bloom_coverage)))
    check("Synthesis not present",             "Synthesis" not in r.bloom_coverage)
    check("error is None",                     r.error is None)
except Exception as e:
    check("no crash — EXCEPTION", False, str(e)); traceback.print_exc()

print("\n[A15] reviewer_notes = integer 42")
try:
    r = _parse({"quality_score":70,"strengths":[],"weaknesses":[],"suggestions":[],
                "duplicate_question_ids":[],"bloom_coverage":{},
                "difficulty_balance_ok":True,"reviewer_notes":42})
    check("no crash", True)
    check("reviewer_notes coerced to '42'", r.reviewer_notes == "42", repr(r.reviewer_notes))
    check("error is None", r.error is None)
except Exception as e:
    check("no crash — EXCEPTION", False, str(e)); traceback.print_exc()

# ===========================================================================
# B — ReviewerAgent.review() edge cases (empty / duplicates / live Groq)
# ===========================================================================
print("\n[B1] review() — empty assessment (no LLM call)")
try:
    rr = reviewer.review(empty_assessment())
    check("returns ReviewerResult", isinstance(rr, ReviewerResult))
    check("error field set",        bool(rr.error))
    check("quality_score == 0.0",   rr.quality_score == 0.0)
except Exception as e:
    check("no crash — EXCEPTION", False, str(e)); traceback.print_exc()

print("\n[B2] review() — duplicate assessment (deterministic safety net)")
try:
    rr = reviewer.review(dup_assessment())
    check("returns ReviewerResult", isinstance(rr, ReviewerResult))
    check("Q1 in duplicate_ids",    "Q1" in rr.duplicate_question_ids, str(rr.duplicate_question_ids))
    check("Q2 in duplicate_ids",    "Q2" in rr.duplicate_question_ids)
except Exception as e:
    check("no crash — EXCEPTION", False, str(e)); traceback.print_exc()

# ===========================================================================
# C — Live Groq call end-to-end
# ===========================================================================
print("\n[C] ReviewerAgent — live Groq call (6-question assessment)")
try:
    rr = reviewer.review(full_assessment())
    check("returns ReviewerResult",        isinstance(rr, ReviewerResult))
    check("no error",                      rr.error is None, rr.error or "")
    check("quality_score in [0,100]",      0 <= rr.quality_score <= 100, str(rr.quality_score))
    check("bloom_coverage has 6 keys",     len(rr.bloom_coverage) == 6)
    check("all bloom values are bool",     all(isinstance(v,bool) for v in rr.bloom_coverage.values()))
    check("strengths is List[str]",        isinstance(rr.strengths,list) and all(isinstance(s,str) for s in rr.strengths))
    check("difficulty_balance_ok is bool", isinstance(rr.difficulty_balance_ok,bool))
    check("to_dict() round-trips",         bool(json.dumps(rr.to_dict())))
    print(f"     quality_score={rr.quality_score}")
    print(f"     strengths:  {rr.strengths[:1]}")
    print(f"     weaknesses: {rr.weaknesses[:1]}")
    print(f"     bloom_coverage: {rr.bloom_coverage}")
except Exception as e:
    check("live Groq call — EXCEPTION", False, str(e)); traceback.print_exc()

print("\n[C2] AnalyticsAgent + ReviewerAgent combined serialisation")
try:
    report: AnalyticsReport = analytics.analyse(full_assessment())
    report.reviewer_result = reviewer.review(full_assessment())
    blob = json.dumps(report.to_dict(), ensure_ascii=False)
    reparsed = json.loads(blob)
    check("combined to_dict() serialises",       bool(blob))
    check("reviewer_result present",             reparsed.get("reviewer_result") is not None)
    check("bloom_coverage in reviewer_result",   "bloom_coverage" in reparsed["reviewer_result"])
    check("quality_score numeric in reviewer",   isinstance(reparsed["reviewer_result"]["quality_score"],(int,float)))
except Exception as e:
    check("combined serialisation — EXCEPTION", False, str(e)); traceback.print_exc()

# ===========================================================================
# D — HTTP 200 on running Streamlit app
# ===========================================================================
print("\n[D] Streamlit app — HTTP 200")
try:
    status = urllib.request.urlopen("http://localhost:8000", timeout=5).getcode()
    check("HTTP 200", status == 200, str(status))
except Exception as e:
    check("HTTP 200 — FAILED", False, str(e))

# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "="*65)
passed = sum(1 for _,ok,_ in results if ok)
failed = sum(1 for _,ok,_ in results if not ok)
print(f"Results: {passed} passed, {failed} failed out of {len(results)} checks")
if failed:
    print("\nFailed checks:")
    for name, ok, detail in results:
        if not ok:
            print(f"  ❌  {name}" + (f"  [{detail}]" if detail else ""))
print("="*65 + "\n")
sys.exit(0 if failed == 0 else 1)
