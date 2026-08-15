"""
EduPilot AI Faculty Assistant
Module: app.py
Version: 5.0.0
Author: EduPilot Team
Purpose: Streamlit application entry point — full faculty dashboard.

        v5.0 changes (Task #43):
          - Pipeline progress replaced with per-stage st.status checklist.
          - "View Assessment" page refactored into 4 tabs:
              Questions | Analytics | AI Review | Downloads
          - Question cards use st.container(border=True) + st.badge.
          - Analytics tab: headline metrics + Plotly charts
            (Bloom / CO / difficulty / marks).
          - AI Review tab: quality score gauge, strengths / weaknesses /
            suggestions, Bloom coverage grid.
          - Downloads tab: real st.download_button for MD / Word / PDF from
            SS_EXPORT_BYTES; history-loaded runs regenerate bytes via
            DownloadEngine.
          - "📜 History" page: reads runs/index.json, re-opens a full run via
            inline reconstruction helper with enum coercion.
          - Fixed src.excerpt loop-variable bug (line ~649 in v4.3).
          - Fixed st.success-before-rerun flash via _pending_success state key.
          - Minimal CSS injected for card polish; respects Streamlit theme.
"""

from __future__ import annotations

import faulthandler
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import plotly.express as px
import streamlit as st

from config import config, ConfigurationError
from logging_utils import get_logger

import auth

# Enable faulthandler so C-level crashes (segfaults, stack overflows) produce
# a traceback to stderr instead of dying silently.
faulthandler.enable()

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EduPilot – AI Faculty Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**EduPilot v5.0** — AI-powered OBE Assessment Generator\n\n"
            "Reduces assessment preparation from hours to seconds while "
            "ensuring Bloom's Taxonomy alignment and Course Outcome mapping."
        )
    },
)

logger.info("EduPilot app.py v5.0 loaded")


# ---------------------------------------------------------------------------
# Minimal CSS — theme-safe polish for cards and badges
# ---------------------------------------------------------------------------

def _inject_css() -> None:
    """Inject minimal CSS that works on both Streamlit light and dark themes."""
    st.markdown(
        """
        <style>
        /* Question card header strip */
        .q-card-header {
            font-size: 0.82rem;
            opacity: 0.75;
            margin-bottom: 0.3rem;
        }
        /* Reviewer score ring — centred large text */
        .score-ring {
            text-align: center;
            font-size: 2.8rem;
            font-weight: 700;
            line-height: 1;
            padding: 0.5rem 0;
        }
        .score-label {
            text-align: center;
            font-size: 0.85rem;
            opacity: 0.65;
            margin-top: -0.4rem;
        }
        /* Muted section divider label */
        .section-eyebrow {
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.5;
            margin-bottom: 0.25rem;
        }
        /* ── Visual polish (light theme) ─────────────────────────────── */
        /* Softer cards with subtle elevation */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 14px;
            box-shadow: 0 1px 3px rgba(30, 34, 53, 0.06);
        }
        /* Rounded, confident buttons */
        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
            border-radius: 10px;
            font-weight: 600;
        }
        /* Sidebar: slightly deeper background + divider */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #EEF0FA 0%, #F7F8FC 100%);
            border-right: 1px solid rgba(79, 70, 229, 0.10);
        }
        /* Metric cards */
        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid rgba(30, 34, 53, 0.07);
            border-radius: 12px;
            padding: 0.6rem 0.9rem;
        }
        /* Tabs: pill-style highlight */
        button[data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            font-weight: 600;
        }
        /* Headings a touch tighter */
        h1, h2, h3 { letter-spacing: -0.01em; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session state keys
# ---------------------------------------------------------------------------

SS_ASSESSMENT       = "generated_assessment"
SS_PLAN             = "last_plan"
SS_RAG_SOURCES      = "last_rag_sources"
SS_ANALYTICS        = "last_analytics"
SS_REVIEWER         = "last_reviewer"
SS_EXPORT_BYTES     = "last_export_bytes"
SS_PIPELINE_RESULT  = "last_pipeline_result"
SS_PENDING_SUCCESS  = "_pending_success"   # flash-fix: message shown on View page
SS_HISTORY_RUN_ID   = "_history_run_id"    # run_id of a history-reopened run


# ---------------------------------------------------------------------------
# Bloom / difficulty badge colours for st.badge
# ---------------------------------------------------------------------------

_BLOOM_BADGE: Dict[str, str] = {
    "Remember":   "blue",
    "Understand": "green",
    "Apply":      "orange",
    "Analyze":    "violet",
    "Evaluate":   "red",
    "Create":     "rainbow",
}

_DIFFICULTY_BADGE: Dict[str, str] = {
    "Easy":   "green",
    "Medium": "orange",
    "Hard":   "red",
}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> str:
    """Render the EduPilot navigation sidebar.

    Returns:
        str: The currently selected page label (may be overridden by
        programmatic navigation stored in ``_nav`` session state key).
    """
    with st.sidebar:
        st.image(
            "https://img.icons8.com/color/96/000000/graduation-cap.png",
            width=72,
        )
        st.title("EduPilot")
        st.caption("AI Faculty Assistant · v5.0")
        st.divider()

        st.markdown(f"👤 **{auth.current_display_name()}**")
        auth.render_logout(location="sidebar")
        st.divider()

        st.markdown("**Navigation**")
        page = st.radio(
            "Go to",
            [
                "🏠 Home",
                "📚 Knowledge Base",
                "✍️ Generate Assessment",
                "📄 View Assessment",
                "📜 History",
                "⚙️ My Settings",
                "🔐 Change Password",
            ],
            label_visibility="collapsed",
            key="nav_radio",
        )

        st.divider()
        st.markdown("**Status**")

        groq_configured = bool(config.groq_api_key)
        if groq_configured:
            st.success("Groq API ✓ Connected", icon="✅")
        else:
            st.warning("Groq API key not set", icon="⚠️")

        st.caption(f"Model: `{config.groq_model_name}`")
        st.caption(f"Embeddings: `{config.embedding_model_name}`")

        vs_path = config.vectorstore_path / "index.faiss"
        if vs_path.exists():
            st.success("Vector store ✓ Ready", icon="📚")
        else:
            st.info("Vector store not built yet", icon="ℹ️")

        # History quick-stat
        index_path = auth.user_runs_dir() / "index.json"
        if index_path.exists():
            try:
                idx = json.loads(index_path.read_text(encoding="utf-8"))
                run_count = len(idx) if isinstance(idx, list) else 0
            except Exception:
                run_count = 0
            st.caption(f"Past runs: **{run_count}**")

        st.divider()
        st.caption("© 2025 EduPilot Team")

    return page


# ---------------------------------------------------------------------------
# Home / Landing page
# ---------------------------------------------------------------------------

def render_landing() -> None:
    """Render the EduPilot landing / home page."""
    col_icon, col_title = st.columns([1, 9])
    with col_icon:
        st.markdown("# 🎓")
    with col_title:
        st.markdown("# EduPilot")
        st.markdown(
            "#### AI Faculty Assistant for Outcome-Based Education Assessment"
        )

    st.divider()

    st.markdown(
        """
        **EduPilot** is an agentic AI platform that assists faculty members
        in creating OBE-aligned assessments in seconds — not hours.

        It combines **Retrieval-Augmented Generation (RAG)** over your own
        course materials with **Groq's LLM** to produce Bloom-aligned,
        CO-mapped questions complete with answer keys, analytics, and
        professional exports.
        """
    )

    col_btn, _ = st.columns([2, 8])
    with col_btn:
        if st.button(
            "✍️ Generate Assessment →",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["_nav"] = "✍️ Generate Assessment"
            st.rerun()

    st.divider()

    st.subheader("Platform Capabilities")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            **📚 RAG Knowledge Base**
            Upload course PDFs, lecture notes, syllabus, and question banks.
            EduPilot indexes them with FAISS for semantic retrieval.
            """
        )
    with c2:
        st.markdown(
            """
            **🤖 Agentic Generation**
            Planning → Assessment agents orchestrate end-to-end question
            generation via Groq LLM with full Bloom/CO mapping.
            """
        )
    with c3:
        st.markdown(
            """
            **📊 Analytics & Exports**
            Bloom distribution, CO coverage tables, AI quality review, and
            professional Markdown / Word / PDF downloads.
            """
        )

    st.divider()
    st.subheader("Supported Assessment Types")
    types = [
        ("📝", "Internal Assessment"),
        ("❓", "Quiz"),
        ("📋", "Assignment"),
        ("📖", "Semester Examination"),
        ("🎤", "Viva"),
    ]
    cols = st.columns(len(types))
    for col, (icon, name) in zip(cols, types):
        with col:
            st.metric(label=icon, value=name)

    st.divider()
    with st.expander("🏗️ Agentic Pipeline Architecture", expanded=False):
        st.code(
            """
Faculty Requirement
      ↓
Planning Agent       — parses requirements → structured plan
      ↓
RAG Module           — retrieves relevant knowledge chunks  (non-fatal)
      ↓
Assessment Agent     — Groq LLM generates questions + answer keys
      ↓
Analytics Agent      — computes CO coverage, marks distribution
      ↓
Reviewer Agent       — AI reviews AI (quality, Bloom, duplicates)
      ↓
Download Agent       — renders Markdown / Word / PDF
      ↓
Faculty Dashboard    — Questions | Analytics | AI Review | Downloads
            """,
            language="text",
        )

    with st.expander("⚙️ Configuration", expanded=False):
        st.table(
            {
                "Setting": [
                    "Groq Model",
                    "Embedding Model",
                    "Vectorstore Path",
                    "Knowledge Dir",
                    "Log Level",
                ],
                "Value": [
                    config.groq_model_name,
                    config.embedding_model_name,
                    str(config.vectorstore_path),
                    str(config.knowledge_dir),
                    config.log_level,
                ],
            }
        )


# ---------------------------------------------------------------------------
# Generation form
# ---------------------------------------------------------------------------
# Knowledge Base page
# ---------------------------------------------------------------------------

_ALLOWED_UPLOAD_TYPES = ["pdf", "txt", "docx"]


def _sanitize_filename(name: str) -> str:
    """Return a filesystem-safe version of an uploaded filename."""
    import re
    base = os.path.basename(name)
    return re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip() or "document"


def _rebuild_knowledge_index() -> None:
    """Rebuild the FAISS index from the knowledge directory (blocking)."""
    from rag import RAGModule

    with st.spinner("Indexing documents… (first run downloads the embedding model, ~1 min)"):
        rag = RAGModule()
        count = rag.ingest(force_rebuild=True)
    if count > 0:
        st.success(f"Knowledge base indexed: {count} chunks ready for retrieval.")
    else:
        st.info("Knowledge base is empty — no documents indexed.")


def render_knowledge_page() -> None:
    """Render the Knowledge Base management page (upload / list / delete)."""
    st.title("📚 Knowledge Base")
    st.markdown(
        "Upload your course materials (**PDF, TXT, DOCX**) — lecture notes, "
        "syllabus, question banks. EduPilot indexes them for semantic "
        "retrieval, and **generated assessments are grounded strictly in "
        "these documents**."
    )

    knowledge_dir = config.knowledge_dir
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    # ---- Upload ----------------------------------------------------------
    uploaded = st.file_uploader(
        "Upload documents",
        type=_ALLOWED_UPLOAD_TYPES,
        accept_multiple_files=True,
        help="PDF, TXT, or DOCX files. Uploading triggers automatic re-indexing.",
    )
    if uploaded and st.button("💾 Save & Index Uploaded Files", type="primary"):
        saved, replaced = [], []
        for uf in uploaded:
            safe_name = _sanitize_filename(uf.name)
            dest = knowledge_dir / safe_name
            (replaced if dest.exists() else saved).append(safe_name)
            dest.write_bytes(uf.getbuffer())
        msg = f"Saved {len(saved) + len(replaced)} file(s)"
        if replaced:
            msg += f" (replaced existing: {', '.join(replaced)})"
        st.success(msg)
        try:
            _rebuild_knowledge_index()
        except Exception as exc:  # surface loudly, never silently
            st.error(f"Indexing failed: {exc}")
        else:
            st.rerun()  # only rerun on success so errors stay visible

    st.divider()

    # ---- Current documents ----------------------------------------------
    st.subheader("Current documents")
    docs = sorted(
        p for p in knowledge_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".pdf", ".txt", ".docx")
    )
    if not docs:
        st.info(
            "No documents uploaded yet. Assessments generated now will fall "
            "back to general academic knowledge — upload your course "
            "materials above to ground them in your own content."
        )
    else:
        for doc in docs:
            col_name, col_size, col_del = st.columns([6, 2, 2])
            with col_name:
                st.markdown(f"**{doc.name}**")
            with col_size:
                st.caption(f"{doc.stat().st_size / 1024:.1f} KB")
            with col_del:
                if st.button("🗑️ Delete", key=f"kb_delete_{doc.name}"):
                    doc.unlink()
                    try:
                        _rebuild_knowledge_index()
                    except Exception as exc:
                        st.error(f"Re-indexing failed: {exc}")
                    else:
                        st.rerun()  # only rerun on success

    st.divider()

    # ---- Index status & manual rebuild ------------------------------------
    col_status, col_rebuild = st.columns([3, 1])
    with col_status:
        index_file = config.vectorstore_path / "index.faiss"
        if index_file.exists():
            st.success("Vector index is built and ready.", icon="✅")
        elif docs:
            st.warning(
                "Documents present but index not built — click Rebuild Index.",
                icon="⚠️",
            )
        else:
            st.info("No index yet — upload documents to create one.", icon="ℹ️")
    with col_rebuild:
        if st.button("🔄 Rebuild Index"):
            try:
                _rebuild_knowledge_index()
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")


# ---------------------------------------------------------------------------

BLOOM_OPTIONS = [
    "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"
]
CO_OPTIONS = [f"CO{i}" for i in range(1, 7)]
ASSESSMENT_TYPES = [
    "Internal Assessment",
    "Quiz",
    "Assignment",
    "Semester Examination",
    "Viva",
    "Role Play",
    "Question Bank",
    "Consultancy Case Study",
]
DIFFICULTY_OPTIONS = ["Mixed", "Easy", "Medium", "Hard"]


def render_generation_form() -> Optional[dict]:
    """Render the assessment requirement form.

    Returns:
        dict | None: Structured plan dict on valid submission, else None.
    """
    from models import (
        AssessmentType,
        VTU_MARKS_PER_FULL_QUESTION,
        VTU_MAX_MODULES,
        VTU_MAX_TOTAL_MARKS,
    )
    from agent import parse_vtu_marks_blueprint

    st.subheader("📋 Faculty Requirement")

    # Load saved department default before the form widget is created so the
    # value is available as the widget's initial state.
    saved_department = auth.get_user_setting("department")
    saved_semester = auth.get_user_setting("semester")
    # Faculty name: saved default, falling back to the account display name.
    saved_faculty = auth.get_user_setting("faculty_name") or auth.current_display_name()

    # "Edit & Regenerate": a plan stashed by the View page pre-fills the form.
    # Read (not pop) so the prefill survives intermediate reruns; it is
    # cleared after a valid submission below.
    edit_plan = st.session_state.get("_edit_plan")
    if not isinstance(edit_plan, dict):
        edit_plan = None
    if edit_plan:
        st.info(
            "✏️ Editing the previous assessment plan — adjust anything below "
            "and regenerate.",
            icon="✏️",
        )

    def _ep(key: str, default: Any = "") -> Any:
        """Value from the edit plan, else the given default."""
        return edit_plan.get(key, default) if edit_plan else default

    def _ep_list(key: str, default: List[str]) -> List[str]:
        """Plan value (comma string or list) → list of stripped items."""
        if not edit_plan:
            return default
        val = edit_plan.get(key, "")
        if isinstance(val, (list, tuple)):
            raw = [str(p).strip() for p in val]
        else:
            raw = [p.strip() for p in str(val).split(",")]
        return [p for p in raw if p] or default

    def _ep_int(key: str, default: int, lo: int, hi: int) -> int:
        """Plan value → int clamped to [lo, hi]; falls back on bad input."""
        try:
            num = int(float(_ep(key, default)))
        except (TypeError, ValueError):
            num = default
        return min(max(num, lo), hi)

    _type_default = str(_ep("assessment_type", ASSESSMENT_TYPES[0]))
    _type_index = (
        ASSESSMENT_TYPES.index(_type_default)
        if _type_default in ASSESSMENT_TYPES else 0
    )
    _difficulty_default = str(_ep("difficulty", "Mixed"))
    if _difficulty_default not in DIFFICULTY_OPTIONS:
        _difficulty_default = "Mixed"
    _test_date_default = None
    if edit_plan and edit_plan.get("test_date"):
        try:
            _test_date_default = datetime.strptime(
                str(edit_plan["test_date"]), "%d %B %Y"
            ).date()
        except ValueError:
            _test_date_default = None

    with st.form("generation_form"):
        col_left, col_right = st.columns(2)

        with col_left:
            assessment_type = st.selectbox(
                "Assessment Type *", ASSESSMENT_TYPES,
                index=_type_index,
                help="Select the OBE assessment category.",
            )
            course_name = st.text_input(
                "Course Name *",
                value=str(_ep("course_name")),
                placeholder="e.g. Data Structures and Algorithms",
            )
            course_code = st.text_input(
                "Course Code",
                value=str(_ep("course_code")),
                placeholder="e.g. CS3001",
            )
            department = st.text_input(
                "Department *",
                value=str(_ep("department", saved_department)),
                placeholder="e.g. Computer Science & Engineering",
                help="Shown on the assessment and all downloaded documents.",
            )
            col_sem, col_fac = st.columns(2)
            with col_sem:
                semester = st.text_input(
                    "Semester",
                    value=str(_ep("semester", saved_semester)),
                    placeholder="e.g. Semester 5, 2026-27",
                    help="Optional — shown on downloaded documents.",
                )
            with col_fac:
                faculty_name = st.text_input(
                    "Faculty Name",
                    value=str(_ep("faculty_name", saved_faculty)),
                    placeholder="e.g. Dr. A. Sharma",
                    help="Optional — pre-filled from your account; shown on "
                         "downloaded documents.",
                )
            topics = st.text_area(
                "Topics to Cover *",
                value=str(_ep("topics")),
                placeholder=(
                    "e.g. Binary Search Trees, AVL Trees, Graph Traversal"
                ),
                height=80,
            )

        with col_right:
            bloom_targets = st.multiselect(
                "Target Bloom Levels *", BLOOM_OPTIONS,
                default=[
                    b for b in _ep_list(
                        "bloom_targets", ["Remember", "Understand", "Apply"]
                    ) if b in BLOOM_OPTIONS
                ],
            )
            co_mapping = st.multiselect(
                "Course Outcomes (COs) *", CO_OPTIONS,
                default=[
                    c for c in _ep_list("co_mapping", ["CO1", "CO2"])
                    if c in CO_OPTIONS
                ],
            )
            col_qc, col_mpq = st.columns(2)
            with col_qc:
                question_count = st.number_input(
                    "No. of Questions *",
                    min_value=1, max_value=30,
                    value=_ep_int("question_count", 5, 1, 30),
                    step=1,
                )
            with col_mpq:
                marks_per_question = st.number_input(
                    "Marks / Question *",
                    min_value=1, max_value=100,
                    value=_ep_int("marks_per_question", 5, 1, 100),
                    step=1,
                )
            difficulty = st.select_slider(
                "Difficulty", options=DIFFICULTY_OPTIONS,
                value=_difficulty_default,
            )
            test_date = st.date_input(
                "Test Date (optional)",
                value=_test_date_default,
                format="DD/MM/YYYY",
                help="Date the assessment will be conducted — shown on all "
                     "exported documents.",
            )

        extra_instructions = st.text_area(
            "Additional Instructions (optional)",
            value=str(_ep("extra_instructions")),
            placeholder=(
                "e.g. Focus on practical applications; "
                "avoid pure definition questions."
            ),
            height=60,
        )

        vtu_marks_blueprint = st.text_area(
            "Custom Sub-Question Marks (optional — Semester Examination only)",
            value=str(_ep("vtu_marks_blueprint")),
            placeholder=(
                "Only used when Assessment Type is \"Semester Examination\". "
                "One line per topic, giving the exact marks for each "
                "lettered sub-part of both OR-alternative questions:\n"
                "Autoencoders: 5,5,10 | 5,5,10\n"
                "GANs: 5,5,10 | 10,10\n"
                "Topics left out fall back to automatic equal-marks "
                "distribution — leave this entirely blank to keep that "
                "default behaviour for every topic."
            ),
            height=90,
            help=(
                "Format: \"<topic>: <sub-part marks for Q-A, comma-"
                "separated> | <sub-part marks for Q-B>\" — one line per "
                "topic. The AI will generate content scoped to match each "
                "marks value (e.g. a 10-mark sub-part gets a more detailed "
                "answer than a 3-mark one). Topic names must match the "
                "Topics field above."
            ),
        )

        submitted = st.form_submit_button(
            "🚀 Generate Assessment",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return None

    errors: List[str] = []
    if not course_name.strip():
        errors.append("Course Name is required.")
    if not topics.strip():
        errors.append("Topics are required.")
    if not department.strip():
        errors.append("Department is required.")
    if not bloom_targets:
        errors.append("Select at least one Bloom level.")
    if not co_mapping:
        errors.append("Select at least one Course Outcome.")

    if assessment_type == AssessmentType.SEMESTER_EXAM.value:
        topic_list = [t.strip() for t in topics.split(",") if t.strip()]
        if len(topic_list) > VTU_MAX_MODULES:
            errors.append(
                f"Semester Examination supports at most {VTU_MAX_MODULES} "
                f"Modules ({VTU_MARKS_PER_FULL_QUESTION} marks each = "
                f"{VTU_MAX_TOTAL_MARKS} marks total). You entered "
                f"{len(topic_list)} topics — please reduce the Topics "
                f"field to {VTU_MAX_MODULES} or fewer."
            )
        if vtu_marks_blueprint.strip():
            parsed_bp = parse_vtu_marks_blueprint(vtu_marks_blueprint)
            for line_topic, (marks_a, marks_b) in parsed_bp.items():
                if sum(marks_a) != VTU_MARKS_PER_FULL_QUESTION:
                    errors.append(
                        f"Custom marks for \"{line_topic}\" (Q-A side) sum "
                        f"to {sum(marks_a)}, not {VTU_MARKS_PER_FULL_QUESTION}. "
                        f"Each full question must total exactly "
                        f"{VTU_MARKS_PER_FULL_QUESTION} marks."
                    )
                if sum(marks_b) != VTU_MARKS_PER_FULL_QUESTION:
                    errors.append(
                        f"Custom marks for \"{line_topic}\" (Q-B side) sum "
                        f"to {sum(marks_b)}, not {VTU_MARKS_PER_FULL_QUESTION}. "
                        f"Each full question must total exactly "
                        f"{VTU_MARKS_PER_FULL_QUESTION} marks."
                    )

    if errors:
        for e in errors:
            st.error(e)
        return None

    # Prefill served its purpose — clear it so a later fresh visit to the
    # form starts from the user's saved defaults again.
    st.session_state.pop("_edit_plan", None)

    # Persist defaults for future form renders.
    auth.save_user_setting("department", department.strip())
    if semester.strip():
        auth.save_user_setting("semester", semester.strip())
    if faculty_name.strip():
        auth.save_user_setting("faculty_name", faculty_name.strip())

    return {
        "assessment_type": assessment_type,
        "course_name": course_name.strip(),
        "course_code": course_code.strip(),
        "topics": topics.strip(),
        "bloom_targets": ", ".join(bloom_targets),
        "co_mapping": ", ".join(co_mapping),
        "question_count": int(question_count),
        "marks_per_question": int(marks_per_question),
        "difficulty": difficulty,
        "extra_instructions": extra_instructions.strip(),
        "vtu_marks_blueprint": vtu_marks_blueprint.strip(),
        "duration_minutes": 0,
        "department": department.strip(),
        "semester": semester.strip(),
        "faculty_name": faculty_name.strip(),
        "test_date": test_date.strftime("%d %B %Y") if test_date else "",
    }


# ---------------------------------------------------------------------------
# Generation pipeline with per-stage st.status display
# ---------------------------------------------------------------------------

_STAGE_LABELS: Dict[str, str] = {
    "planning":    "📋 Planning assessment structure",
    "rag":         "📚 Retrieving knowledge chunks",
    "generation":  "🤖 Generating questions via Groq LLM",
    "analytics":   "📊 Computing analytics",
    "reviewer":    "🔍 Running AI quality reviewer",
    "export_prep": "📄 Preparing document exports",
}

_STAGE_ORDER: List[str] = [
    "planning", "rag", "generation", "analytics", "reviewer", "export_prep"
]


def run_generation_pipeline(plan: dict) -> None:
    """Execute the multi-agent pipeline; display per-stage st.status checklist.

    On success stores all results in session state and triggers navigation to
    the View Assessment page via a pending-success banner (avoids the
    st.success-before-rerun flash).

    Args:
        plan: Structured plan dict from the generation form.
    """
    # Pre-allocate one st.empty() slot per stage so we can update in place.
    with st.status(
        "🚀 Running generation pipeline…", expanded=True
    ) as pipe_status:

        stage_slots: Dict[str, Any] = {
            s: st.empty() for s in _STAGE_ORDER
        }
        # Initialise all stages as pending
        for s, slot in stage_slots.items():
            slot.markdown(f"⬜&nbsp; {_STAGE_LABELS[s]}")

        # Closure: update slot to "running" on first hit for a stage,
        # stage order means a new stage arriving implies the previous is done.
        _active: List[str] = []

        def _cb(stage: str, pct: int) -> None:
            """Progress callback fired by PipelineOrchestrator."""
            if stage not in _active:
                # Previous stage (if any) transitions to ⏳ → done marker
                # is set after pipeline completes from result.stages.
                stage_slots[stage].markdown(
                    f"⏳&nbsp; **{_STAGE_LABELS.get(stage, stage)}**"
                )
                _active.append(stage)

        # ── Batch progress updater: shows "Generating questions 9–16 of 25…"
        # inside the generation stage slot without flickering.
        _gen_slot = stage_slots["generation"]

        def _batch_status_cb(msg: str) -> None:
            """Update the generation stage slot with per-batch progress text."""
            try:
                _gen_slot.markdown(
                    f"⏳&nbsp; **{_STAGE_LABELS['generation']}**\n\n"
                    f"<small>{msg}</small>",
                    unsafe_allow_html=True,
                )
            except Exception:  # noqa: BLE001
                pass  # never let a UI callback kill the pipeline

        # ── Run the pipeline ────────────────────────────────────────────────
        try:
            from orchestrator import PipelineOrchestrator  # lazy import
            result = PipelineOrchestrator(runs_dir=auth.user_runs_dir()).run(
                plan,
                progress_callback=_cb,
                batch_status_callback=_batch_status_cb,
            )
        except ConfigurationError as exc:
            pipe_status.update(label="❌ Configuration error", state="error")
            st.error(
                f"🔑 **Configuration error:** {exc}\n\n"
                "Add your `GROQ_API_KEY` as a Replit secret and restart.",
                icon="🔑",
            )
            logger.error("ConfigurationError during pipeline: %s", exc)
            return
        except Exception as exc:
            pipe_status.update(label="❌ Pipeline error", state="error")
            st.error(f"❌ **Pipeline error:** {exc}", icon="❌")
            logger.exception("Unexpected pipeline error: %s", exc)
            return

        # ── Update stage slots from authoritative result.stages ────────────
        for stg in result.stages:
            label = _STAGE_LABELS.get(stg.stage, stg.stage)
            if stg.status == "ok":
                stage_slots[stg.stage].markdown(f"✅&nbsp; {label}")
            elif stg.status == "skipped" or stg.warning:
                stage_slots[stg.stage].markdown(
                    f"⚠️&nbsp; {label}"
                    + (f" — {stg.warning}" if stg.warning else "")
                )
            else:
                stage_slots[stg.stage].markdown(
                    f"❌&nbsp; {label}"
                    + (f": {stg.error}" if stg.error else "")
                )
        # Mark any stages not reached as skipped
        reached = {stg.stage for stg in result.stages}
        for s in _STAGE_ORDER:
            if s not in reached:
                stage_slots[s].markdown(f"⏭️&nbsp; {_STAGE_LABELS[s]} (skipped)")

        # ── Hard failure ────────────────────────────────────────────────────
        if not result.success:
            pipe_status.update(label="❌ Pipeline failed", state="error")
            st.error(
                f"❌ **Pipeline failed:** {result.fatal_error or 'Unknown error.'}",
                icon="❌",
            )
            logger.error(
                "Pipeline run_id=%s failed: %s", result.run_id, result.fatal_error
            )
            return

        pipe_status.update(
            label="✅ Assessment generated!", state="complete", expanded=False
        )

    # ── Persist results to session state ────────────────────────────────────
    st.session_state[SS_ASSESSMENT]      = result.assessment
    st.session_state[SS_PLAN]            = (
        result.plan.to_dict() if result.plan else plan
    )
    st.session_state[SS_RAG_SOURCES]     = result.rag_sources
    st.session_state[SS_ANALYTICS]       = result.analytics
    st.session_state[SS_REVIEWER]        = result.reviewer
    st.session_state[SS_EXPORT_BYTES]    = result.export_bytes
    st.session_state[SS_PIPELINE_RESULT] = result
    # Clear any stale history-run marker and old MCQ radio selections
    st.session_state.pop(SS_HISTORY_RUN_ID, None)
    for k in [k for k in st.session_state if str(k).startswith("q_options_")]:
        st.session_state.pop(k, None)

    logger.info(
        "Pipeline complete: run_id=%s questions=%d marks=%d "
        "quality=%.1f exports=%s",
        result.run_id,
        result.assessment.question_count,
        result.assessment.total_marks,
        result.analytics.quality_score if result.analytics else 0.0,
        list(result.export_bytes.keys()),
    )

    # Flash-fix: store success message to be shown on the View page
    st.session_state[SS_PENDING_SUCCESS] = (
        f"✅ Assessment generated: **{result.assessment.question_count} "
        f"questions**, **{result.assessment.total_marks} total marks**."
    )
    st.session_state["_nav"] = "📄 View Assessment"
    st.rerun()


# ---------------------------------------------------------------------------
# Generate Assessment page
# ---------------------------------------------------------------------------

def render_generate_page() -> None:
    """Render the Assessment Generation page with form and pipeline trigger."""
    st.title("✍️ Generate Assessment")
    st.markdown(
        "Fill in the faculty requirement below. EduPilot will retrieve "
        "relevant knowledge from your course materials and generate a "
        "complete, Bloom-aligned assessment via Groq LLM."
    )

    if not config.groq_api_key:
        st.error(
            "**GROQ_API_KEY is not configured.**\n\n"
            "Add it as a Replit secret named `GROQ_API_KEY` and restart.",
            icon="🔑",
        )

    st.divider()
    plan = render_generation_form()

    if plan is not None:
        run_generation_pipeline(plan)


# ---------------------------------------------------------------------------
# View Assessment — tab renderers
# ---------------------------------------------------------------------------

def _render_questions_tab(assessment: Any, rag_sources: list) -> None:
    """Render the Questions tab with styled cards and badges.

    Args:
        assessment: Current ``Assessment`` object from session state.
        rag_sources: List of ``SourceAttribution`` objects (may be empty).
    """
    meta = assessment.metadata

    # Scope radio-widget keys to the specific run so selections never bleed
    # between a freshly generated assessment and reopened history runs.
    _result = st.session_state.get(SS_PIPELINE_RESULT)
    _run_scope = (
        st.session_state.get(SS_HISTORY_RUN_ID)
        or (getattr(_result, "run_id", "") if _result else "")
        or "current"
    )

    # Assessment metadata header
    with st.container(border=True):
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(f"**📋 Title:** {meta.title}")
            st.markdown(
                f"**📘 Course:** {meta.course_name}"
                + (f" `{meta.course_code}`" if meta.course_code else "")
            )
            st.markdown(f"**🏫 Department:** {meta.department or '—'}")
            st.markdown(f"**📅 Semester:** {meta.semester or '—'}")
            st.markdown(f"**🗓️ Test Date:** {meta.test_date or '—'}")
        with mc2:
            st.markdown(f"**👤 Faculty:** {meta.faculty_name or '—'}")
            st.markdown(
                f"**⏱️ Duration:** "
                + (
                    f"{meta.duration_minutes} min"
                    if meta.duration_minutes
                    else "Flexible"
                )
            )
            st.markdown(f"**🏆 Total Marks:** {assessment.total_marks}")
        if meta.instructions:
            st.info(f"📌 **Instructions:** {meta.instructions}")
        if assessment.generation_notes:
            st.caption(f"🤖 Generation notes: {assessment.generation_notes}")

    st.divider()
    st.markdown(
        f"<div class='section-eyebrow'>Questions — "
        f"{assessment.question_count} total</div>",
        unsafe_allow_html=True,
    )

    for q in assessment.questions:
        with st.container(border=True):
            # Question header row
            h_col, marks_col = st.columns([9, 1])
            with h_col:
                st.markdown(
                    f"<div class='q-card-header'>{q.question_id} · "
                    f"{q.question_type}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**{q.question_text}**")
                if getattr(q, "options", None):
                    st.radio(
                        "Options",
                        options=[
                            f"{chr(65 + i)}.  {opt}"
                            for i, opt in enumerate(q.options)
                        ],
                        index=None,
                        key=f"q_options_{_run_scope}_{q.question_id}",
                        label_visibility="collapsed",
                    )
            with marks_col:
                st.metric(
                    label="Marks",
                    value=q.marks,
                    label_visibility="visible",
                )

            # Badge row
            badge_cols = st.columns([3, 2, 2, 3])
            with badge_cols[0]:
                bloom_color = _BLOOM_BADGE.get(
                    q.bloom_level.value
                    if hasattr(q.bloom_level, "value")
                    else str(q.bloom_level),
                    "blue",
                )
                bloom_label = (
                    q.bloom_level.value
                    if hasattr(q.bloom_level, "value")
                    else str(q.bloom_level)
                )
                st.markdown(f":{bloom_color}[**🧠 {bloom_label}**]")
            with badge_cols[1]:
                diff_val = (
                    q.difficulty.value
                    if hasattr(q.difficulty, "value")
                    else str(q.difficulty)
                )
                diff_color = _DIFFICULTY_BADGE.get(diff_val, "gray")
                st.markdown(f":{diff_color}[**⚡ {diff_val}**]")
            with badge_cols[2]:
                co_str = ", ".join(q.co_mapping) if q.co_mapping else "—"
                st.markdown(f":violet[**📌 {co_str}**]")
            with badge_cols[3]:
                if q.notes:
                    st.caption(f"📎 {q.notes}")

            # Answer key
            with st.expander("📖 Answer Key / Marking Scheme", expanded=False):
                st.markdown(
                    q.answer_key if q.answer_key else "*No answer key provided.*"
                )

            # Per-question sources — bug fix: iterate q.sources not outer rag_sources
            if q.sources:
                for src in q.sources:
                    src_label = f"📚 **{src.document_name}**"
                    if src.page_number:
                        src_label += f" (p. {src.page_number})"
                    if src.relevance_score:
                        src_label += f" — relevance {src.relevance_score:.2f}"
                    st.caption(src_label)

    # Global RAG sources section
    if rag_sources:
        st.divider()
        st.markdown(
            "<div class='section-eyebrow'>Knowledge Sources Used</div>",
            unsafe_allow_html=True,
        )
        seen: set = set()
        unique_sources = []
        for src in rag_sources:
            if src.document_name not in seen:
                seen.add(src.document_name)
                unique_sources.append(src)

        for src in unique_sources:
            src_line = f"- **{src.document_name}**"
            if src.page_number:
                src_line += f" (p. {src.page_number})"
            if src.relevance_score:
                src_line += f" — relevance: {src.relevance_score:.2f}"
            st.markdown(src_line)

        # Excerpts expander — fixed bug: iterate unique_sources, not leaked var
        has_excerpts = any(s.excerpt for s in rag_sources)
        if has_excerpts:
            with st.expander("View retrieved excerpts"):
                for src in rag_sources:
                    if src.excerpt:
                        st.markdown(f"**{src.document_name}:**")
                        st.caption(
                            src.excerpt[:300]
                            + ("…" if len(src.excerpt) > 300 else "")
                        )
    else:
        st.caption(
            "ℹ️ No RAG knowledge sources were retrieved for this assessment "
            "(vector store empty or embeddings unavailable — normal in this env)."
        )


def _render_analytics_tab(analytics: Optional[Any]) -> None:
    """Render the Analytics tab with headline metrics and Plotly charts.

    Args:
        analytics: ``AnalyticsReport`` from session state, or None.
    """
    if analytics is None:
        st.warning(
            "Analytics were not computed for this run "
            "(stage may have failed or been skipped).",
            icon="⚠️",
        )
        return

    # ── Headline metrics ─────────────────────────────────────────────────────
    st.markdown(
        "<div class='section-eyebrow'>Headline Metrics</div>",
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📝 Questions", analytics.question_count)
    m2.metric(
        "📚 Sources Used",
        len(analytics.knowledge_sources_used),
        help="Distinct knowledge-base documents referenced.",
    )
    m3.metric(
        "⏱️ Time Saved",
        f"{analytics.estimated_time_saved_minutes} min",
        help="Estimated faculty prep time saved.",
    )

    # Quality score: prefer reviewer LLM score if available
    reviewer = getattr(analytics, "reviewer_result", None)
    if reviewer and not reviewer.error and reviewer.quality_score:
        qs_value = f"{reviewer.quality_score:.1f}"
        qs_help = "LLM reviewer quality score (0–100)"
    else:
        qs_value = f"{analytics.quality_score:.1f}"
        qs_help = "Deterministic analytics quality score (0–100)"
    m4.metric("🏅 Quality Score", qs_value, help=qs_help)

    st.divider()

    # ── Charts row 1: Bloom + Difficulty ─────────────────────────────────────
    st.markdown(
        "<div class='section-eyebrow'>Distributions</div>",
        unsafe_allow_html=True,
    )
    ch1, ch2 = st.columns(2)

    with ch1:
        bloom_counts = analytics.bloom_distribution.counts
        if bloom_counts and any(bloom_counts.values()):
            fig_bloom = px.bar(
                x=list(bloom_counts.keys()),
                y=list(bloom_counts.values()),
                title="Bloom's Taxonomy Distribution",
                labels={"x": "Bloom Level", "y": "Questions"},
                color=list(bloom_counts.keys()),
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig_bloom.update_layout(
                showlegend=False,
                margin=dict(t=40, b=20, l=10, r=10),
                xaxis_title=None,
            )
            st.plotly_chart(
                fig_bloom, use_container_width=True, theme="streamlit"
            )
        else:
            st.info("Bloom distribution data unavailable.")

    with ch2:
        diff_dist = analytics.difficulty_distribution
        if diff_dist and any(diff_dist.values()):
            fig_diff = px.pie(
                names=list(diff_dist.keys()),
                values=list(diff_dist.values()),
                title="Difficulty Distribution",
                color=list(diff_dist.keys()),
                color_discrete_map={
                    "Easy": "#52B788",
                    "Medium": "#F4A261",
                    "Hard": "#E63946",
                },
                hole=0.42,
            )
            fig_diff.update_traces(textposition="outside", textinfo="label+percent")
            fig_diff.update_layout(
                showlegend=True,
                margin=dict(t=40, b=20, l=10, r=10),
            )
            st.plotly_chart(
                fig_diff, use_container_width=True, theme="streamlit"
            )
        else:
            st.info("Difficulty distribution data unavailable.")

    # ── Charts row 2: CO Coverage + Marks ────────────────────────────────────
    ch3, ch4 = st.columns(2)

    with ch3:
        co_cov = analytics.co_coverage
        if co_cov:
            fig_co = px.bar(
                x=list(co_cov.keys()),
                y=list(co_cov.values()),
                title="Course Outcome Coverage",
                labels={"x": "Course Outcome", "y": "Questions"},
                color=list(co_cov.keys()),
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_co.update_layout(
                showlegend=False,
                margin=dict(t=40, b=20, l=10, r=10),
                xaxis_title=None,
            )
            st.plotly_chart(
                fig_co, use_container_width=True, theme="streamlit"
            )
        else:
            st.info("CO coverage data unavailable.")

    with ch4:
        marks_dist = analytics.marks_distribution
        if marks_dist:
            fig_marks = px.bar(
                x=list(marks_dist.keys()),
                y=list(marks_dist.values()),
                title="Marks per Question",
                labels={"x": "Question", "y": "Marks"},
                color_discrete_sequence=["#4C78A8"],
            )
            fig_marks.update_layout(
                showlegend=False,
                margin=dict(t=40, b=20, l=10, r=10),
                xaxis_title=None,
            )
            st.plotly_chart(
                fig_marks, use_container_width=True, theme="streamlit"
            )
        else:
            st.info("Marks distribution data unavailable.")

    # ── Knowledge sources list ────────────────────────────────────────────────
    if analytics.knowledge_sources_used:
        st.divider()
        st.markdown(
            "<div class='section-eyebrow'>Knowledge Sources</div>",
            unsafe_allow_html=True,
        )
        for doc in analytics.knowledge_sources_used:
            st.markdown(f"- 📄 {doc}")
    else:
        st.caption(
            "No knowledge-base sources were used "
            "(vector store empty or RAG unavailable)."
        )


def _render_review_tab(reviewer: Optional[Any]) -> None:
    """Render the AI Review tab showing the ReviewerResult.

    Args:
        reviewer: ``ReviewerResult`` from session state, or None.
    """
    if reviewer is None:
        st.info(
            "AI review was not run for this assessment. "
            "Generate a new assessment to get a full review.",
            icon="ℹ️",
        )
        return

    if reviewer.error:
        st.warning(
            f"⚠️ The AI reviewer encountered an error: **{reviewer.error}**\n\n"
            "Partial results may be shown below.",
            icon="⚠️",
        )

    # ── Quality score display ─────────────────────────────────────────────────
    sc1, sc2, sc3 = st.columns([1, 1, 2])
    with sc1:
        score = reviewer.quality_score or 0.0
        score_color = (
            "#52B788" if score >= 75
            else "#F4A261" if score >= 50
            else "#E63946"
        )
        st.markdown(
            f"<div class='score-ring' style='color:{score_color}'>"
            f"{score:.0f}</div>"
            f"<div class='score-label'>LLM Quality Score / 100</div>",
            unsafe_allow_html=True,
        )
    with sc2:
        balance_icon = "✅" if reviewer.difficulty_balance_ok else "⚠️"
        st.markdown(f"**Difficulty Balance:** {balance_icon}")
        if reviewer.duplicate_question_ids:
            st.markdown(
                f"**Duplicate IDs:** {', '.join(reviewer.duplicate_question_ids)}"
            )
        else:
            st.markdown("**Duplicates:** None detected ✅")
    with sc3:
        if reviewer.bloom_coverage:
            st.markdown(
                "<div class='section-eyebrow'>Bloom Coverage</div>",
                unsafe_allow_html=True,
            )
            bc_cols = st.columns(3)
            bloom_levels = list(reviewer.bloom_coverage.items())
            for i, (level, covered) in enumerate(bloom_levels):
                with bc_cols[i % 3]:
                    icon = "✅" if covered else "❌"
                    st.caption(f"{icon} {level}")

    st.divider()

    # ── Strengths / Weaknesses / Suggestions ─────────────────────────────────
    rev_c1, rev_c2 = st.columns(2)

    with rev_c1:
        st.markdown("#### 💪 Strengths")
        if reviewer.strengths:
            for s in reviewer.strengths:
                st.success(s, icon="✅")
        else:
            st.caption("No strengths recorded.")

        st.markdown("#### 📋 Suggestions")
        if reviewer.suggestions:
            for s in reviewer.suggestions:
                st.info(s, icon="💡")
        else:
            st.caption("No suggestions recorded.")

    with rev_c2:
        st.markdown("#### ⚠️ Weaknesses")
        if reviewer.weaknesses:
            for w in reviewer.weaknesses:
                st.warning(w, icon="⚠️")
        else:
            st.caption("No weaknesses recorded.")

        if reviewer.reviewer_notes:
            st.divider()
            st.markdown("#### 🗒️ Reviewer Notes")
            st.markdown(reviewer.reviewer_notes)


def _render_downloads_tab(assessment: Any, export_bytes: Dict[str, bytes]) -> None:
    """Render the Downloads tab with buttons for MD / Word / PDF.

    If ``export_bytes`` is empty (e.g. history-loaded run), bytes are
    regenerated on-the-fly via ``DownloadEngine``.

    Args:
        assessment: Current ``Assessment`` object.
        export_bytes: Pre-built export bytes dict (may be empty).
    """
    meta = assessment.metadata
    base_name = (
        f"edupilot_{meta.course_code or meta.course_name.replace(' ', '_')}"
    ).lower()

    # Regenerate missing bytes (history runs have no cached bytes)
    if not export_bytes:
        with st.spinner("Generating export files…"):
            try:
                from downloads import DownloadEngine
                engine = DownloadEngine()
                md_text: str = engine.export_markdown(assessment)
                export_bytes = {
                    "markdown": md_text.encode("utf-8"),
                }
                try:
                    export_bytes["docx"] = engine.export_word(assessment)
                except Exception as exc:
                    logger.warning("Word export failed during regeneration: %s", exc)
                try:
                    export_bytes["pdf"] = engine.export_pdf(assessment)
                except Exception as exc:
                    logger.warning("PDF export failed during regeneration: %s", exc)

                # Cache for the session so we don't regenerate on every rerun
                st.session_state[SS_EXPORT_BYTES] = export_bytes
                st.success("Export files generated.", icon="✅")
            except Exception as exc:
                st.error(f"Export generation failed: {exc}", icon="❌")
                logger.exception("DownloadEngine failed during tab render: %s", exc)
                return

    st.markdown(
        "<div class='section-eyebrow'>Download Assessment</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "All formats contain the full assessment: questions, answer keys, "
        "Bloom/CO metadata, and a university-branded header."
    )
    st.divider()

    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        with st.container(border=True):
            st.markdown("#### 📄 Markdown")
            st.caption("Plain text with tables — ideal for version control and editing.")
            md_bytes = export_bytes.get("markdown", b"")
            if md_bytes:
                st.download_button(
                    label="⬇️ Download .md",
                    data=md_bytes,
                    file_name=f"{base_name}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    type="primary",
                )
            else:
                st.warning("Markdown export unavailable.", icon="⚠️")

    with dl2:
        with st.container(border=True):
            st.markdown("#### 📝 Word (.docx)")
            st.caption("Professional document with university header/footer and page numbers.")
            docx_bytes = export_bytes.get("docx", b"")
            if docx_bytes:
                st.download_button(
                    label="⬇️ Download .docx",
                    data=docx_bytes,
                    file_name=f"{base_name}.docx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument"
                        ".wordprocessingml.document"
                    ),
                    use_container_width=True,
                    type="primary",
                )
            else:
                st.warning("Word export unavailable.", icon="⚠️")

    with dl3:
        with st.container(border=True):
            st.markdown("#### 📕 PDF")
            st.caption("Print-ready A4 document with Bloom/CO metadata per question.")
            pdf_bytes = export_bytes.get("pdf", b"")
            if pdf_bytes:
                st.download_button(
                    label="⬇️ Download .pdf",
                    data=pdf_bytes,
                    file_name=f"{base_name}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                )
            else:
                st.warning("PDF export unavailable.", icon="⚠️")

    # JSON fallback (always available)
    st.divider()
    st.markdown("<div class='section-eyebrow'>Raw Data</div>", unsafe_allow_html=True)
    assessment_dict = assessment.to_dict()
    st.download_button(
        label="⬇️ Download JSON (raw data)",
        data=json.dumps(assessment_dict, indent=2, ensure_ascii=False),
        file_name=f"{base_name}.json",
        mime="application/json",
    )
    st.caption(
        "JSON export contains the full structured assessment data including "
        "all metadata and answer keys."
    )


# ---------------------------------------------------------------------------
# View Assessment page (tabbed)
# ---------------------------------------------------------------------------

def render_view_page() -> None:
    """Render the View Assessment page with four tabs."""
    st.title("📄 Assessment Output")

    # Flash-fix: show pending success banner from generation
    if SS_PENDING_SUCCESS in st.session_state:
        msg = st.session_state.pop(SS_PENDING_SUCCESS)
        st.success(msg, icon="✅")

    assessment = st.session_state.get(SS_ASSESSMENT)
    if assessment is None:
        st.info(
            "No assessment generated yet. "
            "Go to **✍️ Generate Assessment** to create one.",
            icon="ℹ️",
        )
        return

    rag_sources = st.session_state.get(SS_RAG_SOURCES, [])
    analytics   = st.session_state.get(SS_ANALYTICS)
    reviewer    = st.session_state.get(SS_REVIEWER)
    export_bytes: Dict[str, bytes] = st.session_state.get(SS_EXPORT_BYTES, {})

    # If loaded from history, reviewer may be nested inside analytics
    if reviewer is None and analytics is not None:
        reviewer = getattr(analytics, "reviewer_result", None)

    # Compact header metrics
    meta = assessment.metadata
    hm1, hm2, hm3, hm4 = st.columns(4)
    hm1.metric("Questions", assessment.question_count)
    hm2.metric("Total Marks", assessment.total_marks)
    hm3.metric("Type", meta.assessment_type.value
               if hasattr(meta.assessment_type, "value")
               else str(meta.assessment_type))
    hm4.metric(
        "Duration",
        f"{meta.duration_minutes} min" if meta.duration_minutes else "Flexible",
    )

    # History badge
    if st.session_state.get(SS_HISTORY_RUN_ID):
        st.caption(
            f"📜 Viewing history run: `{st.session_state[SS_HISTORY_RUN_ID]}`"
        )

    # Edit & Regenerate: send the plan back to the form for adjustment.
    plan_dict = st.session_state.get(SS_PLAN)
    if isinstance(plan_dict, dict) and plan_dict:
        if st.button(
            "✏️ Edit & Regenerate",
            help="Reopen the requirement form pre-filled with this "
                 "assessment's details so you can tweak and regenerate.",
        ):
            # Normalise through the typed plan model so malformed history
            # plans can never crash the Generate form.
            try:
                from models import AssessmentPlan
                st.session_state["_edit_plan"] = (
                    AssessmentPlan.from_dict(plan_dict).to_dict()
                )
            except Exception:
                logger.warning(
                    "Edit & Regenerate: could not normalise plan; "
                    "using raw dict.", exc_info=True,
                )
                st.session_state["_edit_plan"] = dict(plan_dict)
            st.session_state["_nav"] = "✍️ Generate Assessment"
            st.rerun()

    st.divider()

    tab_q, tab_a, tab_r, tab_d = st.tabs(
        ["📋 Questions", "📊 Analytics", "🔍 AI Review", "⬇️ Downloads"]
    )

    with tab_q:
        _render_questions_tab(assessment, rag_sources)

    with tab_a:
        _render_analytics_tab(analytics)

    with tab_r:
        _render_review_tab(reviewer)

    with tab_d:
        _render_downloads_tab(assessment, export_bytes)


# ---------------------------------------------------------------------------
# History page helpers — reconstruction with enum coercion
# ---------------------------------------------------------------------------

def _reconstruct_assessment(data: Dict[str, Any]) -> Any:
    """Reconstruct a typed ``Assessment`` object from a ``to_dict()`` snapshot.

    All string enum values are coerced back to their typed enum counterparts.

    Args:
        data: Dict produced by ``Assessment.to_dict()``.

    Returns:
        Assessment: Fully-typed assessment object.

    Raises:
        KeyError | ValueError: When required fields are missing or malformed.
    """
    from models import (
        Assessment, AssessmentMetadata, AssessmentType,
        BloomLevel, DifficultyLevel, Question, SourceAttribution,
    )

    raw_meta = data["metadata"]
    meta = AssessmentMetadata(
        title=raw_meta.get("title", ""),
        course_code=raw_meta.get("course_code", ""),
        course_name=raw_meta.get("course_name", ""),
        assessment_type=AssessmentType(raw_meta.get("assessment_type", "Quiz")),
        semester=raw_meta.get("semester", ""),
        duration_minutes=int(raw_meta.get("duration_minutes", 0)),
        total_marks=int(raw_meta.get("total_marks", 0)),
        department=raw_meta.get("department", ""),
        faculty_name=raw_meta.get("faculty_name", ""),
        test_date=raw_meta.get("test_date", ""),
        instructions=raw_meta.get("instructions", ""),
    )

    questions: List[Any] = []
    for q_raw in data.get("questions", []):
        sources = [
            SourceAttribution(
                document_name=s.get("document_name", ""),
                page_number=s.get("page_number"),
                relevance_score=float(s.get("relevance_score", 0.0)),
                excerpt=s.get("excerpt", ""),
            )
            for s in q_raw.get("sources", [])
        ]
        questions.append(
            Question(
                question_id=q_raw.get("question_id", ""),
                question_text=q_raw.get("question_text", ""),
                bloom_level=BloomLevel(q_raw.get("bloom_level", "Remember")),
                co_mapping=q_raw.get("co_mapping", []),
                difficulty=DifficultyLevel(q_raw.get("difficulty", "Medium")),
                marks=int(q_raw.get("marks", 0)),
                answer_key=q_raw.get("answer_key", ""),
                question_type=q_raw.get("question_type", "Short Answer"),
                sources=sources,
                notes=q_raw.get("notes", ""),
                options=[
                    str(o) for o in (q_raw.get("options") or [])
                    if str(o).strip()
                ],
                topic=q_raw.get("topic", ""),
                case_background=q_raw.get("case_background", ""),
            )
        )

    return Assessment(
        metadata=meta,
        questions=questions,
        generation_notes=data.get("generation_notes", ""),
    )


def _reconstruct_analytics(data: Optional[Dict[str, Any]]) -> Optional[Any]:
    """Reconstruct an ``AnalyticsReport`` from a ``to_dict()`` snapshot.

    Args:
        data: Dict produced by ``AnalyticsReport.to_dict()``, or None.

    Returns:
        AnalyticsReport | None
    """
    if not data:
        return None

    from models import AnalyticsReport, BloomDistribution

    bloom_raw = data.get("bloom_distribution", {})
    bloom_dist = BloomDistribution(
        counts=bloom_raw.get("counts", {})
    )

    reviewer = _reconstruct_reviewer(data.get("reviewer_result"))

    return AnalyticsReport(
        question_count=int(data.get("question_count", 0)),
        total_marks=int(data.get("total_marks", 0)),
        bloom_distribution=bloom_dist,
        co_coverage=data.get("co_coverage", {}),
        difficulty_distribution=data.get("difficulty_distribution", {}),
        knowledge_sources_used=data.get("knowledge_sources_used", []),
        estimated_time_saved_minutes=int(
            data.get("estimated_time_saved_minutes", 0)
        ),
        quality_score=float(data.get("quality_score", 0.0)),
        marks_distribution=data.get("marks_distribution", {}),
        reviewer_result=reviewer,
    )


def _reconstruct_reviewer(data: Optional[Dict[str, Any]]) -> Optional[Any]:
    """Reconstruct a ``ReviewerResult`` from a ``to_dict()`` snapshot.

    Args:
        data: Dict produced by ``ReviewerResult.to_dict()``, or None.

    Returns:
        ReviewerResult | None
    """
    if not data:
        return None

    from models import ReviewerResult

    return ReviewerResult(
        quality_score=float(data.get("quality_score", 0.0)),
        strengths=data.get("strengths", []),
        weaknesses=data.get("weaknesses", []),
        suggestions=data.get("suggestions", []),
        duplicate_question_ids=data.get("duplicate_question_ids", []),
        bloom_coverage=data.get("bloom_coverage", {}),
        difficulty_balance_ok=bool(data.get("difficulty_balance_ok", True)),
        reviewer_notes=data.get("reviewer_notes", ""),
        error=data.get("error"),
    )


def _load_run_into_session(run_id: str) -> bool:
    """Load a history run from disk into session state and navigate to view page.

    Reads ``runs/<run_id>.json``, reconstructs typed objects, populates session
    state (clearing old pipeline results), and sets navigation to View page.

    Args:
        run_id: UUID4 string identifying the run to load.

    Returns:
        bool: True on success, False on any read/parse/reconstruction error.
    """
    run_path = auth.user_runs_dir() / f"{run_id}.json"

    if not run_path.exists():
        st.error(f"Run file not found: `{run_id}`", icon="❌")
        return False

    try:
        raw = json.loads(run_path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Failed to read run file: {exc}", icon="❌")
        logger.error("History load read error run_id=%s: %s", run_id, exc)
        return False

    try:
        assessment = _reconstruct_assessment(raw["assessment"])
        analytics  = _reconstruct_analytics(raw.get("analytics"))
        reviewer   = _reconstruct_reviewer(raw.get("reviewer"))
        rag_raw    = raw.get("rag_sources", [])
    except Exception as exc:
        st.error(f"Failed to reconstruct run data: {exc}", icon="❌")
        logger.exception("History reconstruction error run_id=%s: %s", run_id, exc)
        return False

    from models import SourceAttribution
    rag_sources = [
        SourceAttribution(
            document_name=s.get("document_name", ""),
            page_number=s.get("page_number"),
            relevance_score=float(s.get("relevance_score", 0.0)),
            excerpt=s.get("excerpt", ""),
        )
        for s in rag_raw
    ]

    st.session_state[SS_ASSESSMENT]      = assessment
    st.session_state[SS_PLAN]            = raw.get("plan", {})
    st.session_state[SS_RAG_SOURCES]     = rag_sources
    st.session_state[SS_ANALYTICS]       = analytics
    st.session_state[SS_REVIEWER]        = reviewer
    st.session_state[SS_EXPORT_BYTES]    = {}   # regenerated lazily in Downloads tab
    st.session_state[SS_PIPELINE_RESULT] = None
    st.session_state[SS_HISTORY_RUN_ID]  = run_id

    logger.info("History run loaded: run_id=%s", run_id)
    return True


# ---------------------------------------------------------------------------
# History page — delete helper
# ---------------------------------------------------------------------------

# Session-state key for the run awaiting a delete-confirmation click
_SS_DELETE_CONFIRM = "_history_delete_confirm"


def _delete_run(run_id: str, index_path: Path) -> tuple[bool, str]:
    """Delete a single run: remove its JSON file and update the index.

    Args:
        run_id: UUID string of the run to delete.
        index_path: Path to ``runs/index.json``.

    Returns:
        tuple[bool, str]: (success, message) where message describes the
        outcome or the error encountered.
    """
    run_file = auth.user_runs_dir() / f"{run_id}.json"

    # ── Remove run JSON ──────────────────────────────────────────────────────
    try:
        if run_file.exists():
            run_file.unlink()
            logger.info("Deleted run file: %s", run_file)
        else:
            logger.warning("Run file already absent: %s", run_file)
    except OSError as exc:
        logger.error("Failed to delete run file %s: %s", run_file, exc)
        return False, f"Could not delete run file: {exc}"

    # ── Update index ─────────────────────────────────────────────────────────
    try:
        raw_index: List[Dict[str, Any]] = json.loads(
            index_path.read_text(encoding="utf-8")
        )
        updated = [r for r in raw_index if r.get("run_id") != run_id]
        index_path.write_text(
            json.dumps(updated, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "Index updated after delete: removed run_id=%s (%d → %d entries)",
            run_id, len(raw_index), len(updated),
        )
    except Exception as exc:
        logger.error("Failed to update index after delete: %s", exc)
        return False, f"Run file removed but index update failed: {exc}"

    return True, "Run deleted successfully."


# ---------------------------------------------------------------------------
# History page
# ---------------------------------------------------------------------------

def render_history_page() -> None:
    """Render the Run History page with search, filters, and delete actions."""
    st.title("📜 Assessment History")
    st.markdown(
        "Browse and manage all past generation runs. "
        "Use the search box and filters to find a specific assessment, "
        "or **Delete** failed / obsolete runs to keep your history tidy."
    )

    index_path = auth.user_runs_dir() / "index.json"

    if not index_path.exists():
        st.info(
            "No past runs found. "
            "Generate your first assessment to start the history.",
            icon="ℹ️",
        )
        return

    try:
        index: List[Dict[str, Any]] = json.loads(
            index_path.read_text(encoding="utf-8")
        )
        if not isinstance(index, list):
            index = []
    except Exception as exc:
        st.error(f"Failed to read history index: {exc}", icon="❌")
        return

    if not index:
        st.info("History index is empty.", icon="ℹ️")
        return

    # ── Search & filter controls ──────────────────────────────────────────────
    with st.container(border=True):
        filter_col1, filter_col2, filter_col3 = st.columns([4, 3, 3])

        with filter_col1:
            search_query = st.text_input(
                "🔍 Search",
                placeholder="Course name or code…",
                key="_hist_search",
                label_visibility="collapsed",
            ).strip().lower()

        with filter_col2:
            # Collect the distinct assessment types actually present in the index
            known_types = sorted(
                {r.get("assessment_type", "") for r in index if r.get("assessment_type")}
            )
            type_options = ["All types"] + known_types
            selected_type = st.selectbox(
                "Assessment type",
                type_options,
                key="_hist_type_filter",
                label_visibility="collapsed",
            )

        with filter_col3:
            selected_status = st.selectbox(
                "Status",
                ["All statuses", "✅ Successful", "❌ Failed"],
                key="_hist_status_filter",
                label_visibility="collapsed",
            )

    # ── Apply filters ─────────────────────────────────────────────────────────
    filtered: List[Dict[str, Any]] = []
    for record in index:
        # Search filter
        if search_query:
            haystack = (
                record.get("course_name", "").lower()
                + " "
                + record.get("course_code", "").lower()
            )
            if search_query not in haystack:
                continue

        # Type filter
        if selected_type != "All types":
            if record.get("assessment_type", "") != selected_type:
                continue

        # Status filter
        if selected_status == "✅ Successful" and not record.get("success", False):
            continue
        if selected_status == "❌ Failed" and record.get("success", False):
            continue

        filtered.append(record)

    # ── Summary line ──────────────────────────────────────────────────────────
    total_count    = len(index)
    filtered_count = len(filtered)
    is_filtered    = (
        bool(search_query)
        or selected_type != "All types"
        or selected_status != "All statuses"
    )

    if is_filtered:
        st.caption(
            f"Showing **{filtered_count}** of {total_count} run(s) — newest first."
        )
    else:
        st.caption(f"Showing **{total_count}** run(s) — newest first.")

    # ── Empty state ───────────────────────────────────────────────────────────
    if not filtered:
        st.info(
            "No runs match your current search / filter. "
            "Try clearing the search box or changing the filters.",
            icon="🔎",
        )
        return

    st.divider()

    # ── Run cards ─────────────────────────────────────────────────────────────
    pending_delete = st.session_state.get(_SS_DELETE_CONFIRM)

    for i, record in enumerate(filtered):
        run_id      = record.get("run_id", "")
        course_name = record.get("course_name", "—")
        course_code = record.get("course_code", "")
        atype       = record.get("assessment_type", "—")
        q_count     = record.get("question_count", 0)
        total_marks = record.get("total_marks", 0)
        quality     = record.get("quality_score", 0.0)
        rev_score   = record.get("reviewer_score")
        success     = record.get("success", False)
        fatal_error = record.get("fatal_error")
        started_at  = record.get("started_at", "")[:16].replace("T", " ")

        awaiting_confirm = pending_delete == run_id

        with st.container(border=True):
            # ── Card header row ────────────────────────────────────────────
            if awaiting_confirm:
                # Confirmation banner spans full width
                st.warning(
                    f"⚠️ **Delete this run permanently?**  "
                    f"The JSON file and history entry for `{run_id[:8]}…` will be removed.",
                    icon="🗑️",
                )
                confirm_col, cancel_col, _ = st.columns([2, 2, 6])
                with confirm_col:
                    if st.button(
                        "🗑️ Yes, delete",
                        key=f"hist_confirm_del_{run_id}",
                        type="primary",
                        use_container_width=True,
                    ):
                        ok, msg = _delete_run(run_id, index_path)
                        st.session_state.pop(_SS_DELETE_CONFIRM, None)
                        if ok:
                            st.toast("Run deleted.", icon="🗑️")
                        else:
                            st.error(msg, icon="❌")
                        st.rerun()
                with cancel_col:
                    if st.button(
                        "Cancel",
                        key=f"hist_cancel_del_{run_id}",
                        use_container_width=True,
                    ):
                        st.session_state.pop(_SS_DELETE_CONFIRM, None)
                        st.rerun()
            else:
                # Normal card layout
                hcol, btn_reopen, btn_delete = st.columns([7, 1, 1])

                with hcol:
                    status_badge = "✅" if success else "❌"
                    title_str = (
                        f"{status_badge} **{course_name}**"
                        + (f" `{course_code}`" if course_code else "")
                        + f" — {atype}"
                    )
                    st.markdown(title_str)

                    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                    mc1.caption(f"🕒 {started_at}")
                    mc2.caption(f"📝 {q_count} questions")
                    mc3.caption(f"🏆 {total_marks} marks")
                    mc4.caption(f"📊 Quality: {quality:.0f}")
                    mc5.caption(
                        f"🔍 Reviewer: {rev_score:.0f}"
                        if rev_score is not None
                        else "🔍 Reviewer: —"
                    )

                    if fatal_error:
                        st.caption(f"⚠️ Error: {fatal_error}")

                    # Stage summary strip
                    stages = record.get("stages", [])
                    if stages:
                        stage_icons = []
                        for stg in stages:
                            s_status = stg.get("status", "?")
                            s_name   = stg.get("stage", "?")
                            icon = (
                                "✅" if s_status == "ok"
                                else "⚠️" if (
                                    s_status in ("skipped", "error")
                                    and stg.get("warning")
                                )
                                else "❌" if s_status == "error"
                                else "⏭️"
                            )
                            stage_icons.append(f"{icon} {s_name}")
                        st.caption(" · ".join(stage_icons))

                with btn_reopen:
                    if success and st.button(
                        "Reopen",
                        key=f"history_reopen_{run_id}",
                        use_container_width=True,
                        type="primary",
                    ):
                        with st.spinner(f"Loading run {run_id[:8]}…"):
                            ok = _load_run_into_session(run_id)
                        if ok:
                            st.session_state["_nav"] = "📄 View Assessment"
                            st.rerun()

                with btn_delete:
                    if st.button(
                        "Delete",
                        key=f"history_delete_{run_id}",
                        use_container_width=True,
                    ):
                        st.session_state[_SS_DELETE_CONFIRM] = run_id
                        st.rerun()

        if i < len(filtered) - 1:
            st.markdown("")  # spacer between cards


# ---------------------------------------------------------------------------
# My Settings page
# ---------------------------------------------------------------------------

# Human-readable labels for well-known setting keys.
_SETTING_LABELS: Dict[str, str] = {
    "department":  "Department",
    "semester":    "Semester",
    "faculty_name": "Faculty Name",
}

# Placeholder hints for well-known keys.
_SETTING_HINTS: Dict[str, str] = {
    "department":   "e.g. Computer Science & Engineering",
    "semester":     "e.g. Semester 5, 2026-27",
    "faculty_name": "e.g. Dr. A. Sharma",
}


def render_settings_page() -> None:
    """Render the My Settings page for the signed-in user.

    Reads all keys present in runs/<username>/settings.json and exposes them
    for editing.  Well-known keys get labelled inputs; any extra keys are
    shown generically so future keys surface without code changes.  Saving
    writes back via auth.save_user_setting so every call-site that uses
    auth.get_user_setting picks up the new values.
    """
    st.title("⚙️ My Settings")
    st.markdown(
        "Your saved defaults pre-fill every assessment form.  "
        "Update them here so you never have to re-type them."
    )
    st.divider()

    # Load current settings from disk (empty dict if file missing).
    try:
        settings_file = auth.user_runs_dir() / "settings.json"
        current: dict = (
            json.loads(settings_file.read_text(encoding="utf-8"))
            if settings_file.exists()
            else {}
        )
    except Exception as exc:
        st.error(f"Could not read your settings: {exc}")
        current = {}

    # Build the ordered list of keys to display: well-known first, then any
    # extras found in the file (future-proof).
    known_keys = list(_SETTING_LABELS.keys())
    extra_keys = [k for k in current if k not in known_keys]
    all_keys = known_keys + extra_keys

    with st.form("settings_form"):
        st.subheader("📋 Form defaults")
        st.caption(
            "These values pre-fill the Generate Assessment form.  "
            "Changing them here takes effect on the next form load."
        )

        new_values: Dict[str, str] = {}
        for key in all_keys:
            label = _SETTING_LABELS.get(key, key.replace("_", " ").title())
            hint  = _SETTING_HINTS.get(key, "")
            new_values[key] = st.text_input(
                label,
                value=current.get(key, ""),
                placeholder=hint,
                key=f"setting_{key}",
            )

        st.divider()
        saved = st.form_submit_button("💾 Save Settings", type="primary")

    if saved:
        errors: List[str] = []
        for key, value in new_values.items():
            try:
                auth.save_user_setting(key, value.strip())
            except Exception as exc:
                errors.append(f"{key}: {exc}")
        if errors:
            for e in errors:
                st.error(f"Failed to save setting — {e}")
        else:
            st.success(
                "✅ Settings saved!  Your next assessment form will use "
                "these values.",
                icon="✅",
            )
            logger.info(
                "Settings updated for user %s: %s",
                auth.current_username(),
                list(new_values.keys()),
            )

    # ── Preview card ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("🔍 Current saved values (from disk)", expanded=False):
        try:
            settings_file = auth.user_runs_dir() / "settings.json"
            on_disk: dict = (
                json.loads(settings_file.read_text(encoding="utf-8"))
                if settings_file.exists()
                else {}
            )
        except Exception:
            on_disk = {}

        if on_disk:
            for k, v in on_disk.items():
                label = _SETTING_LABELS.get(k, k.replace("_", " ").title())
                st.markdown(f"**{label}:** {v or '*(empty)*'}")
        else:
            st.info("No settings saved yet.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Application entry point called by Streamlit."""
    _inject_css()

    # ── Authentication gate ─────────────────────────────────────────────
    # Attempt silent cookie re-authentication, then show the login page
    # when no user is signed in.  Everything below requires a user.
    auth.try_cookie_login()
    if not auth.is_authenticated():
        # Render login/registration and stop — even when login just
        # succeeded, we finish this run so the auth-cookie component can
        # mount; it triggers the rerun that brings up the app.
        auth.render_auth_page()
        return

    # ── Session hygiene ─────────────────────────────────────────────────
    # If a different user signs in within the same browser session, clear
    # all user-scoped state so nothing leaks between accounts.
    _current_user = auth.current_username()
    if st.session_state.get("_active_user") != _current_user:
        for _key in (
            SS_ASSESSMENT, SS_PLAN, SS_RAG_SOURCES, SS_ANALYTICS,
            SS_REVIEWER, SS_EXPORT_BYTES, SS_PIPELINE_RESULT,
            SS_PENDING_SUCCESS, SS_HISTORY_RUN_ID, _SS_DELETE_CONFIRM,
        ):
            st.session_state.pop(_key, None)
        st.session_state["_active_user"] = _current_user

    # ── Forced password-change gate ─────────────────────────────────────
    # If the account carries a must_change_password flag (set by the
    # forgot-password flow), block ALL other pages and force the user to
    # set a permanent password first.  No sidebar is rendered so there is
    # no navigation path around this gate.
    if auth.get_must_change_password(_current_user):
        st.warning(
            "🔒 **Security notice:** Your account was issued a temporary "
            "password.  You must set a permanent password before you can "
            "use EduPilot.  Use the form below — no other pages are "
            "accessible until this step is complete.",
            icon="🔒",
        )
        # Minimal top-right logout so the user is never truly stuck.
        with st.sidebar:
            st.markdown(f"👤 **{auth.current_display_name()}**")
            auth.render_logout(location="sidebar")
        auth.render_change_password(forced=True)
        return

    # Handle programmatic navigation: write the target page directly into
    # the sidebar radio's own session-state key BEFORE the radio widget is
    # instantiated.  This makes navigation *sticky* — previously the radio
    # kept its old value, so the next rerun (e.g. a form submit) silently
    # bounced the user back to the old page and swallowed the submission.
    if "_nav" in st.session_state:
        st.session_state["nav_radio"] = st.session_state.pop("_nav")

    page = render_sidebar()

    if page == "🏠 Home":
        render_landing()
    elif page == "📚 Knowledge Base":
        render_knowledge_page()
    elif page == "✍️ Generate Assessment":
        render_generate_page()
    elif page == "📄 View Assessment":
        render_view_page()
    elif page == "📜 History":
        render_history_page()
    elif page == "⚙️ My Settings":
        render_settings_page()
    elif page == "🔐 Change Password":
        auth.render_change_password()
    else:
        render_landing()


if __name__ == "__main__":
    main()
else:
    # Streamlit imports the module; call main() unconditionally.
    main()
