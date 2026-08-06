"""
EduPilot AI Faculty Assistant
Module: downloads.py
Version: 4.1.0
Author: EduPilot Team
Purpose: Professional document generation engine. Exports an Assessment to
         Markdown, Microsoft Word (.docx via python-docx), and PDF (via
         ReportLab). All three formats share a common parsed-structure
         intermediate representation so that documents are built from typed
         data rather than raw LLM text strings. Includes university-grade
         header/footer, page numbers, Bloom / CO / difficulty metadata,
         question numbering, and an answer-key section. Gracefully handles
         optional/missing fields so a partially-generated assessment never
         produces a broken document.
"""

from __future__ import annotations

import io
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import KeepTogether

from docx import Document as DocxDocument
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from logging_utils import get_logger
from models import Assessment, Question

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants — University branding defaults
# ---------------------------------------------------------------------------

_UNIVERSITY_NAME = "Dayananda Sagar Academy of Technology and Management"
_UNIVERSITY_TAGLINE = "Department of Academic Excellence"
_DATE_FMT = "%d %B %Y"


# ===========================================================================
# Shared intermediate representation
# ===========================================================================


@dataclass
class ParsedQuestion:
    """Flat, rendering-friendly view of a single Question.

    All fields are strings so formatters need no further type handling.
    """

    number: int
    question_id: str
    text: str
    question_type: str
    bloom_level: str
    difficulty: str
    co_mapping: str          # comma-joined list, e.g. "CO1, CO3"
    marks: str               # stringified integer
    answer_key: str
    notes: str
    options: List[str] = field(default_factory=list)  # MCQ answer options


@dataclass
class ParsedAssessment:
    """Structured, renderer-neutral view of an Assessment.

    Built once by :class:`AssessmentParser` and consumed by all three
    exporters so that parsing logic lives in exactly one place.
    """

    # Header fields
    university: str
    department: str
    title: str
    course_code: str
    course_name: str
    assessment_type: str
    semester: str
    duration: str           # "90 minutes"
    total_marks: str        # "100 marks"
    faculty_name: str
    date_generated: str
    test_date: str
    instructions: str

    # Body
    questions: List[ParsedQuestion] = field(default_factory=list)
    generation_notes: str = ""

    # Derived helpers
    has_answer_key: bool = False
    has_instructions: bool = False
    has_generation_notes: bool = False


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class AssessmentParser:
    """Convert an :class:`~models.Assessment` into a :class:`ParsedAssessment`.

    The parser is the single source of truth for flattening the typed
    Assessment model into plain strings that all exporters can consume
    without re-implementing the same null-checks.
    """

    # Strips leading/trailing whitespace and normalises internal whitespace runs
    _WS_RE = re.compile(r"\s+")

    @classmethod
    def parse(cls, assessment: Assessment) -> ParsedAssessment:
        """Parse an Assessment into a rendering-ready ParsedAssessment.

        Args:
            assessment: The typed Assessment object to render.

        Returns:
            A fully populated :class:`ParsedAssessment` ready for export.
        """
        meta = assessment.metadata

        # ── Header ──────────────────────────────────────────────────────────
        department = cls._safe(meta.department) or _UNIVERSITY_TAGLINE
        faculty_name = cls._safe(meta.faculty_name) or "Faculty"
        semester = cls._safe(meta.semester) or "Current Semester"
        instructions = cls._safe(meta.instructions)

        duration = (
            f"{meta.duration_minutes} minutes"
            if meta.duration_minutes > 0
            else "As specified"
        )
        total_marks_computed = assessment.total_marks or meta.total_marks
        total_marks = (
            f"{total_marks_computed} marks"
            if total_marks_computed > 0
            else "As specified"
        )

        # ── Questions ───────────────────────────────────────────────────────
        parsed_questions: List[ParsedQuestion] = []
        has_any_answer_key = False

        for idx, q in enumerate(assessment.questions, start=1):
            co_str = ", ".join(q.co_mapping) if q.co_mapping else "—"
            answer_key_text = cls._safe(q.answer_key)
            if answer_key_text:
                has_any_answer_key = True

            parsed_questions.append(
                ParsedQuestion(
                    number=idx,
                    question_id=cls._safe(q.question_id) or f"Q{idx}",
                    text=cls._clean(q.question_text),
                    question_type=cls._safe(q.question_type) or "Short Answer",
                    bloom_level=q.bloom_level.value
                    if hasattr(q.bloom_level, "value")
                    else str(q.bloom_level),
                    difficulty=q.difficulty.value
                    if hasattr(q.difficulty, "value")
                    else str(q.difficulty),
                    co_mapping=co_str,
                    marks=str(q.marks),
                    answer_key=answer_key_text,
                    notes=cls._safe(q.notes),
                    options=[
                        cls._safe(o) for o in getattr(q, "options", []) or []
                        if cls._safe(o)
                    ],
                )
            )

        generation_notes = cls._safe(assessment.generation_notes)

        return ParsedAssessment(
            university=_UNIVERSITY_NAME,
            department=department,
            title=cls._safe(meta.title) or "Assessment",
            course_code=cls._safe(meta.course_code) or "",
            course_name=cls._safe(meta.course_name) or "",
            assessment_type=meta.assessment_type.value
            if hasattr(meta.assessment_type, "value")
            else str(meta.assessment_type),
            semester=semester,
            duration=duration,
            total_marks=total_marks,
            faculty_name=faculty_name,
            date_generated=datetime.now().strftime(_DATE_FMT),
            test_date=cls._safe(getattr(meta, "test_date", "")),
            instructions=instructions,
            questions=parsed_questions,
            generation_notes=generation_notes,
            has_answer_key=has_any_answer_key,
            has_instructions=bool(instructions),
            has_generation_notes=bool(generation_notes),
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _safe(value: object) -> str:
        """Coerce *value* to a stripped string; return empty string for None."""
        if value is None:
            return ""
        return str(value).strip()

    @classmethod
    def _clean(cls, text: str) -> str:
        """Strip, collapse internal whitespace runs, and return the result."""
        return cls._WS_RE.sub(" ", text.strip()) if text else ""


# ===========================================================================
# Markdown exporter
# ===========================================================================


class MarkdownExporter:
    """Export an :class:`~models.Assessment` to a clean Markdown string.

    The output is spec-compliant CommonMark with proper heading hierarchy,
    a metadata table, numbered questions, per-question metadata, and an
    answer-key section.
    """

    def export(self, assessment: Assessment) -> str:
        """Render the assessment as a Markdown document.

        Args:
            assessment: The typed Assessment to export.

        Returns:
            str: UTF-8 Markdown document string.

        Raises:
            ValueError: If *assessment* is None.
        """
        if assessment is None:
            raise ValueError("assessment must not be None")

        logger.info(
            "MarkdownExporter: exporting '%s'",
            assessment.metadata.title,
        )

        pa = AssessmentParser.parse(assessment)
        lines: List[str] = []

        # ── Document header ──────────────────────────────────────────────────
        lines += [
            f"# {pa.university}",
            f"### {pa.department}",
            "",
            f"## {pa.title}",
            "",
        ]

        # Metadata table
        meta_rows = [
            ("Course Code", pa.course_code or "—"),
            ("Course Name", pa.course_name or "—"),
            ("Assessment Type", pa.assessment_type),
            ("Semester", pa.semester),
            ("Duration", pa.duration),
            ("Total Marks", pa.total_marks),
            ("Faculty", pa.faculty_name),
            ("Test Date", pa.test_date or "—"),
            ("Generated On", pa.date_generated),
        ]
        lines += [
            "| Field | Details |",
            "|-------|---------|",
        ]
        for label, value in meta_rows:
            lines.append(f"| **{label}** | {value} |")
        lines.append("")

        # Instructions
        if pa.has_instructions:
            lines += [
                "---",
                "",
                "### Instructions",
                "",
                pa.instructions,
                "",
            ]

        # ── Questions ────────────────────────────────────────────────────────
        lines += [
            "---",
            "",
            "## Questions",
            "",
        ]

        for q in pa.questions:
            lines += [
                f"### {q.number}. {q.text}",
                "",
            ]
            if q.options:
                lines += [
                    f"- ( ) **{chr(65 + i)}.** {opt}"
                    for i, opt in enumerate(q.options)
                ] + [""]
            lines += [
                "| Attribute | Value |",
                "|-----------|-------|",
                f"| **Type** | {q.question_type} |",
                f"| **Bloom Level** | {q.bloom_level} |",
                f"| **Difficulty** | {q.difficulty} |",
                f"| **CO Mapping** | {q.co_mapping} |",
                f"| **Marks** | {q.marks} |",
                "",
            ]
            if q.notes:
                lines += [f"> **Note:** {q.notes}", ""]

        # ── Answer key ───────────────────────────────────────────────────────
        if pa.has_answer_key:
            lines += [
                "---",
                "",
                "## Answer Key",
                "",
                "> *For faculty use only — not to be distributed to students.*",
                "",
            ]
            for q in pa.questions:
                answer_text = q.answer_key if q.answer_key else "*Not provided*"
                lines += [
                    f"### {q.number}. {q.question_id}",
                    "",
                    answer_text,
                    "",
                ]

        # ── Generation notes ─────────────────────────────────────────────────
        if pa.has_generation_notes:
            lines += [
                "---",
                "",
                "## Agent Notes",
                "",
                pa.generation_notes,
                "",
            ]

        # ── Footer ───────────────────────────────────────────────────────────
        lines += [
            "---",
            "",
            f"*Generated by EduPilot AI Faculty Assistant · {pa.date_generated}*",
            "",
        ]

        md = "\n".join(lines)
        logger.info(
            "MarkdownExporter: produced %d characters", len(md)
        )
        return md


# ===========================================================================
# Word exporter
# ===========================================================================


class WordExporter:
    """Export an :class:`~models.Assessment` to a professional DOCX byte stream.

    Uses ``python-docx`` to construct a formatted Word document with:
    - University header with branding colours
    - Metadata table
    - Numbered questions with Bloom/CO/difficulty metadata
    - Answer-key section (page-break separated, faculty-only notice)
    - Automatic page numbers in the footer
    """

    # Colour palette (RGB tuples)
    _COLOUR_PRIMARY = RGBColor(0x1A, 0x37, 0x6C)   # deep navy
    _COLOUR_ACCENT = RGBColor(0xC8, 0x9B, 0x20)    # gold
    _COLOUR_LIGHT = RGBColor(0xF0, 0xF4, 0xF8)     # light blue-grey

    def export(self, assessment: Assessment) -> bytes:
        """Render the assessment as a DOCX byte stream.

        Args:
            assessment: The typed Assessment to export.

        Returns:
            bytes: Raw .docx bytes suitable for ``st.download_button``.

        Raises:
            ValueError: If *assessment* is None.
        """
        if assessment is None:
            raise ValueError("assessment must not be None")

        logger.info(
            "WordExporter: exporting '%s'", assessment.metadata.title
        )

        pa = AssessmentParser.parse(assessment)
        doc = DocxDocument()

        self._configure_page(doc)
        self._add_header(doc, pa)
        self._add_footer(doc, pa)
        self._add_title_block(doc, pa)
        self._add_metadata_table(doc, pa)

        if pa.has_instructions:
            self._add_instructions(doc, pa)

        self._add_questions(doc, pa)

        if pa.has_answer_key:
            self._add_answer_key(doc, pa)

        if pa.has_generation_notes:
            self._add_generation_notes(doc, pa)

        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        raw = buf.read()
        logger.info("WordExporter: produced %d bytes", len(raw))
        return raw

    # ── Document layout ───────────────────────────────────────────────────────

    @staticmethod
    def _configure_page(doc: DocxDocument) -> None:
        """Set A4 page size and 2.5 cm margins."""
        from docx.oxml.ns import nsmap  # noqa: PLC0415
        from docx.oxml import OxmlElement  # noqa: PLC0415

        section = doc.sections[0]
        section.page_height = Inches(11.69)   # A4
        section.page_width = Inches(8.27)
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.2)
        section.right_margin = Inches(1.2)

    # ── Header / footer ───────────────────────────────────────────────────────

    def _add_header(self, doc: DocxDocument, pa: ParsedAssessment) -> None:
        """Add a branded header with university name and course info."""
        section = doc.sections[0]
        header = section.header
        header.is_linked_to_previous = False

        # Clear default paragraph
        for p in header.paragraphs:
            p.clear()

        p = header.paragraphs[0] if header.paragraphs else header.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

        run_uni = p.add_run(pa.university)
        run_uni.bold = True
        run_uni.font.size = Pt(11)
        run_uni.font.color.rgb = self._COLOUR_PRIMARY

        p.add_run(" · ")

        run_dept = p.add_run(pa.department)
        run_dept.font.size = Pt(10)
        run_dept.font.color.rgb = self._COLOUR_PRIMARY

        # Bottom border
        self._set_paragraph_border(p, bottom_color="1A376C", bottom_size=12)

    @staticmethod
    def _add_footer(doc: DocxDocument, pa: ParsedAssessment) -> None:
        """Add page-numbered footer with assessment title."""
        section = doc.sections[0]
        footer = section.footer
        footer.is_linked_to_previous = False

        for p in footer.paragraphs:
            p.clear()

        p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

        run_title = p.add_run(f"{pa.title}  |  ")
        run_title.font.size = Pt(8)

        # Auto page number field
        fld = OxmlElement("w:fldChar")
        fld.set(qn("w:fldCharType"), "begin")
        p.runs[-1]._r.append(fld)

        instrText = OxmlElement("w:instrText")
        instrText.text = "PAGE"
        p.runs[-1]._r.append(instrText)

        fld_end = OxmlElement("w:fldChar")
        fld_end.set(qn("w:fldCharType"), "end")
        p.runs[-1]._r.append(fld_end)

        run_of = p.add_run(" of ")
        run_of.font.size = Pt(8)

        fld2 = OxmlElement("w:fldChar")
        fld2.set(qn("w:fldCharType"), "begin")
        p.runs[-1]._r.append(fld2)

        instrText2 = OxmlElement("w:instrText")
        instrText2.text = "NUMPAGES"
        p.runs[-1]._r.append(instrText2)

        fld2_end = OxmlElement("w:fldChar")
        fld2_end.set(qn("w:fldCharType"), "end")
        p.runs[-1]._r.append(fld2_end)

        run_date = p.add_run(f"  |  Generated: {pa.date_generated}")
        run_date.font.size = Pt(8)

    # ── Content sections ──────────────────────────────────────────────────────

    def _add_title_block(
        self, doc: DocxDocument, pa: ParsedAssessment
    ) -> None:
        """Add the assessment title centred with primary colour."""
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(pa.title)
        run.bold = True
        run.font.size = Pt(16)
        run.font.color.rgb = self._COLOUR_PRIMARY
        doc.add_paragraph()  # spacer

    def _add_metadata_table(
        self, doc: DocxDocument, pa: ParsedAssessment
    ) -> None:
        """Render a two-column metadata info table."""
        rows = [
            ("Course Code", pa.course_code or "—"),
            ("Course Name", pa.course_name or "—"),
            ("Assessment Type", pa.assessment_type),
            ("Semester", pa.semester),
            ("Duration", pa.duration),
            ("Total Marks", pa.total_marks),
            ("Faculty", pa.faculty_name),
            ("Test Date", pa.test_date or "—"),
            ("Generated On", pa.date_generated),
        ]

        table = doc.add_table(rows=len(rows), cols=2)
        table.style = "Table Grid"

        for i, (label, value) in enumerate(rows):
            row = table.rows[i]
            cell_label = row.cells[0]
            cell_value = row.cells[1]

            cell_label.text = label
            cell_value.text = value

            # Style label cell
            run_label = cell_label.paragraphs[0].runs[0]
            run_label.bold = True
            run_label.font.color.rgb = self._COLOUR_PRIMARY
            run_label.font.size = Pt(10)

            cell_value.paragraphs[0].runs[0].font.size = Pt(10)

            # Shade alternate rows
            if i % 2 == 0:
                self._shade_cell(cell_label, "F0F4F8")
                self._shade_cell(cell_value, "F0F4F8")

        # Set column widths
        for row in table.rows:
            row.cells[0].width = Inches(2.0)
            row.cells[1].width = Inches(4.0)

        doc.add_paragraph()

    def _add_instructions(
        self, doc: DocxDocument, pa: ParsedAssessment
    ) -> None:
        """Add a shaded instructions box."""
        h = doc.add_heading("Instructions", level=2)
        h.runs[0].font.color.rgb = self._COLOUR_PRIMARY

        p = doc.add_paragraph(pa.instructions)
        p.style = "Normal"
        p.paragraph_format.left_indent = Inches(0.3)
        doc.add_paragraph()

    def _add_questions(
        self, doc: DocxDocument, pa: ParsedAssessment
    ) -> None:
        """Add numbered questions with metadata sub-tables."""
        h = doc.add_heading("Questions", level=1)
        h.runs[0].font.color.rgb = self._COLOUR_PRIMARY
        doc.add_paragraph()

        for q in pa.questions:
            # Question heading
            p_q = doc.add_paragraph()
            run_num = p_q.add_run(f"{q.number}.  ")
            run_num.bold = True
            run_num.font.size = Pt(11)
            run_num.font.color.rgb = self._COLOUR_PRIMARY

            run_text = p_q.add_run(q.text)
            run_text.font.size = Pt(11)

            # MCQ options — one per line with a selection circle
            for i, opt in enumerate(q.options):
                p_opt = doc.add_paragraph(f"○  {chr(65 + i)}.  {opt}")
                p_opt.paragraph_format.left_indent = Inches(0.4)
                for run in p_opt.runs:
                    run.font.size = Pt(10.5)

            # Metadata bar (inline)
            meta_parts = [
                f"[{q.question_type}]",
                f"Bloom: {q.bloom_level}",
                f"Difficulty: {q.difficulty}",
                f"CO: {q.co_mapping}",
                f"Marks: {q.marks}",
            ]
            p_meta = doc.add_paragraph("    ".join(meta_parts))
            p_meta.paragraph_format.left_indent = Inches(0.25)
            for run in p_meta.runs:
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor(0x55, 0x66, 0x77)

            if q.notes:
                p_note = doc.add_paragraph(f"Note: {q.notes}")
                p_note.paragraph_format.left_indent = Inches(0.25)
                for run in p_note.runs:
                    run.italic = True
                    run.font.size = Pt(9)

            doc.add_paragraph()

    def _add_answer_key(
        self, doc: DocxDocument, pa: ParsedAssessment
    ) -> None:
        """Add answer-key section after a page break."""
        doc.add_page_break()

        h = doc.add_heading("Answer Key", level=1)
        h.runs[0].font.color.rgb = self._COLOUR_PRIMARY

        notice = doc.add_paragraph(
            "For faculty use only — not to be distributed to students."
        )
        for run in notice.runs:
            run.italic = True
            run.font.color.rgb = RGBColor(0xAA, 0x44, 0x00)
        doc.add_paragraph()

        for q in pa.questions:
            p_q = doc.add_paragraph()
            run_num = p_q.add_run(f"{q.number}. {q.question_id}  ")
            run_num.bold = True
            run_num.font.size = Pt(11)
            run_num.font.color.rgb = self._COLOUR_PRIMARY

            answer_text = q.answer_key if q.answer_key else "Not provided."
            p_ans = doc.add_paragraph(answer_text)
            p_ans.paragraph_format.left_indent = Inches(0.3)
            doc.add_paragraph()

    def _add_generation_notes(
        self, doc: DocxDocument, pa: ParsedAssessment
    ) -> None:
        """Add agent generation notes at the end."""
        h = doc.add_heading("Agent Notes", level=2)
        h.runs[0].font.color.rgb = self._COLOUR_PRIMARY
        doc.add_paragraph(pa.generation_notes)

    # ── XML helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _shade_cell(cell, fill_hex: str) -> None:
        """Apply a solid background shade to a table cell."""
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), fill_hex)
        tcPr.append(shd)

    @staticmethod
    def _set_paragraph_border(
        paragraph,
        bottom_color: str = "000000",
        bottom_size: int = 6,
    ) -> None:
        """Add a bottom border to a paragraph via direct XML manipulation."""
        pPr = paragraph._p.get_or_add_pPr()
        pBdr = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), str(bottom_size))
        bottom.set(qn("w:space"), "1")
        bottom.set(qn("w:color"), bottom_color)
        pBdr.append(bottom)
        pPr.append(pBdr)


# ===========================================================================
# PDF exporter
# ===========================================================================


class PDFExporter:
    """Export an :class:`~models.Assessment` to a professional PDF byte stream.

    Uses ``ReportLab`` with the parsed intermediate representation.
    Documents are built from typed :class:`ParsedAssessment` objects — never
    from raw LLM output strings — to avoid ``Paragraph()`` XML-parsing issues.

    Layout features:
    - A4 page with 2.5 cm margins
    - University header (coloured rule + name) and footer (title + page numbers)
    - Metadata grid table
    - Numbered questions with metadata tag line
    - Answer-key section separated by a page break
    """

    # Colour palette (RGB 0–1 floats for ReportLab)
    _NAVY = colors.Color(0x1A / 255, 0x37 / 255, 0x6C / 255)
    _GOLD = colors.Color(0xC8 / 255, 0x9B / 255, 0x20 / 255)
    _LIGHT_BG = colors.Color(0xF0 / 255, 0xF4 / 255, 0xF8 / 255)
    _GREY_TEXT = colors.Color(0x55 / 255, 0x66 / 255, 0x77 / 255)
    _ANSWER_NOTICE_COLOR = colors.Color(0.6, 0.2, 0.0)

    def export(self, assessment: Assessment) -> bytes:
        """Render the assessment as a PDF byte stream.

        Args:
            assessment: The typed Assessment to export.

        Returns:
            bytes: Raw PDF bytes suitable for ``st.download_button``.

        Raises:
            ValueError: If *assessment* is None.
        """
        if assessment is None:
            raise ValueError("assessment must not be None")

        logger.info(
            "PDFExporter: exporting '%s'", assessment.metadata.title
        )

        pa = AssessmentParser.parse(assessment)
        styles = self._build_styles()

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            rightMargin=2.5 * cm,
            leftMargin=2.5 * cm,
            topMargin=2.5 * cm,
            bottomMargin=2.5 * cm,
            title=pa.title,
            author=pa.faculty_name,
            subject=f"{pa.course_code} {pa.assessment_type}",
        )

        story = self._build_story(pa, styles)

        doc.build(
            story,
            onFirstPage=self._make_page_callback(pa),
            onLaterPages=self._make_page_callback(pa),
        )

        buf.seek(0)
        raw = buf.read()
        logger.info("PDFExporter: produced %d bytes", len(raw))
        return raw

    # ── Style sheet ───────────────────────────────────────────────────────────

    def _build_styles(self) -> dict:
        """Build and return a dict of named ParagraphStyle objects."""
        base = getSampleStyleSheet()

        return {
            "title": ParagraphStyle(
                "EduTitle",
                parent=base["Heading1"],
                fontSize=18,
                textColor=self._NAVY,
                alignment=TA_CENTER,
                spaceAfter=6,
                fontName="Helvetica-Bold",
            ),
            "university": ParagraphStyle(
                "EduUniversity",
                parent=base["Normal"],
                fontSize=11,
                textColor=self._NAVY,
                alignment=TA_CENTER,
                fontName="Helvetica-Bold",
                spaceAfter=2,
            ),
            "department": ParagraphStyle(
                "EduDept",
                parent=base["Normal"],
                fontSize=9,
                textColor=self._NAVY,
                alignment=TA_CENTER,
                fontName="Helvetica",
                spaceAfter=4,
            ),
            "section_heading": ParagraphStyle(
                "EduSection",
                parent=base["Heading2"],
                fontSize=13,
                textColor=self._NAVY,
                fontName="Helvetica-Bold",
                spaceBefore=14,
                spaceAfter=6,
            ),
            "question_heading": ParagraphStyle(
                "EduQ",
                parent=base["Normal"],
                fontSize=11,
                leading=15,
                fontName="Helvetica-Bold",
                textColor=colors.black,
                spaceBefore=10,
                spaceAfter=2,
            ),
            "question_text": ParagraphStyle(
                "EduQText",
                parent=base["Normal"],
                fontSize=11,
                leading=15,
                fontName="Helvetica",
                textColor=colors.black,
                spaceAfter=2,
            ),
            "option_line": ParagraphStyle(
                "EduOption",
                parent=base["Normal"],
                fontSize=10.5,
                leading=14,
                fontName="Helvetica",
                textColor=colors.black,
                leftIndent=18,
                spaceAfter=1,
            ),
            "meta_tag": ParagraphStyle(
                "EduMeta",
                parent=base["Normal"],
                fontSize=8.5,
                textColor=self._GREY_TEXT,
                fontName="Helvetica",
                spaceAfter=4,
                leftIndent=12,
            ),
            "note_tag": ParagraphStyle(
                "EduNote",
                parent=base["Normal"],
                fontSize=9,
                textColor=self._GREY_TEXT,
                fontName="Helvetica-Oblique",
                spaceAfter=4,
                leftIndent=12,
            ),
            "answer_notice": ParagraphStyle(
                "EduAnsNotice",
                parent=base["Normal"],
                fontSize=9,
                textColor=self._ANSWER_NOTICE_COLOR,
                fontName="Helvetica-Oblique",
                alignment=TA_CENTER,
                spaceBefore=4,
                spaceAfter=10,
            ),
            "answer_id": ParagraphStyle(
                "EduAnsId",
                parent=base["Normal"],
                fontSize=11,
                fontName="Helvetica-Bold",
                textColor=self._NAVY,
                spaceBefore=8,
                spaceAfter=2,
            ),
            "answer_body": ParagraphStyle(
                "EduAnsBody",
                parent=base["Normal"],
                fontSize=10,
                fontName="Helvetica",
                textColor=colors.black,
                leading=14,
                leftIndent=16,
                spaceAfter=4,
                alignment=TA_JUSTIFY,
            ),
            "footer": ParagraphStyle(
                "EduFooter",
                parent=base["Normal"],
                fontSize=8,
                textColor=self._GREY_TEXT,
                alignment=TA_CENTER,
                fontName="Helvetica",
            ),
            "agent_notes": ParagraphStyle(
                "EduAgentNotes",
                parent=base["Normal"],
                fontSize=10,
                fontName="Helvetica-Oblique",
                textColor=self._GREY_TEXT,
                leading=14,
                spaceAfter=4,
            ),
            "normal": base["Normal"],
        }

    # ── Story builder ─────────────────────────────────────────────────────────

    def _build_story(
        self, pa: ParsedAssessment, styles: dict
    ) -> list:
        """Assemble the ReportLab story (flowable list) from ParsedAssessment.

        All text is sanitised through :meth:`_safe_para` so that stray
        HTML-like characters (``<``, ``>``, ``&``) never cause a Paragraph
        parse error.

        Args:
            pa: The parsed assessment.
            styles: Dict of named ParagraphStyle instances.

        Returns:
            list: List of ReportLab Flowable objects.
        """
        story = []

        # ── Title block ──────────────────────────────────────────────────────
        story.append(Paragraph(self._esc(pa.university), styles["university"]))
        story.append(Paragraph(self._esc(pa.department), styles["department"]))
        story.append(HRFlowable(width="100%", thickness=1.5, color=self._NAVY))
        story.append(Spacer(1, 8))
        story.append(Paragraph(self._esc(pa.title), styles["title"]))
        story.append(Spacer(1, 6))

        # ── Metadata table ───────────────────────────────────────────────────
        meta_data = [
            ["Course Code", pa.course_code or "—",
             "Assessment Type", pa.assessment_type],
            ["Course Name", pa.course_name or "—",
             "Semester", pa.semester],
            ["Duration", pa.duration,
             "Total Marks", pa.total_marks],
            ["Faculty", pa.faculty_name,
             "Test Date", pa.test_date or "—"],
            ["Generated On", pa.date_generated,
             "", ""],
        ]
        meta_table = Table(
            meta_data,
            colWidths=[3.5 * cm, 5.5 * cm, 3.5 * cm, 5.5 * cm],
        )
        meta_table.setStyle(
            TableStyle([
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("TEXTCOLOR", (0, 0), (0, -1), self._NAVY),
                ("TEXTCOLOR", (2, 0), (2, -1), self._NAVY),
                ("BACKGROUND", (0, 0), (-1, -1), self._LIGHT_BG),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1),
                 [self._LIGHT_BG, colors.white]),
                ("BOX", (0, 0), (-1, -1), 0.5, self._NAVY),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(meta_table)
        story.append(Spacer(1, 10))

        # ── Instructions ─────────────────────────────────────────────────────
        if pa.has_instructions:
            story.append(
                Paragraph("Instructions", styles["section_heading"])
            )
            story.append(
                Paragraph(self._esc(pa.instructions), styles["normal"])
            )
            story.append(Spacer(1, 6))

        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))

        # ── Questions ────────────────────────────────────────────────────────
        story.append(Paragraph("Questions", styles["section_heading"]))

        for q in pa.questions:
            meta_line = (
                f"[{q.question_type}]  •  Bloom: {q.bloom_level}  •  "
                f"Difficulty: {q.difficulty}  •  CO: {q.co_mapping}  •  "
                f"Marks: {q.marks}"
            )
            block = [
                Paragraph(
                    f"{q.number}.  {self._esc(q.text)}",
                    styles["question_text"],
                ),
            ]
            for i, opt in enumerate(q.options):
                block.append(
                    Paragraph(
                        f"○&nbsp;&nbsp;{chr(65 + i)}.&nbsp;&nbsp;"
                        f"{self._esc(opt)}",
                        styles["option_line"],
                    )
                )
            block.append(Paragraph(self._esc(meta_line), styles["meta_tag"]))
            if q.notes:
                block.append(
                    Paragraph(f"Note: {self._esc(q.notes)}", styles["note_tag"])
                )
            story.append(KeepTogether(block))

        # ── Answer key ───────────────────────────────────────────────────────
        if pa.has_answer_key:
            story.append(Spacer(1, 6))
            story.append(HRFlowable(width="100%", thickness=1, color=self._NAVY))
            story.append(Paragraph("Answer Key", styles["section_heading"]))
            story.append(
                Paragraph(
                    "For faculty use only — not to be distributed to students.",
                    styles["answer_notice"],
                )
            )

            for q in pa.questions:
                answer_text = q.answer_key if q.answer_key else "Not provided."
                ak_block = [
                    Paragraph(
                        f"{q.number}. {self._esc(q.question_id)}",
                        styles["answer_id"],
                    ),
                    Paragraph(
                        self._esc(answer_text), styles["answer_body"]
                    ),
                ]
                story.append(KeepTogether(ak_block))

        # ── Generation notes ─────────────────────────────────────────────────
        if pa.has_generation_notes:
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
            story.append(Paragraph("Agent Notes", styles["section_heading"]))
            story.append(
                Paragraph(self._esc(pa.generation_notes), styles["agent_notes"])
            )

        return story

    # ── Page callback ─────────────────────────────────────────────────────────

    def _make_page_callback(self, pa: ParsedAssessment):
        """Return an onPage callback that draws the page header and footer.

        Args:
            pa: ParsedAssessment supplying header/footer text.

        Returns:
            Callable[[canvas, doc], None] for SimpleDocTemplate.
        """
        navy = self._NAVY
        grey = self._GREY_TEXT
        title = pa.title

        def _draw_page(canvas, doc):
            canvas.saveState()
            w, h = A4

            # ── Top rule ────────────────────────────────────────────────────
            canvas.setStrokeColor(navy)
            canvas.setLineWidth(1.5)
            canvas.line(2.5 * cm, h - 1.5 * cm, w - 2.5 * cm, h - 1.5 * cm)

            # ── Footer rule + text ───────────────────────────────────────────
            canvas.setLineWidth(0.5)
            canvas.line(2.5 * cm, 1.8 * cm, w - 2.5 * cm, 1.8 * cm)

            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(grey)

            footer_left = title
            footer_right = f"Page {doc.page}"
            canvas.drawString(2.5 * cm, 1.3 * cm, footer_left)
            canvas.drawRightString(w - 2.5 * cm, 1.3 * cm, footer_right)

            canvas.restoreState()

        return _draw_page

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _esc(text: str) -> str:
        """Escape HTML special characters to prevent ReportLab Paragraph errors.

        Replaces ``&``, ``<``, and ``>`` with their HTML entity equivalents
        so that raw LLM output never crashes the paragraph parser.

        Args:
            text: Raw text string.

        Returns:
            str: HTML-safe string for ReportLab Paragraph().
        """
        if not text:
            return ""
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        return text


# ===========================================================================
# Convenience factory
# ===========================================================================


class DownloadEngine:
    """Facade that wraps all three exporters behind a single interface.

    Instantiate once and call the appropriate ``export_*`` method. All
    exceptions from the underlying exporters are caught, logged, and
    re-raised so callers receive a descriptive error message.

    Example::

        engine = DownloadEngine()
        md_text = engine.export_markdown(assessment)
        docx_bytes = engine.export_word(assessment)
        pdf_bytes = engine.export_pdf(assessment)
    """

    def __init__(self) -> None:
        self._md = MarkdownExporter()
        self._word = WordExporter()
        self._pdf = PDFExporter()

    def export_markdown(self, assessment: Assessment) -> str:
        """Export *assessment* to Markdown.

        Args:
            assessment: The Assessment to export.

        Returns:
            str: Markdown document.

        Raises:
            RuntimeError: If export fails.
        """
        try:
            return self._md.export(assessment)
        except Exception as exc:
            logger.exception("MarkdownExporter failed: %s", exc)
            raise RuntimeError(f"Markdown export failed: {exc}") from exc

    def export_word(self, assessment: Assessment) -> bytes:
        """Export *assessment* to a DOCX byte stream.

        Args:
            assessment: The Assessment to export.

        Returns:
            bytes: Raw .docx bytes.

        Raises:
            RuntimeError: If export fails.
        """
        try:
            return self._word.export(assessment)
        except Exception as exc:
            logger.exception("WordExporter failed: %s", exc)
            raise RuntimeError(f"Word export failed: {exc}") from exc

    def export_pdf(self, assessment: Assessment) -> bytes:
        """Export *assessment* to a PDF byte stream.

        Args:
            assessment: The Assessment to export.

        Returns:
            bytes: Raw PDF bytes.

        Raises:
            RuntimeError: If export fails.
        """
        try:
            return self._pdf.export(assessment)
        except Exception as exc:
            logger.exception("PDFExporter failed: %s", exc)
            raise RuntimeError(f"PDF export failed: {exc}") from exc
