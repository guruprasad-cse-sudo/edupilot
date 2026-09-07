"""Faculty-authored automaton (DFA/NFA) diagram parsing and rendering.

Separate from rag.py's PDF-diagram extraction — this handles the OPPOSITE
direction: a faculty describing a brand-new diagram (not sourced from any
uploaded document) in a structured text block within Additional
Instructions, which gets rendered into an actual image and attached to
whichever generated question references it.

Scoped deliberately narrow: only diagrams that are naturally graph-shaped
(states/nodes + labelled transitions/edges) are supported — state
machines, automata, and similar. Freeform ASCII-art diagrams are NOT
parsed; asking a faculty to describe an exact circuit schematic or a
freehand geometric figure this way wouldn't render reliably, so this
module doesn't attempt it. See prompts.py's ASSESSMENT_SYSTEM_PROMPT for
the matching instruction that tells the LLM to actually use a provided
automaton spec in its question text (needed so the automatic diagram-to-
question matching below has something to match against).

Input format (embedded anywhere in Additional Instructions)::

    [DFA]
    Topic: Finite Automata
    States: q0, q1, q2, q3
    Alphabet: 0, 1
    Start: q0
    Accept: q3
    Transitions:
    q0, 0 -> q1
    q0, 1 -> q0
    q1, 0 -> q2
    q1, 1 -> q0
    q2, 0 -> q2
    q2, 1 -> q3
    q3, 0 -> q1
    q3, 1 -> q0
    [/DFA]

``[NFA]`` and ``[AUTOMATON]`` are accepted as aliases of the same format
(NFA transitions may list multiple target states after ``->``, comma-
separated, and/or use "epsilon" as the symbol for an epsilon-transition).

``Topic:`` is optional but strongly recommended whenever more than one
automaton is provided in the same request (e.g. 10 problems each with
their own diagram) — see :func:`_attach_diagrams_to_questions` for why:
without it, diagrams are matched to questions purely by state-name
overlap, which breaks down if multiple automatons reuse the same
generic names (q0, q1, q2...), a very likely scenario. When given,
``Topic:`` must match one of the exact topic names in the faculty's
"Topics to Cover" field — matching is then unambiguous regardless of
state naming.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import config
from logging_utils import get_logger

logger = get_logger(__name__)

_AUTOMATON_BLOCK_RE = re.compile(
    r"\[(DFA|NFA|AUTOMATON)\](.*?)\[/\1\]", re.IGNORECASE | re.DOTALL
)
_TRANSITION_LINE_RE = re.compile(r"^(.+?)\s*,\s*(.+?)\s*->\s*(.+)$")

_CUSTOM_DIAGRAM_SUBDIR = "custom_automata"


def strip_automaton_blocks(text: str) -> str:
    """Remove any raw [DFA]/[NFA]/[AUTOMATON] block from student-facing text.

    Deterministic safety net alongside the prompt instruction (rule 8 in
    ASSESSMENT_SYSTEM_PROMPT) telling the LLM never to paste the raw
    structured block into question_text — prompt compliance isn't
    guaranteed, and this markup must never reach a printed paper (it's
    for internal diagram rendering only). Applied to every question's
    text as a final pass regardless of whether a diagram was actually
    attached.

    Args:
        text: A question's raw text, possibly containing a leaked block.

    Returns:
        str: The text with any [DFA]...[/DFA] (or NFA/AUTOMATON) block
        removed and surrounding whitespace collapsed. Unchanged if no
        block is present.
    """
    if not text or "[" not in text:
        return text
    cleaned = _AUTOMATON_BLOCK_RE.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def parse_automaton_specs(text: str) -> List[dict]:
    """Extract every ``[DFA]``/``[NFA]``/``[AUTOMATON]`` block from *text*.

    Args:
        text: Free text to scan (typically ``plan.extra_instructions``).

    Returns:
        List of parsed spec dicts (see :func:`_parse_single_block`), one
        per well-formed block found. Malformed blocks (missing states,
        start state, or transitions) are silently skipped — a faculty
        typo shouldn't crash generation, it just means no diagram gets
        rendered for that block.
    """
    if not text:
        return []
    specs = []
    for match in _AUTOMATON_BLOCK_RE.finditer(text):
        kind = match.group(1).upper()
        spec = _parse_single_block(match.group(2))
        if spec:
            spec["kind"] = kind
            specs.append(spec)
        else:
            logger.warning(
                "parse_automaton_specs(): found a [%s] block but couldn't "
                "parse it (need at least States, Start, and Transitions) "
                "— skipping.", kind,
            )
    return specs


def _parse_single_block(body: str) -> Optional[dict]:
    """Parse one automaton block's body into a structured spec.

    Args:
        body: Text between the opening and closing tags.

    Returns:
        dict with ``states``, ``alphabet``, ``start``, ``accept``,
        ``transitions`` (list of ``(from, symbol, to)`` tuples), and
        ``topic`` (optional — empty string if not given), or None if the
        block is missing required fields.
    """
    states: List[str] = []
    alphabet: List[str] = []
    start = ""
    accept: List[str] = []
    transitions: List[Tuple[str, str, str]] = []
    topic = ""

    in_transitions = False
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("topic:"):
            topic = line.split(":", 1)[1].strip()
            in_transitions = False
        elif low.startswith("states:"):
            states = [s.strip() for s in line.split(":", 1)[1].split(",") if s.strip()]
            in_transitions = False
        elif low.startswith("alphabet:"):
            alphabet = [s.strip() for s in line.split(":", 1)[1].split(",") if s.strip()]
            in_transitions = False
        elif low.startswith("start:"):
            start = line.split(":", 1)[1].strip()
            in_transitions = False
        elif low.startswith("accept:"):
            accept = [s.strip() for s in line.split(":", 1)[1].split(",") if s.strip()]
            in_transitions = False
        elif low.startswith("transitions:"):
            in_transitions = True
        elif in_transitions:
            m = _TRANSITION_LINE_RE.match(line)
            if not m:
                continue
            frm, sym, tos = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            for to in (t.strip() for t in tos.split(",") if t.strip()):
                transitions.append((frm, sym, to))

    if not states or not start or not transitions:
        return None
    return {
        "states": states,
        "alphabet": alphabet,
        "start": start,
        "accept": accept,
        "transitions": transitions,
        "topic": topic,
    }


def render_automaton_diagram(spec: dict) -> str:
    """Render a parsed automaton spec to a PNG using matplotlib.

    Pure-Python rendering (matplotlib only, no external binary) —
    deliberately NOT using Graphviz, even though it produces a cleaner
    default layout, because Graphviz's Python package is only a wrapper
    around the separate ``dot`` command-line tool, which must be
    installed at the OS level. Render's native Python runtime has no
    way to install system-wide packages (only Docker-based deploys do),
    so a Graphviz-based renderer would silently fail in production even
    though it works in any environment with ``dot`` pre-installed
    (observed in practice: "failed to execute PosixPath('dot')..." at
    runtime). matplotlib is a pure pip dependency, so this works
    anywhere the rest of the app already runs.

    States are arranged on a circle; multiple transitions between the
    same state pair are combined onto one edge with a comma-separated
    label; self-loops get their own small arc above the state; an arrow
    from empty space marks the start state; accepting states get a
    double circle — all standard automaton diagram conventions.

    Args:
        spec: A dict as returned by :func:`parse_automaton_specs`.

    Returns:
        str: Path to the rendered PNG file. Empty string if rendering
        failed — callers should treat that as "no diagram", not raise.
    """
    import math

    import matplotlib
    matplotlib.use("Agg")  # headless — no display server in this environment
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, Circle, FancyArrowPatch

    out_dir = config.diagrams_dir / _CUSTOM_DIAGRAM_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic filename from the spec's content, so re-generating
    # with the exact same automaton reuses the same file instead of
    # piling up duplicates.
    fingerprint = hashlib.md5(repr(sorted(spec.items())).encode()).hexdigest()[:12]
    out_path = out_dir / f"automaton_{fingerprint}.png"

    try:
        states = spec["states"]
        n = len(states)
        accept_set = set(spec.get("accept", []))
        start = spec["start"]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_aspect("equal")
        ax.axis("off")

        radius_layout = max(2.5, n * 0.7)
        node_radius = 0.35
        positions = {}
        for i, s in enumerate(states):
            angle = 2 * math.pi * i / n - math.pi / 2  # start at top, clockwise
            positions[s] = (
                radius_layout * math.cos(angle),
                radius_layout * math.sin(angle),
            )

        grouped: Dict[Tuple[str, str], List[str]] = {}
        for frm, sym, to in spec["transitions"]:
            grouped.setdefault((frm, to), []).append(sym)

        for (frm, to), syms in grouped.items():
            label = ",".join(syms)
            x1, y1 = positions[frm]
            x2, y2 = positions[to]
            if frm == to:
                loop_cx, loop_cy = x1, y1 + node_radius * 1.55
                loop_w, loop_h = node_radius * 1.3, node_radius * 1.1
                ax.add_patch(Arc(
                    (loop_cx, loop_cy), loop_w, loop_h, angle=0,
                    theta1=15, theta2=345, color="black", lw=1.3, zorder=4,
                ))
                end_angle = math.radians(15)
                ex = loop_cx + (loop_w / 2) * math.cos(end_angle)
                ey = loop_cy + (loop_h / 2) * math.sin(end_angle)
                tangent_angle = end_angle + math.pi / 2
                hx = ex - 0.12 * math.cos(tangent_angle)
                hy = ey - 0.12 * math.sin(tangent_angle)
                ax.annotate(
                    "", xy=(ex, ey), xytext=(hx, hy),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3),
                    zorder=4,
                )
                ax.text(loop_cx, loop_cy + loop_h * 0.85, label,
                        ha="center", va="center", fontsize=10)
            else:
                dx, dy = x2 - x1, y2 - y1
                dist = math.hypot(dx, dy)
                ux, uy = dx / dist, dy / dist
                sx, sy = x1 + ux * node_radius, y1 + uy * node_radius
                ex, ey = x2 - ux * node_radius, y2 - uy * node_radius
                has_reverse = (to, frm) in grouped
                connectionstyle = "arc3,rad=0.15" if has_reverse else "arc3,rad=0.0"
                ax.add_patch(FancyArrowPatch(
                    (sx, sy), (ex, ey), arrowstyle="-|>", mutation_scale=15,
                    color="black", lw=1.2, connectionstyle=connectionstyle,
                ))
                mx, my = (sx + ex) / 2, (sy + ey) / 2
                perp_x, perp_y = -uy, ux
                offset = 0.28 if has_reverse else 0.2
                ax.text(
                    mx + perp_x * offset, my + perp_y * offset, label,
                    ha="center", va="center", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"),
                )

        # Start-state arrow (standard automaton convention: an arrow
        # from nowhere pointing at the start state).
        sx, sy = positions[start]
        ax.annotate(
            "", xy=(sx - node_radius, sy),
            xytext=(sx - node_radius * 2.2, sy),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4),
        )

        for s in states:
            x, y = positions[s]
            ax.add_patch(Circle(
                (x, y), node_radius, facecolor="white", edgecolor="black",
                lw=1.4, zorder=5,
            ))
            if s in accept_set:
                ax.add_patch(Circle(
                    (x, y), node_radius * 0.8, facecolor="none",
                    edgecolor="black", lw=1.2, zorder=5,
                ))
            ax.text(x, y, s, ha="center", va="center", fontsize=11, zorder=6)

        margin = radius_layout + node_radius * 4
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)
    except Exception as exc:  # noqa: BLE001 — never let this break generation
        logger.warning(
            "render_automaton_diagram(): rendering failed: %s", exc,
        )
        return ""


def render_all_automaton_specs(text: str) -> List[dict]:
    """Parse and render every automaton spec found in *text*.

    Convenience wrapper combining :func:`parse_automaton_specs` and
    :func:`render_automaton_diagram` for every block found.

    Args:
        text: Free text to scan (typically ``plan.extra_instructions``).

    Returns:
        List of dicts, each the original spec plus an added
        ``"image_path"`` key (only for specs that rendered
        successfully — failed renders are dropped, not included with
        an empty path).
    """
    rendered = []
    for spec in parse_automaton_specs(text):
        path = render_automaton_diagram(spec)
        if path:
            spec["image_path"] = path
            rendered.append(spec)
    return rendered
