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
        ``transitions`` (list of ``(from, symbol, to)`` tuples), or
        None if the block is missing required fields.
    """
    states: List[str] = []
    alphabet: List[str] = []
    start = ""
    accept: List[str] = []
    transitions: List[Tuple[str, str, str]] = []

    in_transitions = False
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("states:"):
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
    }


def render_automaton_diagram(spec: dict) -> str:
    """Render a parsed automaton spec to a PNG using Graphviz.

    Multiple transitions between the same pair of states (e.g. "0" and
    "1" both going q0 -> q0) are combined onto a single edge with a
    comma-separated label, matching standard automaton diagram
    convention, rather than drawing two overlapping arrows.

    Args:
        spec: A dict as returned by :func:`parse_automaton_specs`.

    Returns:
        str: Path to the rendered PNG file. Empty string if rendering
        failed (e.g. Graphviz not available) — callers should treat
        that as "no diagram", not raise.
    """
    import graphviz  # lazy import — only needed for this feature

    out_dir = config.diagrams_dir / _CUSTOM_DIAGRAM_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic filename from the spec's content, so re-generating
    # with the exact same automaton reuses the same file instead of
    # piling up duplicates.
    fingerprint = hashlib.md5(repr(sorted(spec.items())).encode()).hexdigest()[:12]
    out_path = out_dir / f"automaton_{fingerprint}"

    try:
        dot = graphviz.Digraph(format="png")
        dot.attr(rankdir="LR")
        dot.attr("node", shape="circle", fontsize="12")

        accept_set = set(spec.get("accept", []))
        for state in spec["states"]:
            shape = "doublecircle" if state in accept_set else "circle"
            dot.node(state, state, shape=shape)

        # Start-state arrow (standard automaton convention: an arrow
        # from nowhere pointing at the start state).
        dot.node("__start__", "", shape="point")
        dot.edge("__start__", spec["start"])

        grouped: Dict[Tuple[str, str], List[str]] = {}
        for frm, sym, to in spec["transitions"]:
            grouped.setdefault((frm, to), []).append(sym)
        for (frm, to), syms in grouped.items():
            dot.edge(frm, to, label=",".join(syms))

        rendered_path = dot.render(str(out_path), cleanup=True)
        return rendered_path
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
