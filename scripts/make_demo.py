#!/usr/bin/env python3
"""Generate memstrand's README demo: `docs/assets/demo.cast` (asciinema v2)
and `docs/assets/demo.svg` (a self-contained, looping, animated terminal
window — no JS, CSS keyframes only, so it animates inside GitHub's <img>
tag).

HONESTY RULE
------------
This script runs the *real* `memstrand` CLI (in-process, via
`memstrand.cli.main([...])`) against a fresh temp `MEMSTRAND_HOME`, with
no daemon in the way (`MEMSTRAND_PORT` is pointed at a free port so the
CLI's "is a daemon running?" probe fails fast and falls back to local
mode). Every line you see in the committed `docs/assets/demo.*` files is
real CLI output from a real embedder.

The only escape hatch is `--fake-embedder`, which swaps in a tiny
deterministic, offline, keyword-overlap embedder — ONLY useful for
iterating on layout/timing without waiting on (or having) a downloaded
fastembed model. It stamps a big "DRY RUN — fake embedder" watermark
into the SVG and refuses to write into `docs/assets/` (default output
dir becomes a temp directory instead). Never commit a fake-embedder
render.

Usage
-----
    # real demo (downloads the ~70MB fastembed model on first run)
    .venv/bin/python scripts/make_demo.py

    # fast, offline layout dry-run (writes under /tmp, watermarked)
    .venv/bin/python scripts/make_demo.py --fake-embedder

    # just one artifact, 2x playback speed
    .venv/bin/python scripts/make_demo.py --svg-only --speed 2

    # non-animated final-state frame, for screenshotting with qlmanage
    .venv/bin/python scripts/make_demo.py --fake-embedder --debug-static
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import random
import re
import shlex
import shutil
import socket
import sys
import tempfile
import time
import xml.sax.saxutils as xml_escape
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ASSETS = REPO_ROOT / "docs" / "assets"
WATERMARK_TEXT = "DRY RUN — fake embedder"

sys.path.insert(0, str(REPO_ROOT / "src"))

from memstrand import cli

# =============================================================================
# The scripted story
# =============================================================================


@dataclass(frozen=True)
class Step:
    """One beat of the demo: a narrated `# comment` followed by one real
    CLI invocation. `argv` entries may reference `{name}` placeholders
    captured from an earlier step's `save_as`."""

    comment: str
    argv: tuple[str, ...]
    save_as: str | None = None
    read_pause: float = 2.0  # seconds to hold the output on screen


DEMO_STEPS: tuple[Step, ...] = (
    Step(
        comment="Your agent learns a preference (global)",
        argv=("remember", "Sam prefers concise answers with code first", "--category", "preference"),
        save_as="pref_id",
        read_pause=1.5,
    ),
    Step(
        comment="...and a fact while working on one task",
        argv=("remember", "Playwright installed for the agent-testing task", "--task", "agent-testing"),
        read_pause=1.5,
    ),
    Step(
        comment="A different task asks about testing setup — no leak",
        argv=("recall", "testing setup", "--task", "disk-cleanup"),
        read_pause=1.8,
    ),
    Step(
        comment="The right task sees its fact + your globals",
        argv=("recall", "testing setup", "--task", "agent-testing"),
        read_pause=2.0,
    ),
    Step(
        comment="Facts change; edits supersede — they don't delete",
        argv=("edit", "{pref_id}", "--content", "Sam now prefers detailed explanations"),
        read_pause=1.6,
    ),
    Step(
        comment="Hand your local LLM a ready-made context block",
        argv=("context", "write the test plan", "--task", "agent-testing"),
        read_pause=2.2,
    ),
)


@dataclass(frozen=True)
class Scene:
    """A resolved `Step`: narration + the exact command line a viewer
    would see typed + the real stdout it produced."""

    comment: str
    command: str
    output_lines: tuple[str, ...]
    read_pause: float


class _DemoHashEmbedder:
    """Deterministic, offline, keyword-overlap embedder. ONLY used behind
    `--fake-embedder` for layout dry-runs — never for the committed demo.
    Same trick as the `HashEmbedder` test double in tests/test_cli.py."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in t.lower().split():
                idx = int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim
                v[idx] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _env_overrides(overrides: dict[str, str]):
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_session(fake_embedder: bool = False) -> list[Scene]:
    """Run `DEMO_STEPS` for real, in-process, against a fresh temp
    `MEMSTRAND_HOME` with no daemon in the way. Returns the resolved
    `Scene`s with actual CLI stdout."""
    home = Path(tempfile.mkdtemp(prefix="memstrand-demo-home-"))
    overrides = {
        "MEMSTRAND_HOME": str(home),
        "MEMSTRAND_PORT": str(_find_free_port()),
        "MEMSTRAND_TASK": "",
    }
    embedder_patch = (
        patch("memstrand.api.make_embedder", lambda settings: _DemoHashEmbedder())
        if fake_embedder
        else contextlib.nullcontext()
    )
    scenes: list[Scene] = []
    context: dict[str, str] = {}
    try:
        with _env_overrides(overrides), embedder_patch:
            for step in DEMO_STEPS:
                argv = [a.format(**context) for a in step.argv]
                display = "memstrand " + shlex.join(argv)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    rc = cli.main(argv)
                out = buf.getvalue()
                if rc != 0:
                    raise RuntimeError(f"demo step failed (rc={rc}): {display}\n{out}")
                if step.save_as:
                    stripped = out.strip()
                    context[step.save_as] = stripped.splitlines()[-1] if stripped else ""
                lines = tuple(out.rstrip("\n").split("\n")) if out.strip() else ()
                scenes.append(
                    Scene(comment=step.comment, command=display, output_lines=lines, read_pause=step.read_pause)
                )
    finally:
        shutil.rmtree(home, ignore_errors=True)
    return scenes


# =============================================================================
# ANSI -> colored tspans (maps basic SGR codes; strips/ignores unknown ones)
# =============================================================================

_ANSI_RE = re.compile(r"\x1b\[([0-9;]*)m")
_SGR_COLORS = {
    30: "#45475a", 31: "#f38ba8", 32: "#a6e3a1", 33: "#f9e2af",
    34: "#89b4fa", 35: "#cba6f7", 36: "#94e2d5", 37: "#cdd6f4",
    90: "#585b70", 91: "#f38ba8", 92: "#a6e3a1", 93: "#f9e2af",
    94: "#89b4fa", 95: "#cba6f7", 96: "#94e2d5", 97: "#cdd6f4",
}


def ansi_segments(text: str, default_color: str) -> list[tuple[str, str, bool]]:
    """Split `text` on SGR escape codes into `(plain_text, color, bold)`
    runs, mapping known foreground codes and bold (1), resetting on 0,
    and silently dropping anything else (unknown codes are stripped, not
    rendered)."""
    segments: list[tuple[str, str, bool]] = []
    color = default_color
    bold = False
    pos = 0
    for m in _ANSI_RE.finditer(text):
        if m.start() > pos:
            segments.append((text[pos : m.start()], color, bold))
        codes = m.group(1)
        for raw in (codes.split(";") if codes else ["0"]):
            code = int(raw) if raw else 0
            if code == 0:
                color, bold = default_color, False
            elif code == 1:
                bold = True
            elif code in _SGR_COLORS:
                color = _SGR_COLORS[code]
            # else: unrecognized SGR code -> stripped, no effect
        pos = m.end()
    if pos < len(text):
        segments.append((text[pos:], color, bold))
    return segments or [("", default_color, False)]


def _xml_text(s: str) -> str:
    return xml_escape.escape(s)


def _weight_attr(bold: bool) -> str:
    return ' font-weight="600"' if bold else ""


# =============================================================================
# Timeline: one computation shared by the .cast and .svg renderers so they
# always agree on exactly when everything happens.
# =============================================================================

CHAR_DELAY_RANGE = (0.035, 0.055)  # seconds/char, typing effect
COMMENT_HOLD = 0.3  # comment visible -> typing starts
POST_TYPE_PAUSE = 0.2  # "enter" pressed -> output starts
OUTPUT_LINE_STAGGER = 0.10
INTRO_HOLD = 0.4
LOOP_TAIL = 0.5

FONT_SIZE = 14
LINE_HEIGHT = 20
CHAR_WIDTH = FONT_SIZE * 0.6
LEFT_PAD = 36
PROMPT = "$ "
VISIBLE_ROWS = 22
TERM_X, TERM_Y, TERM_W, TERM_H = 20, 50, 860, 474
RIGHT_MARGIN = 12
MAX_LINE_WIDTH_PX = TERM_X + TERM_W - RIGHT_MARGIN - (TERM_X + LEFT_PAD)

COLOR_BG = "#1e1e2e"
COLOR_CHROME = "#181825"
COLOR_FG = "#cdd6f4"
COLOR_DIM = "#6c7086"
COLOR_COMMENT = "#6c7086"
COLOR_PROMPT = "#a6e3a1"
COLOR_WARN = "#f38ba8"


@dataclass
class LineEvent:
    kind: str  # "comment" | "command" | "output"
    row: int
    text: str
    color: str
    appear_time: float
    char_times: list[float] | None = None  # for "command": absolute per-char reveal times
    type_end: float | None = None
    width_px: float = 0.0


@dataclass
class Timeline:
    line_events: list[LineEvent]
    cast_events: list[tuple[float, str]]
    total_duration: float
    scroll_stops: list[tuple[float, int]]  # (time, row_count_visible_through)
    cols: int = 90
    rows: int = VISIBLE_ROWS


def build_timeline(scenes: list[Scene], seed: int = 1337, speed: float = 1.0) -> Timeline:
    rng = random.Random(seed)
    t = INTRO_HOLD / speed
    row = 0
    line_events: list[LineEvent] = []
    cast_events: list[tuple[float, str]] = []
    scroll_stops: list[tuple[float, int]] = [(0.0, 0)]

    def bump_row(at: float) -> None:
        nonlocal row
        row += 1
        scroll_stops.append((at, row))

    for scene in scenes:
        # -- comment -----------------------------------------------------
        comment_text = f"# {scene.comment}"
        line_events.append(
            LineEvent(
                kind="comment",
                row=row,
                text=comment_text,
                color=COLOR_COMMENT,
                appear_time=t,
                width_px=len(comment_text) * CHAR_WIDTH,
            )
        )
        cast_events.append((t, f"\x1b[2;3m{comment_text}\x1b[0m\r\n"))
        bump_row(t)
        t += COMMENT_HOLD / speed

        # -- command (typed) ----------------------------------------------
        cmd_start = t
        cast_events.append((t, f"\x1b[32m{PROMPT}\x1b[0m"))
        char_times: list[float] = []
        for ch in scene.command:
            delay = rng.uniform(*CHAR_DELAY_RANGE) / speed
            t += delay
            char_times.append(t)
            cast_events.append((t, ch))
        type_end = t
        cast_events.append((t, "\r\n"))
        line_events.append(
            LineEvent(
                kind="command",
                row=row,
                text=scene.command,
                color=COLOR_FG,
                appear_time=cmd_start,
                char_times=char_times,
                type_end=type_end,
                width_px=len(scene.command) * CHAR_WIDTH,
            )
        )
        bump_row(cmd_start)
        t += POST_TYPE_PAUSE / speed

        # -- output --------------------------------------------------------
        for i, line in enumerate(scene.output_lines):
            if i:
                t += OUTPUT_LINE_STAGGER / speed
            color = COLOR_WARN if line.strip().startswith("⚠") else (
                COLOR_DIM if line.startswith("  ") else COLOR_FG
            )
            line_events.append(
                LineEvent(
                    kind="output", row=row, text=line, color=color, appear_time=t,
                    width_px=len(line) * CHAR_WIDTH,
                )
            )
            cast_events.append((t, line + "\r\n"))
            bump_row(t)

        # spacer row (blank line between scenes)
        bump_row(t)

        t += 0.1 / speed + scene.read_pause / speed

    total_duration = t + LOOP_TAIL / speed
    return Timeline(
        line_events=line_events,
        cast_events=cast_events,
        total_duration=total_duration,
        scroll_stops=scroll_stops,
    )


# =============================================================================
# asciinema v2 .cast renderer
# =============================================================================


def render_cast(timeline: Timeline, title: str = "memstrand — 30 second demo") -> str:
    header = {
        "version": 2,
        "width": timeline.cols,
        "height": timeline.rows,
        "timestamp": int(time.time()),
        "env": {"TERM": "xterm-256color", "SHELL": "/bin/bash"},
        "title": title,
    }
    lines = [json.dumps(header)]
    for t, data in timeline.cast_events:
        lines.append(json.dumps([round(t, 4), "o", data]))
    return "\n".join(lines) + "\n"


# =============================================================================
# Animated SVG renderer
# =============================================================================


def _scroll_keyframes(timeline: Timeline) -> str:
    """Build a discrete (snap, not eased) scroll animation: each time a new
    row is appended beyond the visible window, the buffer jumps up by one
    line height. Expressed as CSS `%`-keyframes on `transform: translateY`."""
    total = timeline.total_duration
    stops: list[tuple[float, int]] = []
    last_offset = -1
    for at, row_count in timeline.scroll_stops:
        offset = max(0, row_count - VISIBLE_ROWS) * LINE_HEIGHT
        if offset != last_offset:
            stops.append((at, offset))
            last_offset = offset
    if not stops or stops[0][0] > 0:
        stops.insert(0, (0.0, 0))

    parts = []
    prev_offset = 0
    for i, (at, offset) in enumerate(stops):
        pct = max(0.0, min(100.0, at / total * 100))
        if i > 0:
            eps_pct = max(0.0, pct - 0.01)
            parts.append(f"{eps_pct:.4f}% {{ transform: translateY(-{prev_offset}px); }}")
        parts.append(f"{pct:.4f}% {{ transform: translateY(-{offset}px); }}")
        prev_offset = offset
    parts.append(f"100% {{ transform: translateY(-{prev_offset}px); }}")
    return "@keyframes scrollBuf {\n  " + "\n  ".join(parts) + "\n}"


def _mask_keyframes(timeline: Timeline) -> str:
    """A terminal-colored overlay that fades in right at the loop seam (and
    fades out right after), so the instantaneous reset back to frame 0
    doesn't read as a jarring jump."""
    total = timeline.total_duration
    fade = min(0.35, total * 0.03)
    p1 = fade / total * 100
    p2 = 100 - (fade / total * 100)
    return (
        "@keyframes maskFade {\n"
        f"  0% {{ opacity: 1; }}\n"
        f"  {p1:.4f}% {{ opacity: 0; }}\n"
        f"  {p2:.4f}% {{ opacity: 0; }}\n"
        "  100% { opacity: 1; }\n"
        "}"
    )


def render_svg(
    timeline: Timeline,
    fake_embedder: bool = False,
    debug_static: bool = False,
    title: str = "memstrand — 30 second demo",
) -> str:
    total = timeline.total_duration
    style_rules = [
        "@keyframes fadeIn { 0% { opacity: 0; } 100% { opacity: 1; } }",
        _scroll_keyframes(timeline) if not debug_static else "",
        _mask_keyframes(timeline) if not debug_static else "",
    ]

    line_markup: list[str] = []
    for idx, ev in enumerate(timeline.line_events):
        x = TERM_X + LEFT_PAD
        y = TERM_Y + 24 + ev.row * LINE_HEIGHT
        if ev.kind == "command":
            prompt_width = len(PROMPT) * CHAR_WIDTH
            if ev.width_px > MAX_LINE_WIDTH_PX - prompt_width:
                raise ValueError(
                    f"demo line too wide for the terminal frame ({ev.width_px:.0f}px > "
                    f"{MAX_LINE_WIDTH_PX - prompt_width:.0f}px available): {ev.text!r}"
                )
            line_markup.append(
                f'<text x="{x:.1f}" y="{y:.1f}" fill="{COLOR_PROMPT}" '
                f'font-size="{FONT_SIZE}">{_xml_text(PROMPT)}</text>'
            )
            cmd_x = x + prompt_width
            segs = ansi_segments(ev.text, ev.color)
            tspans = "".join(
                f'<tspan fill="{c}"{_weight_attr(b)}>{_xml_text(s)}</tspan>'
                for s, c, b in segs
                if s
            )
            style = ""
            if not debug_static and ev.type_end is not None:
                kf_name = f"type{idx}"
                style_rules.append(
                    f"@keyframes {kf_name} {{ "
                    f"0% {{ clip-path: inset(0 {ev.width_px:.1f}px 0 0); }} "
                    f"100% {{ clip-path: inset(0 0 0 0); }} }}"
                )
                dur = max(0.01, ev.type_end - ev.appear_time)
                style = (
                    f"animation: {kf_name} {dur:.3f}s steps({max(1, len(ev.text))}, end) "
                    f"{ev.appear_time:.3f}s both;"
                )
            line_markup.append(
                f'<text x="{cmd_x:.1f}" y="{y:.1f}" font-size="{FONT_SIZE}" '
                f'textLength="{ev.width_px:.1f}" lengthAdjust="spacingAndGlyphs" '
                f'style="{style}">{tspans}</text>'
            )
        else:
            if ev.width_px > MAX_LINE_WIDTH_PX:
                raise ValueError(
                    f"demo line too wide for the terminal frame ({ev.width_px:.0f}px > "
                    f"{MAX_LINE_WIDTH_PX:.0f}px available): {ev.text!r}"
                )
            cls = "comment" if ev.kind == "comment" else "out-line"
            style = "" if debug_static else f"animation: fadeIn 0.18s ease-out {ev.appear_time:.3f}s both;"
            font_style = ' font-style="italic"' if ev.kind == "comment" else ""
            segs = ansi_segments(ev.text, ev.color)
            tspans = "".join(
                f'<tspan fill="{c}"{_weight_attr(b)}>{_xml_text(s)}</tspan>'
                for s, c, b in segs
                if s
            ) or "&#160;"
            line_markup.append(
                f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" font-size="{FONT_SIZE}"{font_style} '
                f'style="{style}">{tspans}</text>'
            )

    if debug_static:
        final_offset = max(
            (max(0, rc - VISIBLE_ROWS) * LINE_HEIGHT for _, rc in timeline.scroll_stops), default=0
        )
        buffer_style = f"transform: translateY(-{final_offset}px);"
        mask_style = "opacity: 0;"
    else:
        buffer_style = f"animation: scrollBuf {total:.3f}s linear infinite;"
        mask_style = f"animation: maskFade {total:.3f}s linear infinite;"

    watermark = ""
    if fake_embedder:
        watermark = f"""
  <g opacity="0.32" transform="translate(450,300) rotate(-18)" pointer-events="none">
    <text text-anchor="middle" font-size="42" font-weight="700" fill="{COLOR_WARN}">{_xml_text(WATERMARK_TEXT)}</text>
  </g>
  <g pointer-events="none">
    <rect x="{TERM_X + TERM_W - 232}" y="{TERM_Y + 10}" width="222" height="22" rx="4" fill="{COLOR_WARN}" opacity="0.9"/>
    <text x="{TERM_X + TERM_W - 121}" y="{TERM_Y + 25}" text-anchor="middle" font-size="11" font-weight="700" fill="#1e1e2e">{_xml_text(WATERMARK_TEXT)}</text>
  </g>"""

    svg = f"""<svg width="900" height="560" viewBox="0 0 900 560" xmlns="http://www.w3.org/2000/svg"
     role="img" aria-label="{_xml_text(title)}">
  <title>{_xml_text(title)}</title>
  <style>
    svg {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: {COLOR_CHROME}; }}
    text {{ font-size: {FONT_SIZE}px; }}
    .comment {{ fill: {COLOR_COMMENT}; }}
    {chr(10).join(r for r in style_rules if r)}
  </style>
  <defs>
    <clipPath id="termclip"><rect x="{TERM_X}" y="{TERM_Y}" width="{TERM_W}" height="{TERM_H}" rx="4"/></clipPath>
  </defs>
  <rect width="900" height="560" rx="12" fill="{COLOR_CHROME}"/>
  <circle cx="34" cy="26" r="6" fill="#ff5f56"/>
  <circle cx="54" cy="26" r="6" fill="#ffbd2e"/>
  <circle cx="74" cy="26" r="6" fill="#27c93f"/>
  <text x="450" y="31" text-anchor="middle" fill="#9399b2" font-size="13">{_xml_text(title)}</text>
  <rect x="{TERM_X}" y="{TERM_Y}" width="{TERM_W}" height="{TERM_H}" rx="4" fill="{COLOR_BG}"/>
  <g clip-path="url(#termclip)">
    <g style="{buffer_style}">
      {chr(10).join(line_markup)}
    </g>
  </g>
  <rect x="{TERM_X}" y="{TERM_Y}" width="{TERM_W}" height="{TERM_H}" rx="4" fill="{COLOR_BG}" style="{mask_style}"/>{watermark}
</svg>
"""
    return svg


# =============================================================================
# Orchestration / CLI
# =============================================================================


@dataclass
class GenerateResult:
    cast_path: Path | None
    svg_path: Path | None
    total_duration: float
    scenes: int
    fake_embedder: bool


def generate(
    out_dir: Path | None = None,
    fake_embedder: bool = False,
    speed: float = 1.0,
    seed: int = 1337,
    cast_only: bool = False,
    svg_only: bool = False,
    debug_static: bool = False,
) -> GenerateResult:
    if out_dir is None:
        out_dir = Path(tempfile.gettempdir()) / "memstrand-demo-dry" if fake_embedder else DOCS_ASSETS
    out_dir = Path(out_dir).expanduser().resolve()

    if fake_embedder and out_dir == DOCS_ASSETS.resolve():
        raise ValueError(
            "refusing to write a --fake-embedder (DRY RUN) render into docs/assets/; "
            "pass --out-dir to write somewhere else, e.g. /tmp"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = run_session(fake_embedder=fake_embedder)
    timeline = build_timeline(scenes, seed=seed, speed=speed)

    cast_path = svg_path = None
    if not svg_only:
        cast_path = out_dir / ("demo.debug.cast" if debug_static else "demo.cast")
        cast_path.write_text(render_cast(timeline))
    if not cast_only:
        svg_path = out_dir / ("demo.debug.svg" if debug_static else "demo.svg")
        svg_path.write_text(render_svg(timeline, fake_embedder=fake_embedder, debug_static=debug_static))

    return GenerateResult(
        cast_path=cast_path,
        svg_path=svg_path,
        total_duration=timeline.total_duration,
        scenes=len(scenes),
        fake_embedder=fake_embedder,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--fake-embedder", action="store_true",
        help="offline layout dry-run: deterministic fake embedder, watermarked SVG, "
        "refuses to write into docs/assets (default output becomes a temp dir)",
    )
    p.add_argument("--out-dir", default=None, help="default: docs/assets (real) or a temp dir (--fake-embedder)")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier (2.0 = twice as fast)")
    p.add_argument("--seed", type=int, default=1337, help="deterministic RNG seed for typing jitter")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--cast-only", action="store_true", help="only write demo.cast")
    group.add_argument("--svg-only", action="store_true", help="only write demo.svg")
    p.add_argument(
        "--debug-static", action="store_true",
        help="write a non-animated, fully-revealed final-state frame for visual QA "
        "(e.g. `qlmanage -t -s 900 -o /tmp demo.debug.svg`)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else None
    try:
        result = generate(
            out_dir=out_dir,
            fake_embedder=args.fake_embedder,
            speed=args.speed,
            seed=args.seed,
            cast_only=args.cast_only,
            svg_only=args.svg_only,
            debug_static=args.debug_static,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.fake_embedder:
        print(f"DRY RUN ({WATERMARK_TEXT}) — not a real demo, not for docs/assets", file=sys.stderr)
    if result.cast_path:
        print(f"wrote {result.cast_path}")
    if result.svg_path:
        print(f"wrote {result.svg_path}")
    print(f"{result.scenes} scenes, ~{result.total_duration:.1f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
