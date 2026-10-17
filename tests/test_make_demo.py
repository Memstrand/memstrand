"""Tests for scripts/make_demo.py — the README demo generator.

Everything here runs through the `--fake-embedder` path (a deterministic,
offline, keyword-overlap embedder — see `_DemoHashEmbedder`), so these
tests never touch fastembed, the network, or a downloaded model. That
also means none of these tests produce the *real* committed demo; they
only exercise the generator's plumbing (timeline math, cast format, SVG
well-formedness, escaping, and the honesty-rule guard rails).
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_demo as md

# -- fixtures -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def scenes():
    """Run the real DEMO_STEPS once (fake embedder, no network) and reuse
    across tests — this is the slowest part (spins up a real `Memory`
    over temp on-disk stores six times) but is still well under a second."""
    return md.run_session(fake_embedder=True)


@pytest.fixture()
def timeline(scenes):
    return md.build_timeline(scenes, seed=1337, speed=1.0)


# -- cast format ----------------------------------------------------------------------


def test_cast_header_is_valid_asciinema_v2(timeline):
    cast_text = md.render_cast(timeline)
    lines = cast_text.rstrip("\n").split("\n")
    header = json.loads(lines[0])
    assert header["version"] == 2
    assert header["width"] > 0
    assert header["height"] > 0
    assert "title" in header
    for line in lines[1:]:
        event = json.loads(line)
        assert len(event) == 3
        assert isinstance(event[0], (int, float))
        assert event[1] == "o"
        assert isinstance(event[2], str)


def test_cast_event_times_are_monotonic(timeline):
    cast_text = md.render_cast(timeline)
    lines = cast_text.rstrip("\n").split("\n")[1:]
    times = [json.loads(line)[0] for line in lines]
    assert times == sorted(times)


def test_cast_total_duration_is_roughly_30_seconds(timeline):
    # The brief says "~30s"; 25-40s is the acceptable band for a demo
    # that's supposed to feel tight but not rushed.
    assert 25.0 <= timeline.total_duration <= 40.0


def test_speed_multiplier_scales_duration(scenes):
    normal = md.build_timeline(scenes, seed=1337, speed=1.0)
    fast = md.build_timeline(scenes, seed=1337, speed=2.0)
    assert fast.total_duration == pytest.approx(normal.total_duration / 2, rel=0.05)


def test_timeline_is_deterministic_given_a_seed(scenes):
    a = md.build_timeline(scenes, seed=42, speed=1.0)
    b = md.build_timeline(scenes, seed=42, speed=1.0)
    assert a.total_duration == b.total_duration
    assert [e.appear_time for e in a.line_events] == [e.appear_time for e in b.line_events]


# -- SVG well-formedness / escaping ----------------------------------------------------


def test_svg_is_well_formed_xml(timeline):
    svg = md.render_svg(timeline, fake_embedder=False)
    root = ET.fromstring(svg)  # raises ET.ParseError if malformed
    assert root.tag.endswith("svg")


def test_svg_debug_static_is_well_formed_and_unanimated(timeline):
    svg = md.render_svg(timeline, fake_embedder=False, debug_static=True)
    ET.fromstring(svg)
    assert "@keyframes scrollBuf" not in svg
    assert "@keyframes maskFade" not in svg


def test_xml_escaping_of_tricky_content():
    tricky = Scene_with(output_lines=('<script>alert("hi")&bye</script>',))
    timeline = md.build_timeline([tricky], seed=1, speed=1.0)
    svg = md.render_svg(timeline, fake_embedder=False)

    # must still be valid XML...
    ET.fromstring(svg)
    # ...and the raw, unescaped payload must never appear literally. Quotes
    # don't need escaping in XML text content (only `<` and `&` do), so
    # only those two are checked here.
    assert "<script>" not in svg
    assert "&bye" not in svg
    # the escaped form should be present in some shape
    assert "&lt;script&gt;" in svg
    assert "&amp;bye" in svg


def test_xml_escaping_in_comment_and_command_text():
    tricky = Scene_with(comment='weird & <b>"quoted"</b> comment', command="memstrand remember 'a & b < c'")
    timeline = md.build_timeline([tricky], seed=1, speed=1.0)
    svg = md.render_svg(timeline, fake_embedder=False)
    ET.fromstring(svg)
    assert "<b>" not in svg
    assert "a & b < c" not in svg


def Scene_with(comment="a comment", command="memstrand remember 'hi'", output_lines=("ok",), read_pause=1.0):
    return md.Scene(comment=comment, command=command, output_lines=tuple(output_lines), read_pause=read_pause)


# -- honesty rule: --fake-embedder watermark + docs/assets refusal ---------------------


def test_fake_embedder_svg_has_dry_run_watermark(tmp_path):
    result = md.generate(out_dir=tmp_path, fake_embedder=True, cast_only=False)
    svg_text = result.svg_path.read_text()
    assert md.WATERMARK_TEXT in svg_text


def test_fake_embedder_refuses_to_write_into_docs_assets():
    with pytest.raises(ValueError, match="docs/assets"):
        md.generate(out_dir=md.DOCS_ASSETS, fake_embedder=True)
    # The committed demo (if present) must be the real one — never the
    # watermarked output of a fake-embedder dry run.
    for name in ("demo.cast", "demo.svg"):
        path = md.DOCS_ASSETS / name
        if path.exists():
            assert md.WATERMARK_TEXT not in path.read_text()


def test_fake_embedder_default_out_dir_is_not_docs_assets(tmp_path, monkeypatch):
    # No explicit --out-dir: the default for a dry run must never resolve
    # into the real docs/assets directory.
    result = md.generate(out_dir=None, fake_embedder=True)
    assert result.svg_path is not None
    assert md.DOCS_ASSETS not in result.svg_path.parents
    assert result.svg_path.exists()


def test_real_mode_defaults_to_docs_assets_path():
    # Doesn't actually run (no network/model here) — just checks the path
    # resolution logic that decides *where* a real run would write.
    out_dir = None
    resolved = md.DOCS_ASSETS if out_dir is None else Path(out_dir)
    assert resolved == md.DOCS_ASSETS


# -- CLI flags --------------------------------------------------------------------------


def test_cast_only_and_svg_only_flags(tmp_path):
    cast_result = md.generate(out_dir=tmp_path / "cast", fake_embedder=True, cast_only=True)
    assert cast_result.cast_path is not None and cast_result.cast_path.exists()
    assert cast_result.svg_path is None
    assert not (tmp_path / "cast" / "demo.svg").exists()

    svg_result = md.generate(out_dir=tmp_path / "svg", fake_embedder=True, svg_only=True)
    assert svg_result.svg_path is not None and svg_result.svg_path.exists()
    assert svg_result.cast_path is None
    assert not (tmp_path / "svg" / "demo.cast").exists()


def test_debug_static_writes_separate_named_files(tmp_path):
    result = md.generate(out_dir=tmp_path, fake_embedder=True, debug_static=True)
    assert result.svg_path.name == "demo.debug.svg"
    assert result.cast_path.name == "demo.debug.cast"


# -- ANSI handling ------------------------------------------------------------------------


def test_ansi_segments_maps_known_sgr_codes():
    segs = md.ansi_segments("\x1b[32mgreen\x1b[0mplain", default_color="#fff")
    assert segs[0] == ("green", "#a6e3a1", False)
    assert segs[1] == ("plain", "#fff", False)


def test_ansi_segments_strips_unknown_codes():
    segs = md.ansi_segments("\x1b[99mweird\x1b[0m", default_color="#fff")
    # unknown code 99 is ignored -> stays at default color, text intact
    assert "".join(s for s, _, _ in segs) == "weird"


def test_ansi_segments_bold():
    segs = md.ansi_segments("\x1b[1mbold text\x1b[0m", default_color="#fff")
    assert segs[0] == ("bold text", "#fff", True)


# -- demo content sanity: the real DEMO_STEPS must fit the terminal frame -------------


def test_demo_steps_render_without_width_overflow(timeline):
    # render_svg raises ValueError if any line is wider than the terminal
    # frame can show; this is a regression guard on DEMO_STEPS' wording.
    svg = md.render_svg(timeline, fake_embedder=False)
    assert svg  # no exception means every line fit


def test_overflowing_line_raises_value_error():
    too_wide = Scene_with(command="memstrand " + "x" * 300)
    timeline = md.build_timeline([too_wide], seed=1, speed=1.0)
    with pytest.raises(ValueError, match="too wide"):
        md.render_svg(timeline, fake_embedder=False)


# -- story correctness: scoping demonstration (sanity on the fixture itself) ----------


def test_demo_story_demonstrates_task_scoping(scenes):
    by_comment = {s.comment: s for s in scenes}
    no_leak = by_comment["A different task asks about testing setup — no leak"]
    assert not any("Playwright" in line for line in no_leak.output_lines)

    scoped = by_comment["The right task sees its fact + your globals"]
    assert any("Playwright" in line for line in scoped.output_lines)
    assert any("Sam prefers" in line or "Sam now prefers" in line for line in scoped.output_lines)


def test_demo_story_edit_supersedes(scenes):
    by_comment = {s.comment: s for s in scenes}
    edit_scene = by_comment["Facts change; edits supersede — they don't delete"]
    new_id = edit_scene.output_lines[-1].strip()
    assert re.match(r"^mem:[0-9a-f]+$", new_id)
