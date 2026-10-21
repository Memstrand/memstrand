"""Neutralize stored memory text before it's shown to a model.

Lives apart from `api.py` (which pulls in qdrant/fastembed/onnxruntime) so
the Claude Code hooks can use it on every prompt without that import cost.
`api.py` re-exports `_sanitize_memory_text` for existing callers.
"""

from __future__ import annotations

import re
import unicodedata

_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s*")
_ROLE_PREFIX_RE = re.compile(r"^(system|assistant|user|tool)\s*:", re.IGNORECASE)

# Variation selectors (VS1-16 and the IVS/mongolian supplement block):
# invisible codepoints that only ever modify how the PRECEDING character
# renders (e.g. U+FE0F forces emoji-style presentation) -- they carry no
# text of their own. A role word with one spliced in the middle
# ("system︎:") still READS as "system:" to a human but no longer
# matches `_ROLE_PREFIX_RE` literally. Safe to drop unconditionally from
# the real display text (not just a detection copy): removing one never
# changes what text is visibly present, only a rendering hint on the
# character right before it (e.g. "❤️" keeps its heart character,
# "coffee" keeps every letter).
_VARIATION_SELECTOR_RE = re.compile("[︀-️\U000e0100-\U000e01ef]")

# Cyrillic/Greek characters that are visually indistinguishable (in most
# fonts) from a Latin letter used in "system"/"assistant"/"user"/"tool" --
# a known homoglyph trick (the same one IDN/domain spoofing uses) that
# would otherwise defeat `_ROLE_PREFIX_RE` entirely, e.g. "ѕystem:" (with
# a Cyrillic "ѕ", U+0455) reads as "system:" to a human but doesn't match
# an ASCII-only regex. Used ONLY to build a translated copy of the text for
# the role-prefix *check* below -- the memory's actual displayed content is
# never rewritten by this map, only a throwaway detection copy (see
# `_detection_skeleton`) used to decide WHETHER a prefix is present.
_CONFUSABLES_TO_LATIN = {
    # Cyrillic
    "а": "a", "А": "A",
    "е": "e", "Е": "E",
    "і": "i", "І": "I",
    "о": "o", "О": "O",
    "р": "p", "Р": "P",
    "с": "c", "С": "C",
    "ѕ": "s", "Ѕ": "S",
    "т": "t", "Т": "T",
    "у": "y", "У": "Y",
    "х": "x", "Х": "X",
    "ј": "j", "Ј": "J",
    # Greek
    "α": "a", "Α": "A",
    "ε": "e", "Ε": "E",
    "ι": "i", "Ι": "I",
    "κ": "k", "Κ": "K",
    "ο": "o", "Ο": "O",
    "ρ": "p", "Ρ": "P",
    "τ": "t", "Τ": "T",
    "υ": "u", "Υ": "Y",
    "χ": "x", "Χ": "X",
}


def _to_latin_confusables(text: str) -> str:
    return "".join(_CONFUSABLES_TO_LATIN.get(ch, ch) for ch in text)


def _detection_skeleton(text: str) -> str:
    """Build a THROWAWAY copy of `text` used ONLY to DETECT a disguised
    leading role prefix or markdown heading -- never returned, stored, or
    displayed anywhere.

    NFKD-decomposes first, which splits a precomposed accented letter back
    into its base letter plus a combining mark -- e.g. "ś" (U+015B, a
    single codepoint) becomes "s" + U+0301 (combining acute). Dropping
    Unicode category "Mn" (combining marks, plus "Cf"/format chars again,
    harmless belt-and-suspenders) and variation selectors afterward then
    lets "śystem:" read as "system:" for detection, the same way the
    confusables map below lets a homoglyph ("ѕystem:", Cyrillic "ѕ") read
    as "system:". Casefolds last so detection is case-insensitive.

    Deliberately asymmetric from the real sanitized output: legitimate
    accented content ("café", "naïve") must never have its accents
    stripped in what's actually stored/displayed -- only this disposable
    skeleton drops them, purely to decide WHETHER a prefix is present.
    What actually gets REMOVED is always sliced out of the real,
    accent-preserving text (see `_strip_leading_structure`), never out of
    this skeleton."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(
        ch
        for ch in decomposed
        if unicodedata.category(ch) not in ("Mn", "Cf") and not _VARIATION_SELECTOR_RE.match(ch)
    )
    return _to_latin_confusables(stripped).casefold()


def _strip_leading_structure(text: str) -> str:
    """One pass of stripping a leading markdown heading or disguised role
    prefix from `text` -- always the real, accent/content-preserving
    display string, never the `_detection_skeleton` used to decide
    whether one is present.

    Detection goes through the skeleton (so a combining-mark-disguised
    role word like "śystem:" or a homoglyph one like "ѕystem:" is still
    recognized), but the actual removal is computed directly against the
    real text: the leading '#'-run for a heading, or "everything up to
    and including the first ':'" for a role prefix -- never a slice
    length borrowed from the differently-sized (NFKD-expanded) skeleton,
    which would misalign against the real string's indices."""
    skeleton = _detection_skeleton(text)
    if _MARKDOWN_HEADING_RE.match(skeleton):
        return _MARKDOWN_HEADING_RE.sub("", text, count=1).strip()
    if _ROLE_PREFIX_RE.match(skeleton):
        colon_idx = text.find(":")
        if colon_idx != -1:
            return text[colon_idx + 1 :].strip()
    return text


def _sanitize_memory_text(text: str) -> str:
    """Neutralize markdown structure a memory's own CONTENT might use to
    impersonate part of the `context()` block's structure (a heading, a
    fake "system:"/"assistant:" turn, ...) — relevant because atom content
    can come from an untrusted import, not just from a trusted `remember()`
    call.

    Before any of that: NFKC-normalize (collapses fullwidth/compatibility
    forms like the fullwidth '＃' down to plain '#', so they can't dodge
    `_MARKDOWN_HEADING_RE`) and drop every Unicode category "Cf" (format)
    character plus U+FEFF (BOM / zero-width no-break space; Cf already
    covers it, but it's called out explicitly since this is a security
    boundary) and every variation selector (invisible rendering hints,
    e.g. "system︎:" with a stray VS15 — removing one never deletes any
    visible text, only a presentation hint on the character before it) --
    zero-width/invisible characters such as U+200B (ZWSP) would otherwise
    sit silently inside a heading/role marker and defeat the regexes below
    without being visible anywhere.

    Then collapses embedded newlines to spaces (so a multi-line "fake
    block" can't masquerade as several separate headings/list items once
    rendered), and strips any leading markdown heading marker or
    role-like prefix -- including homoglyph (e.g. Cyrillic "ѕystem:") and
    combining-mark-disguised (e.g. "śystem:") role words, detected via
    `_detection_skeleton`/`_strip_leading_structure` without ever
    stripping accents from legitimate content like "café"/"naïve" in the
    actual output -- repeating a few times in case of nesting (e.g.
    "### assistant: ...", "### ### system: system: ...")."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = "".join(
        ch
        for ch in normalized
        if ch != "﻿" and unicodedata.category(ch) != "Cf" and not _VARIATION_SELECTOR_RE.match(ch)
    )
    collapsed = re.sub(r"\s+", " ", normalized).strip()
    for _ in range(5):
        stripped = _strip_leading_structure(collapsed)
        if stripped == collapsed:
            break
        collapsed = stripped
    return collapsed
