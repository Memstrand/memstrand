"""Second-brain ingestion: pull a folder of notes (Markdown / Obsidian vault,
plain text, PDF) into memory, re-ingesting only changed files.

Change detection lives in a small JSON state file (sha256 per absolute
path). Markdown gets heading-aware chunking with frontmatter `tags:` and
Obsidian `[[wikilink]]` resolution into `related_to` graph edges; plain
text and PDF get paragraph-based chunking.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path

import yaml

from .schema import Category, MemoryEdge, Provenance, RelationType

TEXT_EXTS = {".md", ".markdown", ".txt", ".rst", ".org"}
MARKDOWN_EXTS = {".md", ".markdown"}
PDF_EXTS = {".pdf"}
SKIP_DIRS = {".git", ".obsidian", "node_modules", "__pycache__", ".venv"}
MAX_FILE_BYTES = 5 * 1024 * 1024
MIN_SECTION_CHARS = 40

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")


class _MissingDependency(Exception):
    """A file needs an optional dependency we don't have."""


# -- pure, independently testable helpers ------------------------------------


def extract_wikilinks(text: str) -> set[str]:
    """Collect `[[Note]]` / `[[Note|alias]]` / `[[Note#Heading]]` targets."""
    return {m.group(1).strip() for m in _WIKILINK_RE.finditer(text) if m.group(1).strip()}


def _parse_frontmatter(text: str) -> tuple[str, list[str]]:
    """Strip a leading YAML frontmatter block, return (remaining, tags)."""
    if not text.startswith("---"):
        return text, []
    lines = text.splitlines()
    if lines[0].strip() != "---":
        return text, []
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return text, []
    fm_text = "\n".join(lines[1:end])
    rest = "\n".join(lines[end + 1 :])
    tags: list[str] = []
    try:
        data = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        data = {}
    if isinstance(data, dict):
        raw_tags = data.get("tags")
        if isinstance(raw_tags, list):
            tags = [str(t) for t in raw_tags]
        elif isinstance(raw_tags, str):
            tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
    return rest, tags


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown body into (heading_path, content) by heading, tracking depth."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    current_path = ""
    buf: list[str] = []

    def flush() -> None:
        content = "\n".join(buf).strip()
        if content:
            sections.append((current_path, content))
        buf.clear()

    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current_path = " > ".join(t for _, t in stack)
        else:
            buf.append(line)
    flush()
    return sections


def _merge_tiny_sections(
    sections: list[tuple[str, str]], min_chars: int = MIN_SECTION_CHARS
) -> list[tuple[str, str]]:
    """Fold sections shorter than `min_chars` into the following section."""
    n = len(sections)
    merged: list[tuple[str, str]] = []
    pending_text: str | None = None
    for idx, (path, text) in enumerate(sections):
        if pending_text is not None:
            text = f"{pending_text}\n\n{text}"
            pending_text = None
        is_last = idx == n - 1
        if len(text) < min_chars and not is_last:
            pending_text = text
            continue
        merged.append((path, text))
    if pending_text is not None:
        if merged:
            prev_path, prev_text = merged[-1]
            merged[-1] = (prev_path, f"{prev_text}\n\n{pending_text}")
        else:
            merged.append(("", pending_text))
    return merged


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split on paragraph boundaries to <= max_chars, carrying `overlap` chars forward."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paras:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{para}" if tail else para
        else:
            # a single paragraph longer than max_chars: hard split
            step = max(max_chars - overlap, 1)
            for i in range(0, len(para), step):
                piece = para[i : i + max_chars]
                if piece.strip():
                    chunks.append(piece)
            current = ""
    if current.strip():
        chunks.append(current)
    return chunks


def chunk_markdown(
    text: str, max_chars: int = 1200, overlap: int = 150
) -> list[tuple[str, str]]:
    """Heading-aware markdown chunking. Returns (heading_path, chunk_text) pairs.

    Frontmatter is stripped; a `tags:` list (if present) is prepended as
    "Tags: a, b" to every chunk produced from this text.
    """
    body, tags = _parse_frontmatter(text)
    sections = _merge_tiny_sections(_split_sections(body))
    tag_prefix = f"Tags: {', '.join(tags)}\n\n" if tags else ""

    chunks: list[tuple[str, str]] = []
    for heading_path, section_text in sections:
        for piece in _split_long(section_text, max_chars, overlap):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append((heading_path, tag_prefix + piece))
    return chunks


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Paragraph-based chunking for plain text / PDF page text."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return []
    joined = "\n\n".join(paras)
    return [c for c in _split_long(joined, max_chars, overlap) if c.strip()]


# -- filesystem walking / state -----------------------------------------------


def _iter_files(root: Path):
    if root.is_file():
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS)
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            yield Path(dirpath) / name


def _state_key(task_id: str | None, abs_path: str) -> str:
    """Change-detection state is keyed by (task_id, abs_path), not path
    alone — otherwise re-ingesting the same path into a different task_id
    is a silent no-op against the first task's cached hash, and the
    second task never gets its atoms."""
    return f"{task_id or ''}\x1f{abs_path}"


def _decode_state_key(key: str) -> tuple[str | None, str]:
    """Decode a state key into (task_id, abs_path). Legacy (pre-scope)
    state files used the bare abs_path as the key with no separator —
    those are migrated in `_load_state` to `task_id=None` entries, but
    this also tolerates an unmigrated key defensively."""
    if "\x1f" not in key:
        return None, key
    task_part, abs_path = key.split("\x1f", 1)
    return (task_part or None), abs_path


def _migrate_legacy_keys(state: dict) -> None:
    """Pre-scope state files keyed `files` by bare abs_path. Rewrite those
    entries to the new `task_id\x1fabs_path` form (task_id=None, i.e. the
    entries behave exactly as before for the global/no-task ingest case)
    so lookups against the new key format still find them instead of
    treating every previously-ingested file as new."""
    files = state.get("files")
    if not isinstance(files, dict):
        return
    for key in list(files.keys()):
        if "\x1f" in key:
            continue
        new_key = _state_key(None, key)
        if new_key not in files:
            files[new_key] = files[key]
        del files[key]


def _load_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {"files": {}}
    else:
        state = {"files": {}}
    state.setdefault("files", {})
    _migrate_legacy_keys(state)
    return state


def _save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, state_path)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -- per-file ingestion --------------------------------------------------------


def _ingest_markdown_or_text(
    mem,
    abs_path: str,
    rel: str,
    text: str,
    *,
    markdown: bool,
    task_id: str | None,
    category: Category,
) -> int:
    if markdown:
        pieces = chunk_markdown(text)
    else:
        pieces = [("", c) for c in chunk_text(text)]

    stem = Path(abs_path).stem
    for i, (heading_path, chunk) in enumerate(pieces):
        content = f"{stem} — {heading_path}\n\n{chunk}"
        source_ref = f"{rel}#{heading_path} [{i}]"
        mem.remember(
            content,
            category=category,
            provenance=Provenance.imported,
            confidence=0.9,
            task_id=task_id,
            source=abs_path,
            source_ref=source_ref,
        )
    return len(pieces)


def _ingest_pdf(
    mem,
    abs_path: str,
    rel: str,
    raw: bytes,
    *,
    task_id: str | None,
    category: Category,
) -> int:
    try:
        import pypdf
    except ImportError as exc:
        raise _MissingDependency("install memstrand[pdf]") from exc

    import io

    reader = pypdf.PdfReader(io.BytesIO(raw))
    stem = Path(abs_path).stem
    count = 0
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for i, chunk in enumerate(chunk_text(page_text)):
            content = f"{stem} — page {page_num}\n\n{chunk}"
            source_ref = f"{rel}#page {page_num} [{i}]"
            mem.remember(
                content,
                category=category,
                provenance=Provenance.imported,
                confidence=0.9,
                task_id=task_id,
                source=abs_path,
                source_ref=source_ref,
            )
            count += 1
    return count


def _first_chunk_id(mem, abs_path: str) -> str | None:
    ids = mem.graph.ids_by_source(abs_path)
    if not ids:
        return None
    atoms = mem.graph.get_atoms(ids)
    for atom_id, atom in atoms.items():
        ref = getattr(atom, "source_ref", "") or ""
        if ref.endswith("[0]"):
            return atom_id
    return None


def _rebuild_wikilink_edges(
    mem,
    task_id: str | None,
    files_state: dict,
    changed_abs_paths: set[str],
    links_by_file: dict[str, set[str]],
    stem_by_file: dict[str, str],
) -> int:
    """Add related_to wikilink edges between first chunks of linked notes.

    A naive "only look at outgoing links from files scanned this run"
    approach loses edges on a partial re-ingest: `ingest_path` can be
    called with `path` pointing at a single changed file (e.g. the file
    watcher), in which case `links_by_file`/`stem_by_file` only contain
    that one file. `forget_source` already wiped its old atoms (and, via
    cascading edge deletes, every edge that pointed at them) — so any
    *incoming* wikilink from a file outside this run's scan would simply
    never come back.

    Fix: resolve targets using both this run's scan AND the persisted
    per-file state (which records every previously-seen file's stem and
    outgoing links), and rebuild edges for every file changed this run
    PLUS every other known file in the same task scope whose recorded
    outgoing links point at one of those changed files.
    """
    target_lookup: dict[str, str] = dict(stem_by_file)
    for key, info in files_state.items():
        entry_task_id, abs_path = _decode_state_key(key)
        if entry_task_id != task_id:
            continue
        stem = info.get("stem") or Path(abs_path).stem.lower()
        target_lookup.setdefault(stem, abs_path)

    changed_stems = {stem_by_file.get(p) or Path(p).stem.lower() for p in changed_abs_paths}

    files_to_relink: dict[str, set[str]] = {
        p: links_by_file.get(p, set()) for p in changed_abs_paths
    }
    for key, info in files_state.items():
        entry_task_id, abs_path = _decode_state_key(key)
        if entry_task_id != task_id or abs_path in files_to_relink:
            continue
        stored_links = info.get("links") or []
        if any(Path(t).stem.lower() in changed_stems for t in stored_links):
            files_to_relink[abs_path] = set(stored_links)

    links_added = 0
    for abs_path, targets in files_to_relink.items():
        if not targets:
            continue
        source_first = _first_chunk_id(mem, abs_path)
        if source_first is None:
            continue
        for target_name in targets:
            target_stem = Path(target_name).stem.lower()
            target_abs = target_lookup.get(target_stem)
            if not target_abs or target_abs == abs_path:
                continue
            target_first = _first_chunk_id(mem, target_abs)
            if target_first is None:
                continue
            mem.graph.add_edge(
                MemoryEdge(
                    source_id=source_first,
                    target_id=target_first,
                    relation=RelationType.related_to,
                    reason="wikilink",
                )
            )
            links_added += 1
    return links_added


# -- public entry points -------------------------------------------------------


def ingest_path(
    mem,
    path: str | os.PathLike,
    task_id: str | None = None,
    *,
    category: Category = Category.fact,
    state_path: Path | None = None,
    force: bool = False,
) -> dict:
    """Ingest a file or directory into `mem`. Returns a report dict.

    Unchanged files (by sha256) are skipped unless `force`. Changed files
    are re-ingested (`forget_source` then re-added). Files that vanished
    since the last run are forgotten too. One bad file never aborts the run.
    """
    root = Path(path).resolve()
    root_dir = root if root.is_dir() else root.parent
    state_path = Path(state_path) if state_path else mem.home / "ingest_state.json"
    state = _load_state(state_path)
    files_state = state.setdefault("files", {})

    report: dict = {
        "root": str(root),
        "files_seen": 0,
        "files_ingested": 0,
        "files_unchanged": 0,
        "chunks": 0,
        "links": 0,
        "removed": [],
        "skipped": [],
        "errors": [],
    }

    links_by_file: dict[str, set[str]] = {}
    stem_by_file: dict[str, str] = {}
    seen_abs: set[str] = set()
    changed_abs_paths: set[str] = set()

    for file_path in _iter_files(root):
        report["files_seen"] += 1
        abs_path = str(file_path.resolve())
        seen_abs.add(abs_path)
        suffix = file_path.suffix.lower()

        try:
            size = file_path.stat().st_size
        except OSError as exc:
            report["errors"].append({"path": abs_path, "error": str(exc)})
            continue

        if size > MAX_FILE_BYTES:
            report["skipped"].append({"path": abs_path, "reason": "file too large (> 5MB)"})
            continue

        if suffix not in TEXT_EXTS and suffix not in PDF_EXTS:
            report["skipped"].append({"path": abs_path, "reason": "unsupported file type"})
            continue

        try:
            raw = file_path.read_bytes()
        except OSError as exc:
            report["errors"].append({"path": abs_path, "error": str(exc)})
            continue

        stem_by_file[abs_path] = file_path.stem.lower()
        if suffix in MARKDOWN_EXTS:
            try:
                links_by_file[abs_path] = extract_wikilinks(raw.decode("utf-8", errors="replace"))
            except Exception:  # noqa: BLE001 - link extraction must never abort ingestion
                links_by_file[abs_path] = set()
        else:
            links_by_file[abs_path] = set()

        digest = _sha256(raw)
        state_key = _state_key(task_id, abs_path)
        prior = files_state.get(state_key)
        unchanged = bool(prior) and prior.get("hash") == digest
        if unchanged and not force:
            report["files_unchanged"] += 1
            continue

        rel = os.path.relpath(abs_path, root_dir)
        try:
            if prior:
                # Scoped to this exact task: re-ingesting a changed file
                # must never wipe another task's chunks of the same source.
                mem.forget_source(abs_path, task_id=task_id)
            if suffix in PDF_EXTS:
                n_chunks = _ingest_pdf(mem, abs_path, rel, raw, task_id=task_id, category=category)
            else:
                text = raw.decode("utf-8", errors="replace")
                n_chunks = _ingest_markdown_or_text(
                    mem,
                    abs_path,
                    rel,
                    text,
                    markdown=suffix in MARKDOWN_EXTS,
                    task_id=task_id,
                    category=category,
                )
        except _MissingDependency as exc:
            report["skipped"].append({"path": abs_path, "reason": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001 - one bad file must never abort the run
            report["errors"].append({"path": abs_path, "error": str(exc)})
            continue

        report["files_ingested"] += 1
        report["chunks"] += n_chunks
        changed_abs_paths.add(abs_path)
        files_state[state_key] = {
            "hash": digest,
            "root": str(root),
            "stem": stem_by_file.get(abs_path, Path(abs_path).stem.lower()),
            "links": sorted(links_by_file.get(abs_path, set())),
        }

    for key, info in list(files_state.items()):
        entry_task_id, abs_path = _decode_state_key(key)
        if (
            info.get("root") == str(root)
            and abs_path not in seen_abs
            and not os.path.exists(abs_path)
        ):
            mem.forget_source(abs_path, task_id=entry_task_id)
            del files_state[key]
            report["removed"].append(abs_path)

    report["links"] = _rebuild_wikilink_edges(
        mem, task_id, files_state, changed_abs_paths, links_by_file, stem_by_file
    )

    _save_state(state_path, state)
    return report


def watch_path(
    mem,
    path: str | os.PathLike,
    task_id: str | None = None,
    *,
    interval: float = 5.0,
    stop_event: threading.Event | None = None,
    on_report=None,
) -> None:
    """Poll `path` every `interval` seconds, calling `on_report` when anything changed.

    Exits cleanly when `stop_event` is set, or on KeyboardInterrupt.
    """
    stop_event = stop_event or threading.Event()
    try:
        while not stop_event.is_set():
            report = ingest_path(mem, path, task_id=task_id)
            changed = report["files_ingested"] > 0 or report["removed"]
            if changed and on_report:
                on_report(report)
            stop_event.wait(interval)
    except KeyboardInterrupt:
        return
