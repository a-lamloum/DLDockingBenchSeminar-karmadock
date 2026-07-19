#!/usr/bin/env python3
"""Build the offline KarmaDock seminar course website using only the stdlib."""

from __future__ import annotations

import csv
import hashlib
import html
from html.parser import HTMLParser
import io
import json
import keyword
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import token
import tokenize
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import unquote, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = OUTPUT_ROOT / "assets"

CORE_PAGES = [
    "index.html",
    "pipelines.html",
    "results.html",
    "reproduce.html",
    "training.html",
    "reference.html",
]

CORE_STEMS = [Path(name).stem for name in CORE_PAGES]
ARABIC_WALKTHROUGH_STEMS = [
    "walkthrough-condor",
    "walkthrough-train",
    "walkthrough-convert-karmadock-to-seminar",
    "walkthrough-convert-seminar-to-karmadock",
    "walkthrough-run-full-stage2-ddp",
    "walkthrough-run-infer",
    "walkthrough-seminar-csv",
    "walkthroughs",
    "walkthrough-evaluation",
    "walkthrough-evaluate",
    "walkthrough-run-full-stage2-ddp-v2",
    "walkthrough-run-full-train",
    "walkthrough-run-train",
    "walkthrough-train-ddp",
]
ARABIC_PAGE_STEMS = set(CORE_STEMS + ARABIC_WALKTHROUGH_STEMS)

WALKTHROUGH_FILES = sorted((REPO_ROOT / "docs/code-walkthrough").glob("*.md"))


def walkthrough_output(path: Path) -> str:
    if path.name == "README.md":
        return "walkthroughs.html"
    if path.name == "condor.md":
        return "walkthrough-condor.html"
    if path.name == "evaluation.evaluation.md":
        return "walkthrough-evaluation.html"
    name = path.stem
    if name.startswith("scripts."):
        name = name[len("scripts.") :]
    return "walkthrough-" + name.replace("_", "-").replace(".", "-") + ".html"


WALKTHROUGH_OUTPUTS = [walkthrough_output(path) for path in WALKTHROUGH_FILES]
ENGLISH_HTML_MANIFEST = CORE_PAGES + WALKTHROUGH_OUTPUTS


def localized_filename(stem: str, lang: str) -> str:
    return f"{stem}.ar.html" if lang == "ar" else f"{stem}.html"


ARABIC_HTML_MANIFEST = [
    localized_filename(stem, "ar")
    for stem in CORE_STEMS + ARABIC_WALKTHROUGH_STEMS
]
HTML_MANIFEST = ENGLISH_HTML_MANIFEST + ARABIC_HTML_MANIFEST

ARABIC_CONTENT_DIR = OUTPUT_ROOT / "content/i18n/ar"
QUIZ_CONTENT_DIR = OUTPUT_ROOT / "content/quizzes"


@dataclass(frozen=True)
class ContentSegment:
    source_path: Path
    link_base: Path
    markdown: str


@dataclass
class Block:
    kind: str
    source: Path
    text: str = ""
    level: int = 0
    language: str = ""
    rows: list[list[str]] = field(default_factory=list)
    aligns: list[str | None] = field(default_factory=list)
    items: list[str] = field(default_factory=list)
    ordered: bool = False
    start: int = 1
    children: list["Block"] = field(default_factory=list)
    attrs: dict[str, str] = field(default_factory=dict)
    heading_id: str = ""
    explicit_id: str = ""
    link_base: Path | None = None


@dataclass(frozen=True)
class CodeSpan:
    start: int
    end: int
    value: str


@dataclass
class Page:
    stem: str
    lang: str
    direction: str
    filename: str
    title: str
    nav_label: str
    counterpart: str | None
    canonical_source: Path
    segments: list[ContentSegment]
    blocks: list[Block] = field(default_factory=list)
    quizzes: dict[str, "QuizBlock"] = field(default_factory=dict)
    status: str = ""


@dataclass
class QuizQuestion:
    qid: str
    question: list[Block]
    answer: list[Block]


@dataclass
class QuizBlock:
    section_key: str
    questions: list[QuizQuestion]
    source: Path


@dataclass(frozen=True)
class LinkResolution:
    href: str | None
    external: bool = False
    annotation: str = ""


def relpath(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")


EN_UI = {
    "skip_link": "Skip to content",
    "menu_label": "Toggle course navigation",
    "site_subtitle": "Course Site",
    "search_button": "Search",
    "open_search": "Open search",
    "theme_light": "Switch to light theme",
    "theme_dark": "Switch to dark theme",
    "nav_course": "Course",
    "nav_walkthroughs": "Code walkthroughs",
    "nav_reference": "Reference",
    "nav_label": "Course pages",
    "on_this_page": "On this page",
    "no_subsections": "No subsections",
    "footer": "KarmaDock seminar project · Saarland University · Offline course companion",
    "search_title": "Search the course site",
    "close_search": "Close search",
    "search_label": "Search titles, headings, and explanatory text",
    "search_placeholder": "Try “pos_r” or “PoseBusters”",
    "search_empty": "Type to search across the course site.",
    "search_none": "No matching sections.",
    "result_singular": "result",
    "result_plural": "results",
    "external_link": "external link",
    "quiz_title": "Check your understanding",
    "show_answer": "Show answer",
    "not_bundled": "available in the repository, not bundled into the website",
    "draft_banner": "",
    "language_label": "Arabic",
    "language_aria": "Switch to Arabic",
    "page_labels": {
        "index": "Overview",
        "pipelines": "Pipelines",
        "results": "Results",
        "reproduce": "Reproduce",
        "training": "Training",
        "reference": "Reference",
    },
}


def load_json_object(path: Path) -> dict:
    value = json.loads(read_text(path))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_arabic_ui() -> dict:
    path = ARABIC_CONTENT_DIR / "ui.json"
    ui = load_json_object(path)
    required = set(EN_UI)
    if set(ui) != required or set(ui.get("page_labels", {})) != set(CORE_STEMS):
        raise ValueError("Arabic UI keys must exactly match the English UI contract")
    return ui


def load_arabic_statuses() -> dict[str, str]:
    path = ARABIC_CONTENT_DIR / "status.json"
    statuses = load_json_object(path)
    if set(statuses) != ARABIC_PAGE_STEMS or any(value not in {"draft", "reviewed"} for value in statuses.values()):
        raise ValueError("Arabic status.json must exactly cover the available Arabic pages with draft/reviewed")
    return statuses


def markdown_plain(text: str) -> str:
    text = re.sub(r"`+([^`]*)`+", r"\1", text)
    text = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
    text = text.replace("**", "").replace("__", "").replace("*", "").replace("_", "")
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


def slugify(text: str) -> str:
    """Close match for GitHub heading slugs, retaining repeated whitespace hyphens."""
    plain = markdown_plain(text).casefold()
    out: list[str] = []
    for char in plain:
        category = unicodedata.category(char)
        if char.isspace():
            out.append("-")
        elif char in "-_":
            out.append(char)
        elif category[0] in {"L", "N"}:
            out.append(char)
    return "".join(out).strip("-") or "section"


def normalize_search(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch) and ch != "ـ")
    stripped = stripped.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا"}))
    return " ".join(stripped.split())


NORMALIZATION_FIXTURE_INPUTS = [
    "التَّدريبُ",
    "كـارمـا دوك",
    "أحمد إبراهيم آمنة",
    "RMSD MDN 82.2 %",
]


def normalization_fixtures() -> list[dict[str, str]]:
    return [{"input": value, "expected": normalize_search(value)} for value in NORMALIZATION_FIXTURE_INPUTS]


FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,})([^`]*)$")
HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
HR_RE = re.compile(r"^ {0,3}(?:-{3,}|\*{3,}|_{3,})\s*$")
LIST_RE = re.compile(r"^( {0,3})([-+*]|(\d+)[.)])\s+(.+)$")
RAW_IMAGE_RE = re.compile(
    r'^<a\s+href="([^"]+)"><img\s+src="([^"]+)"\s+alt="([^"]*)"\s+width="(\d+)"></a>$'
)


def split_table_row(line: str) -> list[str]:
    value = line.strip()
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|") and not value.endswith(r"\|"):
        value = value[:-1]
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    tick_run = 0
    i = 0
    while i < len(value):
        char = value[i]
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
            current.append(char)
        elif char == "`":
            run = 1
            while i + run < len(value) and value[i + run] == "`":
                run += 1
            if tick_run == 0:
                tick_run = run
            elif tick_run == run:
                tick_run = 0
            current.extend("`" * run)
            i += run - 1
        elif char == "|" and tick_run == 0:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        i += 1
    if escaped:
        current.append("\\")
    cells.append("".join(current).strip())
    return cells


def table_delimiter(line: str) -> list[str | None] | None:
    cells = split_table_row(line)
    if not cells:
        return None
    aligns: list[str | None] = []
    for cell in cells:
        compact = cell.replace(" ", "")
        if not re.fullmatch(r":?-{3,}:?", compact):
            return None
        if compact.startswith(":") and compact.endswith(":"):
            aligns.append("center")
        elif compact.startswith(":"):
            aligns.append("left")
        elif compact.endswith(":"):
            aligns.append("right")
        else:
            aligns.append(None)
    return aligns


def is_table_start(lines: list[str], index: int) -> bool:
    return index + 1 < len(lines) and "|" in lines[index] and table_delimiter(lines[index + 1]) is not None


def is_block_opener(lines: list[str], index: int) -> bool:
    line = lines[index]
    stripped = line.strip()
    return bool(
        not stripped
        or FENCE_OPEN_RE.match(line)
        or RAW_IMAGE_RE.match(stripped)
        or HEADING_RE.match(line)
        or HR_RE.match(line)
        or is_table_start(lines, index)
        or re.match(r"^ {0,3}>", line)
        or LIST_RE.match(line)
    )


EXPLICIT_ID_RE = re.compile(r"\s+\{#([a-z0-9][a-z0-9-]*)\}\s*$")


def parse_blocks(markdown: str, source: Path, link_base: Path | None = None) -> list[Block]:
    link_base = link_base or source.parent
    lines = markdown.split("\n")
    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        fence = FENCE_OPEN_RE.match(line)
        if fence:
            marker = fence.group(1)
            language = fence.group(2).strip().split(maxsplit=1)[0] if fence.group(2).strip() else ""
            close_re = re.compile(rf"^ {{0,3}}`{{{len(marker)},}}\s*$")
            raw: list[str] = []
            i += 1
            while i < len(lines) and not close_re.match(lines[i]):
                raw.append(lines[i])
                i += 1
            if i >= len(lines):
                raise ValueError(f"Unclosed code fence in {relpath(source)}")
            blocks.append(Block("code", source, text="\n".join(raw), language=language.lower(), link_base=link_base))
            i += 1
            continue

        raw_image = RAW_IMAGE_RE.match(line.strip())
        if raw_image:
            href, src, alt, width = raw_image.groups()
            if href != src or not src.startswith("docs/workflow_") or not src.endswith(".png"):
                raise ValueError(f"Unsafe raw image wrapper in {relpath(source)}:{i + 1}")
            image_path = (link_base / src).resolve()
            if not image_path.is_relative_to(REPO_ROOT) or not image_path.is_file():
                raise ValueError(f"Missing raw image {src}")
            blocks.append(Block("raw_image", source, attrs={"src": src, "alt": alt, "width": width}, link_base=link_base))
            i += 1
            continue

        heading = HEADING_RE.match(line)
        if heading:
            heading_text = heading.group(2)
            explicit = EXPLICIT_ID_RE.search(heading_text)
            explicit_id = explicit.group(1) if explicit else ""
            if explicit:
                heading_text = heading_text[: explicit.start()].rstrip()
            blocks.append(
                Block(
                    "heading",
                    source,
                    text=heading_text,
                    level=len(heading.group(1)),
                    explicit_id=explicit_id,
                    link_base=link_base,
                )
            )
            i += 1
            continue

        if HR_RE.match(line):
            blocks.append(Block("hr", source, link_base=link_base))
            i += 1
            continue

        if is_table_start(lines, i):
            header = split_table_row(lines[i])
            aligns = table_delimiter(lines[i + 1]) or []
            rows = [header]
            i += 2
            while i < len(lines) and lines[i].strip() and "|" in lines[i]:
                row = split_table_row(lines[i])
                if len(row) != len(header):
                    break
                rows.append(row)
                i += 1
            blocks.append(Block("table", source, rows=rows, aligns=aligns, link_base=link_base))
            continue

        if re.match(r"^ {0,3}>", line):
            quote_lines: list[str] = []
            while i < len(lines) and re.match(r"^ {0,3}>", lines[i]):
                quote_lines.append(re.sub(r"^ {0,3}> ?", "", lines[i]))
                i += 1
            blocks.append(Block("blockquote", source, children=parse_blocks("\n".join(quote_lines), source, link_base), link_base=link_base))
            continue

        list_match = LIST_RE.match(line)
        if list_match:
            ordered = bool(list_match.group(3))
            start = int(list_match.group(3) or 1)
            items: list[str] = []
            while i < len(lines):
                item_match = LIST_RE.match(lines[i])
                if not item_match or bool(item_match.group(3)) != ordered:
                    break
                items.append(item_match.group(4).strip())
                i += 1
            blocks.append(Block("list", source, items=items, ordered=ordered, start=start, link_base=link_base))
            continue

        paragraph = [line.strip()]
        i += 1
        while i < len(lines) and not is_block_opener(lines, i):
            paragraph.append(lines[i].strip())
            i += 1
        blocks.append(Block("paragraph", source, text=" ".join(part for part in paragraph if part), link_base=link_base))
    return blocks


def split_h2_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    lines = text.splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith("## ")]
    if not starts:
        return text, []
    preamble = "\n".join(lines[: starts[0]]).strip() + "\n"
    sections: list[tuple[str, str]] = []
    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(lines)
        chunk = "\n".join(lines[start:end]).strip() + "\n"
        sections.append((lines[start][3:].strip(), chunk))
    return preamble, sections


def promote_first_heading(markdown: str) -> str:
    return re.sub(r"^## ", "# ", markdown, count=1)


def training_summary_markdown(paths: list[Path]) -> str:
    labels = {
        "full_stage2_train_log.csv": "Full-data Stage 2",
        "p2_stage1_train_log.csv": "P2 Stage 1",
        "p2_stage2_train_log.csv": "P2 Stage 2",
        "p3_finetune_train_log.csv": "P3 fine-tune",
    }
    lines = [
        "## Training-log summary",
        "",
        "Computed directly from the checked-in per-epoch logs. Best values are minima.",
        "",
        "| run | epochs | best val loss | best val RMSD | final val loss | logged time | CSV |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        required = {"epoch", "val_loss", "val_rmsd", "seconds"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"Invalid training log schema: {relpath(path)}")
        best_loss = min(rows, key=lambda row: float(row["val_loss"]))
        best_rmsd = min(rows, key=lambda row: float(row["val_rmsd"]))
        final = rows[-1]
        seconds = sum(float(row["seconds"]) for row in rows)
        rmsd = (
            "N/A (`pos_r=0`)"
            if path.name == "p2_stage1_train_log.csv"
            else f'{float(best_rmsd["val_rmsd"]):.5f} (epoch {int(best_rmsd["epoch"])})'
        )
        lines.append(
            f'| {labels[path.name]} | {len(rows)} | {float(best_loss["val_loss"]):.5f} '
            f'(epoch {int(best_loss["epoch"])}) | {rmsd} | {float(final["val_loss"]):.5f} | '
            f'{seconds / 3600:.2f} h | [`{path.name}`](docs/{path.name}) |'
        )
    return "\n".join(lines) + "\n"


def results_download_markdown(paths: list[Path]) -> str:
    lines = ["## Per-complex evaluation downloads", ""]
    for path in paths:
        lines.append(f"- [`{path.name}`](results/{path.name})")
    return "\n".join(lines) + "\n"


QUIZ_HEADER_RE = re.compile(r"^# quiz: ([a-z0-9][a-z0-9-]*)$")
QUIZ_SECTION_RE = re.compile(r"^## ([a-z0-9][a-z0-9-]*)$")
QUIZ_Q_RE = re.compile(r"^### Q([1-9][0-9]*)$")
QUIZ_A_RE = re.compile(r"^### A([1-9][0-9]*)$")


def parse_quiz_body(text: str, source: Path, link_base: Path, label: str) -> list[Block]:
    if not text.strip():
        raise ValueError(f"Empty {label} in {source.relative_to(OUTPUT_ROOT)}")
    screened = list(text)
    for span in extract_code_spans(text):
        for index in range(span.start, span.end):
            if screened[index] != "\n":
                screened[index] = " "
    if re.search(
        r"^ {0,3}`{3,}|^ {0,3}#{1,3}\s|^\s*\|.*\|\s*$|<[/!A-Za-z][^>]*>",
        "".join(screened),
        re.MULTILINE,
    ):
        raise ValueError(f"Forbidden quiz-body syntax in {source.relative_to(OUTPUT_ROOT)} ({label})")
    blocks = parse_blocks(text.strip() + "\n", source, link_base)
    if any(block.kind not in {"paragraph", "list"} for block in blocks):
        raise ValueError(f"Quiz bodies allow only paragraphs and lists: {source.relative_to(OUTPUT_ROOT)}")
    return blocks


def parse_quiz_file(path: Path, expected_stem: str, link_base: Path) -> dict[str, QuizBlock]:
    lines = read_text(path).splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines or not QUIZ_HEADER_RE.fullmatch(lines[0]) or QUIZ_HEADER_RE.fullmatch(lines[0]).group(1) != expected_stem:
        raise ValueError(f"Quiz header must be '# quiz: {expected_stem}' in {path.relative_to(OUTPUT_ROOT)}")
    result: dict[str, QuizBlock] = {}
    i = 1
    while i < len(lines):
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        section_match = QUIZ_SECTION_RE.fullmatch(lines[i])
        if not section_match:
            raise ValueError(f"Expected quiz section at line {i + 1} in {path.relative_to(OUTPUT_ROOT)}")
        section_key = section_match.group(1)
        if section_key in result:
            raise ValueError(f"Duplicate quiz key {section_key} in {path.relative_to(OUTPUT_ROOT)}")
        i += 1
        questions: list[QuizQuestion] = []
        expected_number = 1
        while True:
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i >= len(lines) or QUIZ_SECTION_RE.fullmatch(lines[i]):
                break
            q_match = QUIZ_Q_RE.fullmatch(lines[i])
            if not q_match or int(q_match.group(1)) != expected_number:
                raise ValueError(f"Expected contiguous Q{expected_number} in {path.relative_to(OUTPUT_ROOT)}")
            qid = f"Q{expected_number}"
            i += 1
            question_lines: list[str] = []
            while i < len(lines) and not QUIZ_A_RE.fullmatch(lines[i]):
                if QUIZ_Q_RE.fullmatch(lines[i]) or QUIZ_SECTION_RE.fullmatch(lines[i]):
                    raise ValueError(f"Missing A{expected_number} in {path.relative_to(OUTPUT_ROOT)}")
                question_lines.append(lines[i])
                i += 1
            if i >= len(lines) or not QUIZ_A_RE.fullmatch(lines[i]) or int(QUIZ_A_RE.fullmatch(lines[i]).group(1)) != expected_number:
                raise ValueError(f"Expected A{expected_number} in {path.relative_to(OUTPUT_ROOT)}")
            i += 1
            answer_lines: list[str] = []
            while i < len(lines) and not QUIZ_Q_RE.fullmatch(lines[i]) and not QUIZ_SECTION_RE.fullmatch(lines[i]):
                answer_lines.append(lines[i])
                i += 1
            question = parse_quiz_body("\n".join(question_lines), path, link_base, qid)
            if len(question) != 1 or question[0].kind != "paragraph":
                raise ValueError(f"{qid} must be one paragraph in {path.relative_to(OUTPUT_ROOT)}")
            answer = parse_quiz_body("\n".join(answer_lines), path, link_base, f"A{expected_number}")
            questions.append(QuizQuestion(qid, question, answer))
            expected_number += 1
        if not 3 <= len(questions) <= 5:
            raise ValueError(f"Quiz key {section_key} must contain 3-5 questions in {path.relative_to(OUTPUT_ROOT)}")
        result[section_key] = QuizBlock(section_key, questions, path)
    if not result:
        raise ValueError(f"No quiz sections in {path.relative_to(OUTPUT_ROOT)}")
    return result


def load_core_quizzes(page: Page) -> dict[str, QuizBlock]:
    path = QUIZ_CONTENT_DIR / f"{page.stem}.{page.lang}.md"
    if not path.is_file():
        raise ValueError(f"Missing core quiz file: {path.relative_to(OUTPUT_ROOT)}")
    quizzes = parse_quiz_file(path, page.stem, page.canonical_source.parent)
    if set(quizzes) != {"page"}:
        raise ValueError(f"Core quiz {path.name} must contain exactly the 'page' key")
    return quizzes


def load_walkthrough_quizzes(page: Page) -> dict[str, QuizBlock]:
    path = QUIZ_CONTENT_DIR / f"{page.stem}.{page.lang}.md"
    if not path.is_file():
        raise ValueError(f"Missing walkthrough quiz file: {path.relative_to(OUTPUT_ROOT)}")
    return parse_quiz_file(path, page.stem, page.canonical_source.parent)


def build_pages() -> list[Page]:
    readme_path = REPO_ROOT / "README.md"
    preamble, readme_sections = split_h2_sections(read_text(readme_path))
    section_map = {title: body for title, body in readme_sections}

    final_title = next(title for title in section_map if title.startswith("🚧 Final submission"))
    final_chunk = section_map[final_title]
    if "**Contents:**" not in final_chunk:
        raise ValueError("README Contents marker not found")
    final_chunk, omitted_contents = final_chunk.split("**Contents:**", 1)
    final_chunk = final_chunk.rstrip() + "\n"
    if "What we did & why" not in omitted_contents:
        raise ValueError("README Contents list could not be identified")

    def section(prefix: str) -> str:
        matches = [body for title, body in readme_sections if title.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one README section starting with {prefix!r}")
        return matches[0]

    results_readme = REPO_ROOT / "results/README.md"
    _, result_sections = split_h2_sections(read_text(results_readme))
    result_map = {title: body for title, body in result_sections}
    selected_result_docs = result_map["Layout"] + "\n" + result_map["Eval CSV columns"]

    scripts_readme = REPO_ROOT / "scripts/README.md"
    provenance = read_text(scripts_readme)
    provenance_lines = provenance.splitlines()
    provenance_lines[0] = "## Authorship & provenance"
    provenance = "\n".join(provenance_lines)
    provenance = re.sub(r"^## 1\. Files WE created\s*$", "### Files we created", provenance, flags=re.MULTILINE)
    provenance = re.sub(r"^### ", "### ", provenance, flags=re.MULTILINE)

    train_logs = sorted((REPO_ROOT / "docs").glob("*_train_log.csv"))
    eval_csvs = sorted((REPO_ROOT / "results").glob("*_evaluation.csv"))
    if len(train_logs) != 4 or len(eval_csvs) != 10:
        raise ValueError("Expected four training logs and ten evaluation CSVs")

    def segment(source: Path, markdown: str, link_base: Path | None = None) -> ContentSegment:
        return ContentSegment(source, link_base or source.parent, markdown)

    pages = [
        Page(
            "index", "en", "ltr", "index.html", "Overview", "Overview", "index", readme_path,
            [segment(readme_path, preamble + "\n" + section("1. What we did"))],
        ),
        Page(
            "pipelines", "en", "ltr", "pipelines.html", "Pipelines", "Pipelines", "pipelines", readme_path,
            [segment(readme_path, promote_first_heading(section("2. The three pipelines")))],
        ),
        Page(
            "results", "en", "ltr", "results.html", "Results", "Results", "results", readme_path,
            [
                segment(readme_path, "# Results\n\n" + final_chunk + "\n" + section("3. Results")),
                segment(results_readme, selected_result_docs),
                segment(readme_path, results_download_markdown(eval_csvs)),
            ],
        ),
        Page(
            "reproduce", "en", "ltr", "reproduce.html", "Reproduce", "Reproduce", "reproduce", readme_path,
            [segment(readme_path, promote_first_heading(section("4. evaluate / reproduce")))],
        ),
        Page(
            "training", "en", "ltr", "training.html", "Training", "Training", "training", readme_path,
            [
                segment(readme_path, promote_first_heading(section("5. Training information"))),
                segment(readme_path, training_summary_markdown(train_logs)),
            ],
        ),
        Page(
            "reference", "en", "ltr", "reference.html", "Reference", "Reference", "reference", readme_path,
            [
                segment(readme_path, "# Reference\n\n" + section("6. Repository layout") + "\n" + section("7. Issues & fixes")),
                segment(scripts_readme, provenance),
            ],
        ),
    ]
    for source in WALKTHROUGH_FILES:
        title = "Code walkthroughs" if source.name == "README.md" else markdown_plain(read_text(source).splitlines()[0][2:])
        nav_label = title
        if len(nav_label) > 38:
            nav_label = source.stem.removeprefix("scripts.").replace("_", "-")
        filename = walkthrough_output(source)
        stem = Path(filename).stem
        pages.append(
            Page(
                stem,
                "en",
                "ltr",
                filename,
                title,
                nav_label,
                stem if stem in ARABIC_PAGE_STEMS else None,
                source,
                [segment(source, read_text(source))],
            )
        )

    ui_ar = load_arabic_ui()
    statuses = load_arabic_statuses()
    for stem in CORE_STEMS:
        source = ARABIC_CONTENT_DIR / f"{stem}.md"
        if not source.is_file():
            raise ValueError(f"Missing Arabic core content: {source.relative_to(OUTPUT_ROOT)}")
        markdown = read_text(source)
        first_heading = next((line[2:].strip() for line in markdown.splitlines() if line.startswith("# ")), "")
        first_heading = EXPLICIT_ID_RE.sub("", first_heading).strip()
        pages.append(
            Page(
                stem,
                "ar",
                "rtl",
                localized_filename(stem, "ar"),
                first_heading,
                ui_ar["page_labels"][stem],
                stem,
                readme_path,
                [segment(source, markdown, REPO_ROOT)],
                status=statuses[stem],
            )
        )

    english_walkthroughs = {
        page.stem: page
        for page in pages
        if page.lang == "en" and page.stem.startswith("walkthrough")
    }
    for stem in ARABIC_WALKTHROUGH_STEMS:
        english_page = english_walkthroughs[stem]
        source = ARABIC_CONTENT_DIR / f"{stem}.md"
        if not source.is_file():
            raise ValueError(f"Missing Arabic walkthrough content: {source.relative_to(OUTPUT_ROOT)}")
        markdown = read_text(source)
        first_heading = next((line[2:].strip() for line in markdown.splitlines() if line.startswith("# ")), "")
        first_heading = EXPLICIT_ID_RE.sub("", first_heading).strip()
        pages.append(
            Page(
                stem,
                "ar",
                "rtl",
                localized_filename(stem, "ar"),
                first_heading,
                first_heading,
                stem,
                english_page.canonical_source,
                [segment(source, markdown, english_page.canonical_source.parent)],
                status=statuses[stem],
            )
        )

    if [page.filename for page in pages] != HTML_MANIFEST:
        raise ValueError("Page construction does not match declared HTML manifest")
    return pages


class Resolver:
    def __init__(
        self,
        lang: str,
        page_by_source: dict[str, str],
        heading_routes: dict[tuple[str, str], tuple[str, str]],
        global_heading_routes: dict[tuple[str, str], tuple[str, str]],
    ):
        self.lang = lang
        self.page_by_source = page_by_source
        self.heading_routes = heading_routes
        self.global_heading_routes = global_heading_routes
        self.downloads: dict[str, Path] = {}
        self.images: dict[str, Path] = {}

        for path in sorted((REPO_ROOT / "results").glob("*_evaluation.csv")):
            self._register_download(path)
        for path in sorted((REPO_ROOT / "docs").glob("*_train_log.csv")):
            self._register_download(path)
        self._register_download(REPO_ROOT / "data/proto_test.csv")
        self._register_download(REPO_ROOT / "notebooks/results_and_comparison.ipynb")
        for name in ("workflow_training.png", "workflow_inference.png"):
            path = REPO_ROOT / "docs" / name
            self.images[f"assets/images/{name}"] = path

        self.source_routes = {
            "scripts/train.py": "walkthrough-train.html",
            "scripts/train_ddp.py": "walkthrough-train-ddp.html",
            "scripts/seminar_csv.py": "walkthrough-seminar-csv.html",
            "scripts/convert_seminar_to_karmadock.py": "walkthrough-convert-seminar-to-karmadock.html",
            "scripts/convert_karmadock_to_seminar.py": "walkthrough-convert-karmadock-to-seminar.html",
            "scripts/run_train.sh": "walkthrough-run-train.html",
            "scripts/run_full_train.sh": "walkthrough-run-full-train.html",
            "scripts/run_infer.sh": "walkthrough-run-infer.html",
            "scripts/run_full_stage2_ddp.sh": "walkthrough-run-full-stage2-ddp.html",
            "scripts/run_full_stage2_ddp_v2.sh": "walkthrough-run-full-stage2-ddp-v2.html",
            "scripts/evaluate.sh": "walkthrough-evaluate.html",
            "evaluation/evaluation.py": "walkthrough-evaluation.html",
        }

    def _register_download(self, path: Path) -> None:
        target = f"assets/downloads/{path.name}"
        if target in self.downloads and self.downloads[target] != path:
            raise ValueError(f"Download basename collision: {path.name}")
        self.downloads[target] = path

    def localize_href(self, href: str) -> str:
        path, separator, fragment = href.partition("#")
        stem = Path(path).stem
        if self.lang == "ar" and stem in ARABIC_PAGE_STEMS:
            path = localized_filename(stem, "ar")
        return path + (separator + fragment if separator else "")

    def image_href(self, source: Path, destination: str, link_base: Path | None = None) -> str:
        resolved = ((link_base or source.parent) / unquote(destination)).resolve()
        try:
            relative = relpath(resolved)
        except ValueError as error:
            raise ValueError(f"Image escapes repository: {destination}") from error
        if relative not in {"docs/workflow_training.png", "docs/workflow_inference.png"}:
            raise ValueError(f"Unclassified image: {relative}")
        return f"assets/images/{resolved.name}"

    def resolve(self, source: Path, destination: str, link_base: Path | None = None) -> LinkResolution:
        destination = destination.strip()
        parts = urlsplit(destination)
        if parts.scheme in {"http", "https", "mailto"}:
            return LinkResolution(destination, external=True)
        if parts.scheme:
            raise ValueError(f"Unsupported URL scheme in {destination!r}")

        source_key = relpath(source)
        if not parts.path and parts.fragment:
            route = self.heading_routes.get((source_key, unquote(parts.fragment)))
            if not route:
                route = self.global_heading_routes.get((self.lang, unquote(parts.fragment)))
            if not route:
                raise ValueError(f"Unknown fragment {destination!r} in {source_key}")
            return LinkResolution(self.localize_href(f"{route[0]}#{route[1]}"))

        target = ((link_base or source.parent) / unquote(parts.path)).resolve()
        try:
            target_key = relpath(target)
        except ValueError as error:
            raise ValueError(f"Local link escapes repository: {destination}") from error

        if target_key.endswith(".md"):
            page = self.page_by_source.get(target_key)
            if not page:
                raise ValueError(f"Markdown target is not routed: {target_key}")
            if parts.fragment:
                route = self.heading_routes.get((target_key, unquote(parts.fragment)))
                if not route:
                    raise ValueError(f"Unknown Markdown fragment: {target_key}#{parts.fragment}")
                return LinkResolution(self.localize_href(f"{route[0]}#{route[1]}"))
            return LinkResolution(self.localize_href(page))

        if target_key in self.source_routes:
            return LinkResolution(self.localize_href(self.source_routes[target_key]))

        if target_key.startswith("condor/") and target_key.endswith(".sub"):
            anchor = Path(target_key).stem.replace("_", "-") + "-sub"
            return LinkResolution(self.localize_href(f"walkthrough-condor.html#{anchor}"))

        directory_routes = {
            "scripts": "walkthroughs.html",
            "condor": "walkthrough-condor.html",
            "results": "results.html",
            "docs": "training.html",
            "model": "reference.html#6-repository-layout",
        }
        if target_key.rstrip("/") in directory_routes:
            return LinkResolution(self.localize_href(directory_routes[target_key.rstrip("/")]))

        if target_key == "Dockerfile":
            return LinkResolution(self.localize_href("reference.html#authorship--provenance"))

        for output_name, original in self.downloads.items():
            if target == original.resolve():
                suffix = f"#{parts.fragment}" if parts.fragment else ""
                return LinkResolution(output_name + suffix)

        if target_key.startswith("model/") and target_key.endswith(".pkl"):
            return LinkResolution(None, annotation="not_bundled")
        if target_key == "data/prototype_model_data.zip":
            return LinkResolution(None, annotation="not_bundled")

        raise ValueError(f"Unclassified local target {target_key!r} linked from {source_key}")


def extract_code_spans(text: str) -> list[CodeSpan]:
    spans: list[CodeSpan] = []
    i = 0
    while i < len(text):
        if text[i] != "`":
            i += 1
            continue
        run = 1
        while i + run < len(text) and text[i + run] == "`":
            run += 1
        closing = text.find("`" * run, i + run)
        if closing < 0:
            i += run
            continue
        value = text[i + run : closing]
        if value.startswith(" ") and value.endswith(" ") and value.strip():
            value = value[1:-1]
        spans.append(CodeSpan(i, closing + run, value.replace("\n", " ")))
        i = closing + run
    return spans


def find_closing(text: str, start: int, opening: str, closing: str, code_by_start: dict[int, CodeSpan]) -> int:
    depth = 1
    i = start
    while i < len(text):
        span = code_by_start.get(i)
        if span:
            i = span.end
            continue
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == opening:
            depth += 1
        elif text[i] == closing:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


LTR_RUN_RE = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9][A-Za-z0-9_.,/:+@%Å°=<>×-]*)(?![A-Za-z0-9_])")


def escape_plain(text: str, lang: str) -> str:
    if lang != "ar":
        return html.escape(text)
    output: list[str] = []
    cursor = 0
    for match in LTR_RUN_RE.finditer(text):
        output.append(html.escape(text[cursor : match.start()]))
        output.append(f'<bdi dir="ltr">{html.escape(match.group(0))}</bdi>')
        cursor = match.end()
    output.append(html.escape(text[cursor:]))
    return "".join(output)


def render_inline(
    text: str,
    source: Path,
    resolver: Resolver,
    lang: str = "en",
    link_base: Path | None = None,
    ui: dict | None = None,
) -> str:
    ui = ui or EN_UI
    spans = extract_code_spans(text)
    code_by_start = {span.start: span for span in spans}

    def render_region(start: int, end: int) -> str:
        output: list[str] = []
        plain: list[str] = []

        def flush() -> None:
            if plain:
                output.append(escape_plain("".join(plain), lang))
                plain.clear()

        i = start
        while i < end:
            span = code_by_start.get(i)
            if span and span.end <= end:
                flush()
                output.append(f'<code dir="ltr">{html.escape(span.value)}</code>')
                i = span.end
                continue

            image = text.startswith("![", i)
            if image or text[i] == "[":
                label_start = i + (2 if image else 1)
                close_label = find_closing(text, label_start, "[", "]", code_by_start)
                if close_label >= 0 and close_label + 1 < end and text[close_label + 1] == "(":
                    close_dest = find_closing(text, close_label + 2, "(", ")", code_by_start)
                    if 0 <= close_dest < end:
                        flush()
                        destination = text[close_label + 2 : close_dest].strip()
                        if image:
                            href = resolver.image_href(source, destination, link_base)
                            alt = markdown_plain(text[label_start:close_label])
                            output.append(f'<img src="{html.escape(href, quote=True)}" alt="{html.escape(alt, quote=True)}">')
                        else:
                            resolution = resolver.resolve(source, destination, link_base)
                            label = render_region(label_start, close_label)
                            if resolution.href is None:
                                output.append(
                                    f'<span class="unbundled">{label} '
                                    f'<span class="annotation">({html.escape(ui.get(resolution.annotation, resolution.annotation))})</span></span>'
                                )
                            else:
                                external = " external" if resolution.external else ""
                                output.append(
                                    f'<a class="content-link{external}" href="{html.escape(resolution.href, quote=True)}">{label}'
                                )
                                if resolution.external:
                                    output.append(
                                        '<span class="external-mark" aria-hidden="true">↗</span>'
                                        f'<span class="sr-only"> ({html.escape(ui["external_link"])})</span>'
                                    )
                                output.append("</a>")
                        i = close_dest + 1
                        continue

            if text.startswith("**", i) or text.startswith("__", i):
                marker = text[i : i + 2]
                close = text.find(marker, i + 2, end)
                if close >= 0:
                    flush()
                    output.append(f"<strong>{render_region(i + 2, close)}</strong>")
                    i = close + 2
                    continue

            if text[i] in "*_":
                marker = text[i]
                close = text.find(marker, i + 1, end)
                if close > i + 1:
                    flush()
                    output.append(f"<em>{render_region(i + 1, close)}</em>")
                    i = close + 1
                    continue

            if text[i] == "<":
                close = text.find(">", i + 1, end)
                if close >= 0:
                    candidate = text[i + 1 : close]
                    if re.match(r"https?://", candidate):
                        flush()
                        resolution = resolver.resolve(source, candidate, link_base)
                        output.append(
                            f'<a class="content-link external" href="{html.escape(resolution.href or "", quote=True)}">'
                            f'{html.escape(candidate)}<span class="external-mark" aria-hidden="true">↗</span>'
                            f'<span class="sr-only"> ({html.escape(ui["external_link"])})</span></a>'
                        )
                        i = close + 1
                        continue

            if text[i] == "\\" and i + 1 < end:
                plain.append(text[i + 1])
                i += 2
                continue
            plain.append(text[i])
            i += 1
        flush()
        return "".join(output)

    return render_region(0, len(text))


def position_offsets(raw: str) -> list[int]:
    starts = [0]
    for match in re.finditer("\n", raw):
        starts.append(match.end())
    starts.append(len(raw))
    return starts


def highlight_python(raw: str) -> str:
    offsets = position_offsets(raw)
    output: list[str] = []
    cursor = 0
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(raw).readline))
        for item in tokens:
            if item.type in {tokenize.ENDMARKER, tokenize.ENCODING}:
                continue
            start = offsets[min(item.start[0] - 1, len(offsets) - 1)] + item.start[1]
            end = offsets[min(item.end[0] - 1, len(offsets) - 1)] + item.end[1]
            if start < cursor:
                continue
            output.append(html.escape(raw[cursor:start]))
            css = ""
            if item.type == token.NAME and keyword.iskeyword(item.string):
                css = "tok-keyword"
            elif item.type == token.STRING:
                css = "tok-string"
            elif item.type == token.COMMENT:
                css = "tok-comment"
            elif item.type == token.NUMBER:
                css = "tok-number"
            escaped = html.escape(raw[start:end])
            output.append(f'<span class="{css}">{escaped}</span>' if css else escaped)
            cursor = end
        output.append(html.escape(raw[cursor:]))
        return "".join(output)
    except (tokenize.TokenError, IndentationError, SyntaxError, IndexError, ValueError):
        return html.escape(raw)


def highlight_plain_segment(segment: str, keywords: set[str]) -> str:
    output: list[str] = []
    cursor = 0
    for match in re.finditer(r"\b(?:\d+(?:\.\d+)?|[A-Za-z_][A-Za-z0-9_]*)\b", segment):
        output.append(html.escape(segment[cursor : match.start()]))
        value = match.group(0)
        css = "tok-number" if value[0].isdigit() else ("tok-keyword" if value in keywords else "")
        escaped = html.escape(value)
        output.append(f'<span class="{css}">{escaped}</span>' if css else escaped)
        cursor = match.end()
    output.append(html.escape(segment[cursor:]))
    return "".join(output)


def highlight_shell_like(raw: str, keywords: set[str]) -> str:
    output: list[str] = []
    for line in raw.splitlines(keepends=True):
        i = 0
        plain_start = 0
        while i < len(line):
            if line[i] in "'\"":
                output.append(highlight_plain_segment(line[plain_start:i], keywords))
                quote = line[i]
                j = i + 1
                escaped = False
                while j < len(line):
                    if escaped:
                        escaped = False
                    elif line[j] == "\\" and quote == '"':
                        escaped = True
                    elif line[j] == quote:
                        j += 1
                        break
                    j += 1
                output.append(f'<span class="tok-string">{html.escape(line[i:j])}</span>')
                i = j
                plain_start = j
                continue
            if line[i] == "#":
                output.append(highlight_plain_segment(line[plain_start:i], keywords))
                output.append(f'<span class="tok-comment">{html.escape(line[i:])}</span>')
                plain_start = len(line)
                i = len(line)
                continue
            i += 1
        if plain_start < len(line):
            output.append(highlight_plain_segment(line[plain_start:], keywords))
    return "".join(output)


BASH_KEYWORDS = {"if", "then", "else", "elif", "fi", "for", "while", "until", "do", "done", "case", "esac", "in", "function", "select", "time"}
CONDOR_KEYWORDS = {"universe", "executable", "arguments", "environment", "requirements", "request_gpus", "request_cpus", "request_memory", "should_transfer_files", "transfer_input_files", "transfer_output_files", "when_to_transfer_output", "output", "error", "log", "queue"}


def highlight_code(raw: str, language: str) -> str:
    if language in {"python", "py"}:
        return highlight_python(raw)
    if language in {"bash", "sh", "shell"}:
        return highlight_shell_like(raw, BASH_KEYWORDS)
    if language == "condor":
        return highlight_shell_like(raw, CONDOR_KEYWORDS)
    return html.escape(raw)


def iter_blocks(blocks: Iterable[Block]) -> Iterable[Block]:
    for block in blocks:
        yield block
        if block.children:
            yield from iter_blocks(block.children)


def render_blocks(blocks: list[Block], resolver: Resolver, page: Page, ui: dict) -> str:
    output: list[str] = []
    for block in blocks:
        if block.kind == "heading":
            output.append(
                f'<h{block.level} id="{html.escape(block.heading_id, quote=True)}">'
                f'{render_inline(block.text, block.source, resolver, page.lang, block.link_base, ui)}</h{block.level}>'
            )
        elif block.kind == "paragraph":
            output.append(f"<p>{render_inline(block.text, block.source, resolver, page.lang, block.link_base, ui)}</p>")
        elif block.kind == "hr":
            output.append("<hr>")
        elif block.kind == "code":
            language_class = f" language-{re.sub(r'[^a-z0-9_-]', '', block.language)}" if block.language else ""
            output.append(
                f'<div class="code-scroll"><pre dir="ltr"><code dir="ltr" class="{language_class.strip()}">'
                f'{highlight_code(block.text, block.language)}</code></pre></div>'
            )
        elif block.kind == "raw_image":
            href = resolver.image_href(block.source, block.attrs["src"], block.link_base)
            output.append(
                f'<figure class="workflow"><a href="{html.escape(href, quote=True)}">'
                f'<img src="{html.escape(href, quote=True)}" alt="{html.escape(block.attrs["alt"], quote=True)}" '
                f'width="{int(block.attrs["width"])}"></a></figure>'
            )
        elif block.kind == "table":
            output.append('<div class="table-scroll"><table><thead><tr>')
            for index, cell in enumerate(block.rows[0]):
                align = f' class="align-{block.aligns[index]}"' if index < len(block.aligns) and block.aligns[index] else ""
                output.append(f"<th{align}>{render_inline(cell, block.source, resolver, page.lang, block.link_base, ui)}</th>")
            output.append("</tr></thead><tbody>")
            for row in block.rows[1:]:
                output.append("<tr>")
                for index, cell in enumerate(row):
                    align = f' class="align-{block.aligns[index]}"' if index < len(block.aligns) and block.aligns[index] else ""
                    output.append(f"<td{align}>{render_inline(cell, block.source, resolver, page.lang, block.link_base, ui)}</td>")
                output.append("</tr>")
            output.append("</tbody></table></div>")
        elif block.kind == "blockquote":
            output.append(f"<blockquote>{render_blocks(block.children, resolver, page, ui)}</blockquote>")
        elif block.kind == "list":
            tag = "ol" if block.ordered else "ul"
            start = f' start="{block.start}"' if block.ordered and block.start != 1 else ""
            output.append(f"<{tag}{start}>")
            for item in block.items:
                output.append(f"<li>{render_inline(item, block.source, resolver, page.lang, block.link_base, ui)}</li>")
            output.append(f"</{tag}>")
        else:
            raise ValueError(f"Unknown block kind: {block.kind}")
    return "\n".join(output)


def render_quiz(quiz: QuizBlock, resolver: Resolver, page: Page, ui: dict) -> str:
    title_id = f"quiz-{quiz.section_key}-title"
    output = [
        f'<section class="quiz" aria-labelledby="{title_id}">',
        f'<h3 class="quiz-title" id="{title_id}">{html.escape(ui["quiz_title"])}</h3>',
        "<ol>",
    ]
    for question in quiz.questions:
        question_html = render_blocks(question.question, resolver, page, ui)
        question_html = re.sub(r"^<p>", '<p class="quiz-question">', question_html, count=1)
        answer_html = render_blocks(question.answer, resolver, page, ui)
        output.extend(
            [
                f'<li id="quiz-{quiz.section_key}-{question.qid.lower()}">',
                question_html,
                "<details>",
                f'<summary>{html.escape(ui["show_answer"])}</summary>',
                f'<div class="quiz-answer">{answer_html}</div>',
                "</details>",
                "</li>",
            ]
        )
    output.extend(["</ol>", "</section>"])
    return "\n".join(output)


def render_article(page: Page, resolver: Resolver, ui: dict) -> str:
    output: list[str] = []
    current_h2 = ""
    draft_inserted = False
    for block in page.blocks:
        if block.kind == "heading" and block.level == 2:
            if current_h2 and current_h2 in page.quizzes:
                output.append(render_quiz(page.quizzes[current_h2], resolver, page, ui))
            current_h2 = block.heading_id
        output.append(render_blocks([block], resolver, page, ui))
        if block.kind == "heading" and block.level == 1 and page.lang == "ar" and page.status == "draft":
            output.append(f'<aside class="translation-status" role="note">{html.escape(ui["draft_banner"])}</aside>')
            draft_inserted = True
    if current_h2 and current_h2 in page.quizzes:
        output.append(render_quiz(page.quizzes[current_h2], resolver, page, ui))
    if "page" in page.quizzes:
        output.append(render_quiz(page.quizzes["page"], resolver, page, ui))
    if page.lang == "ar" and page.status == "draft" and not draft_inserted:
        raise ValueError(f"Could not place draft banner on {page.filename}")
    return "\n".join(output)


def block_plain_text(block: Block) -> str:
    if block.kind in {"paragraph", "heading"}:
        return markdown_plain(block.text)
    if block.kind == "list":
        return " ".join(markdown_plain(item) for item in block.items)
    if block.kind == "table":
        return " ".join(markdown_plain(cell) for row in block.rows for cell in row)
    if block.kind == "blockquote":
        return " ".join(block_plain_text(child) for child in block.children if child.kind != "code")
    return ""


def make_search_entries(page: Page) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    flat = list(iter_blocks(page.blocks))
    heading_indexes = [index for index, block in enumerate(flat) if block.kind == "heading" and block.level in {2, 3}]
    quiz_questions = {
        key: " ".join(
            block_plain_text(block)
            for question in quiz.questions
            for block in question.question
        )
        for key, quiz in page.quizzes.items()
    }
    for position, index in enumerate(heading_indexes):
        heading = flat[index]
        end = heading_indexes[position + 1] if position + 1 < len(heading_indexes) else len(flat)
        prose = " ".join(block_plain_text(block) for block in flat[index + 1 : end] if block.kind != "code")
        prose = " ".join(prose.split())
        excerpt = prose[:220].rstrip()
        if len(prose) > 220:
            excerpt += "…"
        heading_text = markdown_plain(heading.text)
        questions = quiz_questions.get(heading.heading_id, "")
        if position == 0 and "page" in quiz_questions:
            questions = f'{questions} {quiz_questions["page"]}'.strip()
        combined = f"{page.title} {heading_text} {prose} {questions}"
        entries.append(
            {
                "title": page.title,
                "heading": heading_text,
                "url": f"{page.filename}#{heading.heading_id}",
                "excerpt": excerpt or heading_text,
                "text": normalize_search(combined),
            }
        )
    if not entries:
        raise ValueError(f"Search index has no h2/h3 entry for {page.filename}")
    return entries


def nav_html(current_page: Page, pages: list[Page], ui: dict) -> str:
    page_by_key = {(page.stem, page.lang): page for page in pages}

    def target(stem: str) -> Page:
        return page_by_key.get((stem, current_page.lang), page_by_key[(stem, "en")])

    def link(stem: str, label: str | None = None) -> str:
        page = target(stem)
        current_attr = ' aria-current="page"' if page.filename == current_page.filename else ""
        text = label or page.nav_label
        rendered = html.escape(text)
        if current_page.lang == "ar" and page.lang == "en" and label is None:
            rendered = f'<bdi dir="ltr">{rendered}</bdi>'
        return f'<a href="{page.filename}"{current_attr}>{rendered}</a>'

    walkthrough_links = "".join(
        f'<li>{link(Path(name).stem, ui["nav_walkthroughs"] if current_page.lang == "ar" and Path(name).stem == "walkthroughs" else None)}</li>'
        for name in WALKTHROUGH_OUTPUTS
    )
    return (
        f'<nav class="course-nav" aria-label="{html.escape(ui["nav_label"], quote=True)}">'
        f'<p class="nav-group">{html.escape(ui["nav_course"])}</p><ul>'
        f'<li>{link("index")}</li><li>{link("pipelines")}</li><li>{link("results")}</li>'
        f'<li>{link("reproduce")}</li><li>{link("training")}</li></ul>'
        f'<p class="nav-group">{html.escape(ui["nav_walkthroughs"])}</p><ul class="walkthrough-nav">'
        f"{walkthrough_links}</ul>"
        f'<p class="nav-group">{html.escape(ui["nav_reference"])}</p><ul><li>{link("reference")}</li></ul></nav>'
    )


def toc_html(page: Page, ui: dict) -> str:
    headings = [block for block in iter_blocks(page.blocks) if block.kind == "heading" and block.level in {2, 3}]
    if not headings:
        return f'<p class="toc-empty">{html.escape(ui["no_subsections"])}</p>'
    return "<ol>" + "".join(
        f'<li class="toc-level-{block.level}"><a href="#{html.escape(block.heading_id, quote=True)}">'
        f'{escape_plain(markdown_plain(block.text), page.lang)}</a></li>' for block in headings
    ) + "</ol>"


def render_page(page: Page, pages: list[Page], resolver: Resolver) -> str:
    ui = EN_UI if page.lang == "en" else load_arabic_ui()
    content = render_article(page, resolver, ui)
    counterpart_page = next(
        (candidate for candidate in pages if candidate.stem == page.counterpart and candidate.lang != page.lang),
        None,
    ) if page.counterpart else None
    alternate = ""
    language_toggle = ""
    counterpart_attr = ""
    if counterpart_page:
        alternate = (
            f'  <link rel="alternate" hreflang="{counterpart_page.lang}" href="{counterpart_page.filename}">\n'
        )
        language_toggle = (
            f'<a class="text-button language-toggle" id="language-toggle" '
            f'href="{counterpart_page.filename}" hreflang="{counterpart_page.lang}" '
            f'data-language="{counterpart_page.lang}" aria-label="{html.escape(ui["language_aria"], quote=True)}">'
            f'{html.escape(ui["language_label"])}</a>'
        )
        counterpart_attr = f' data-counterpart="{counterpart_page.filename}"'
    home = localized_filename("index", page.lang) if page.lang == "ar" else "index.html"
    return f'''<!doctype html>
<html lang="{page.lang}" dir="{page.direction}"{counterpart_attr}>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>{html.escape(page.title)} | KarmaDock {html.escape(ui["site_subtitle"])}</title>
{alternate.rstrip()}
  <link rel="stylesheet" href="assets/styles.css">
  <script defer src="assets/search-normalization-fixtures.js"></script>
  <script defer src="assets/search-index.{page.lang}.js"></script>
  <script defer src="assets/app.js"></script>
  <script defer src="assets/search.js"></script>
</head>
<body>
  <a class="skip-link" href="#main-content">{html.escape(ui["skip_link"])}</a>
  <header class="site-header">
    <button class="icon-button menu-button" id="menu-toggle" type="button" aria-label="{html.escape(ui["menu_label"], quote=True)}" aria-expanded="false" aria-controls="sidebar">☰</button>
    <a class="site-title" href="{home}"><bdi dir="ltr">KarmaDock</bdi> <span>{html.escape(ui["site_subtitle"])}</span></a>
    <div class="header-actions">
      {language_toggle}
      <button class="text-button" id="search-trigger" type="button" aria-label="{html.escape(ui["open_search"], quote=True)}">{html.escape(ui["search_button"])} <kbd dir="ltr">/</kbd></button>
      <button class="icon-button" id="theme-toggle" type="button" aria-label="{html.escape(ui["theme_dark"], quote=True)}" aria-pressed="false" data-label-light="{html.escape(ui["theme_light"], quote=True)}" data-label-dark="{html.escape(ui["theme_dark"], quote=True)}">◐</button>
    </div>
  </header>
  <div class="page-shell">
    <aside class="sidebar" id="sidebar">{nav_html(page, pages, ui)}</aside>
    <main id="main-content" tabindex="-1"><article>{content}</article></main>
    <aside class="page-toc" aria-label="{html.escape(ui["on_this_page"], quote=True)}"><p class="toc-title">{html.escape(ui["on_this_page"])}</p>{toc_html(page, ui)}</aside>
  </div>
  <footer class="site-footer"><p>{escape_plain(ui["footer"], page.lang)}</p></footer>
  <dialog class="search-dialog" id="search-dialog" aria-labelledby="search-title" data-empty="{html.escape(ui["search_empty"], quote=True)}" data-none="{html.escape(ui["search_none"], quote=True)}" data-result-singular="{html.escape(ui["result_singular"], quote=True)}" data-result-plural="{html.escape(ui["result_plural"], quote=True)}">
    <div class="search-head"><h2 id="search-title">{html.escape(ui["search_title"])}</h2><button class="icon-button" id="search-close" type="button" aria-label="{html.escape(ui["close_search"], quote=True)}">×</button></div>
    <label for="search-input">{html.escape(ui["search_label"])}</label>
    <input id="search-input" type="search" dir="auto" autocomplete="off" spellcheck="false" placeholder="{html.escape(ui["search_placeholder"], quote=True)}">
    <p class="search-status" id="search-status" role="status" aria-live="polite"></p>
    <ol class="search-results" id="search-results"></ol>
  </dialog>
</body>
</html>
'''


STYLES_CSS = r''':root {
  color-scheme: light;
  --hue: 203;
  --bg: oklch(97.5% 0.012 var(--hue));
  --surface: oklch(94.5% 0.018 var(--hue));
  --surface-strong: oklch(90% 0.025 var(--hue));
  --text: oklch(25% 0.035 var(--hue));
  --muted: oklch(48% 0.035 var(--hue));
  --line: oklch(82% 0.035 var(--hue));
  --accent: oklch(52% 0.15 214);
  --accent-strong: oklch(42% 0.16 214);
  --code-bg: oklch(22% 0.035 var(--hue));
  --code-text: oklch(91% 0.025 var(--hue));
  --focus: oklch(67% 0.18 55);
  --header-height: 4rem;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 16px;
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: oklch(20% 0.025 var(--hue));
    --surface: oklch(24% 0.032 var(--hue));
    --surface-strong: oklch(29% 0.04 var(--hue));
    --text: oklch(90% 0.025 var(--hue));
    --muted: oklch(70% 0.035 var(--hue));
    --line: oklch(38% 0.04 var(--hue));
    --accent: oklch(72% 0.13 210);
    --accent-strong: oklch(80% 0.11 210);
  }
}

:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: oklch(20% 0.025 var(--hue));
  --surface: oklch(24% 0.032 var(--hue));
  --surface-strong: oklch(29% 0.04 var(--hue));
  --text: oklch(90% 0.025 var(--hue));
  --muted: oklch(70% 0.035 var(--hue));
  --line: oklch(38% 0.04 var(--hue));
  --accent: oklch(72% 0.13 210);
  --accent-strong: oklch(80% 0.11 210);
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; overflow-x: hidden; }
body { margin: 0; min-width: 0; overflow-x: hidden; background: var(--bg); color: var(--text); line-height: 1.65; }
a { color: var(--accent-strong); text-underline-offset: 0.18em; }
a:hover { color: var(--accent); }
:focus-visible { outline: 3px solid var(--focus); outline-offset: 3px; }
.sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
.skip-link { position: fixed; top: 0.5rem; inset-inline-start: 0.5rem; z-index: 100; padding: 0.6rem 0.85rem; background: var(--text); color: var(--bg); transform: translateY(-160%); transition: transform 140ms ease-out; }
.skip-link:focus { transform: translateY(0); }

.site-header { position: sticky; top: 0; z-index: 20; min-height: var(--header-height); display: flex; align-items: center; gap: 1rem; padding: 0.65rem 1.1rem; border-bottom: 1px solid var(--line); background: var(--bg); }
.site-title { color: var(--text); font-family: Georgia, "Times New Roman", serif; font-weight: 700; font-size: 1.25rem; text-decoration: none; }
.site-title span { color: var(--muted); font-weight: 400; }
.header-actions { margin-inline-start: auto; display: flex; align-items: center; gap: 0.55rem; }
button { font: inherit; color: inherit; }
.icon-button, .text-button { border: 1px solid var(--line); background: var(--surface); border-radius: 0.45rem; cursor: pointer; min-height: 2.4rem; transition: background-color 140ms ease-out, border-color 140ms ease-out; }
.icon-button:hover, .text-button:hover { background: var(--surface-strong); border-color: var(--accent); }
.icon-button { min-width: 2.5rem; padding: 0.35rem 0.6rem; }
.text-button { padding: 0.35rem 0.75rem; }
kbd { border: 1px solid var(--line); border-bottom-width: 2px; border-radius: 0.25rem; padding: 0.05rem 0.3rem; font: 0.8em ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.menu-button { display: none; }

.page-shell { display: grid; grid-template-columns: minmax(13rem, 17rem) minmax(0, 72ch) minmax(11rem, 15rem); grid-template-areas: "sidebar main toc"; justify-content: center; gap: clamp(1.5rem, 3vw, 3.5rem); padding: 2rem clamp(1rem, 3vw, 3rem) 4rem; }
.sidebar { grid-area: sidebar; }
main { grid-area: main; }
.page-toc { grid-area: toc; }
.sidebar, .page-toc { position: sticky; top: calc(var(--header-height) + 1.5rem); align-self: start; max-height: calc(100vh - var(--header-height) - 3rem); overflow-y: auto; }
.course-nav ul, .page-toc ol { list-style: none; padding: 0; margin: 0; }
.nav-group, .toc-title { margin: 1.2rem 0 0.35rem; color: var(--muted); font-size: 0.73rem; font-weight: 750; letter-spacing: 0.09em; text-transform: uppercase; }
.nav-group:first-child { margin-top: 0; }
.course-nav a { display: block; padding: 0.28rem 0.55rem; border-radius: 0.35rem; color: var(--muted); text-decoration: none; line-height: 1.35; }
.course-nav a:hover { color: var(--text); background: var(--surface); }
.course-nav a[aria-current="page"] { color: var(--accent-strong); background: var(--surface-strong); font-weight: 700; }
.walkthrough-nav { font-size: 0.88rem; }
.page-toc a { display: block; padding: 0.22rem 0; color: var(--muted); text-decoration: none; font-size: 0.85rem; line-height: 1.35; }
.page-toc a:hover { color: var(--accent-strong); }
.toc-level-3 { padding-inline-start: 0.8rem; }
.toc-empty { color: var(--muted); font-size: 0.85rem; }

main { min-width: 0; }
article { min-width: 0; }
h1, h2, h3 { color: var(--text); font-family: Georgia, "Times New Roman", serif; line-height: 1.18; scroll-margin-top: calc(var(--header-height) + 1.2rem); }
h1 { margin: 0 0 1.2rem; font-size: clamp(2.15rem, 5vw, 3.6rem); letter-spacing: -0.035em; }
h2 { margin: 3rem 0 0.85rem; padding-top: 0.4rem; font-size: clamp(1.55rem, 3vw, 2.1rem); }
h3 { margin: 2rem 0 0.65rem; font-size: 1.28rem; }
p, li { max-width: 72ch; }
hr { margin: 2.5rem 0; border: 0; border-top: 1px solid var(--line); }
blockquote { margin: 1.5rem 0; padding: 0.85rem 1.1rem; background: var(--surface); border-radius: 0.45rem; color: var(--muted); }
blockquote p { margin: 0.3rem 0; }
code { border-radius: 0.24rem; padding: 0.08em 0.28em; background: var(--surface-strong); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.9em; }
.code-scroll { max-width: 100%; overflow-x: auto; margin: 1.2rem 0 1.7rem; border-radius: 0.5rem; background: var(--code-bg); }
pre { margin: 0; padding: 1rem 1.1rem; min-width: max-content; color: var(--code-text); line-height: 1.5; tab-size: 4; direction: ltr; unicode-bidi: isolate; text-align: left; }
pre code { padding: 0; background: transparent; color: inherit; font-size: 0.84rem; }
.ltr-island, bdi[dir="ltr"], code[dir="ltr"], kbd[dir="ltr"] { direction: ltr; unicode-bidi: isolate; }
.tok-keyword { color: oklch(78% 0.13 305); font-weight: 650; }
.tok-string { color: oklch(80% 0.12 145); }
.tok-comment { color: oklch(67% 0.03 var(--hue)); font-style: italic; }
.tok-number { color: oklch(80% 0.12 72); }
.table-scroll { max-width: 100%; overflow-x: auto; margin: 1.2rem 0 1.8rem; }
table { width: 100%; min-width: 34rem; border-collapse: collapse; font-size: 0.91rem; }
th, td { padding: 0.65rem 0.75rem; border-bottom: 1px solid var(--line); text-align: start; vertical-align: top; }
th { background: var(--surface); font-weight: 720; }
.align-center { text-align: center; }
.align-right { text-align: end; }
.workflow { margin: 1.6rem 0 2rem; }
.workflow img, article > p img { display: block; max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 0.45rem; }
.external-mark { margin-inline-start: 0.18em; font-size: 0.82em; text-decoration: none; }
.unbundled .annotation { color: var(--muted); font-size: 0.9em; }

.site-footer { border-top: 1px solid var(--line); padding: 1.5rem 1rem 2.5rem; color: var(--muted); text-align: center; font-size: 0.85rem; }
.site-footer p { margin: auto; }
.search-dialog { width: min(42rem, calc(100vw - 2rem)); max-height: min(42rem, calc(100vh - 2rem)); border: 1px solid var(--line); border-radius: 0.65rem; padding: 1rem; background: var(--bg); color: var(--text); box-shadow: 0 1.5rem 4rem oklch(10% 0.03 var(--hue) / 0.35); }
.search-dialog::backdrop { background: oklch(15% 0.02 var(--hue) / 0.68); }
.search-head { display: flex; align-items: center; gap: 1rem; }
.search-head h2 { margin: 0; padding: 0; font-size: 1.45rem; }
.search-head button { margin-inline-start: auto; }
.search-dialog label { display: block; margin: 1rem 0 0.35rem; color: var(--muted); font-size: 0.86rem; }
.search-dialog input { width: 100%; border: 1px solid var(--line); border-radius: 0.4rem; padding: 0.75rem 0.85rem; background: var(--surface); color: var(--text); font: inherit; }
.search-status { color: var(--muted); font-size: 0.85rem; }
.search-results { list-style: none; padding: 0; margin: 0; }
.search-results a { display: block; padding: 0.7rem 0.75rem; border-radius: 0.4rem; color: var(--text); text-decoration: none; }
.search-results a:hover, .search-results a[data-active="true"] { background: var(--surface-strong); }
.result-title { display: block; color: var(--accent-strong); font-weight: 720; }
.result-excerpt { display: block; color: var(--muted); font-size: 0.86rem; }
.translation-status { margin: 1rem 0 1.5rem; padding: 0.75rem 0.9rem; border-radius: 0.45rem; background: var(--surface); color: var(--muted); }
.quiz { margin: 2.5rem 0 1rem; padding: 1rem 1.1rem; border: 1px solid var(--line); border-radius: 0.55rem; background: var(--surface); }
.quiz-title { margin: 0 0 0.75rem; }
.quiz > ol { margin-block: 0; padding-inline-start: 1.5rem; }
.quiz li + li { margin-block-start: 1rem; }
.quiz-question { font-weight: 650; }
.quiz details { margin-block-start: 0.35rem; }
.quiz summary { color: var(--accent-strong); cursor: pointer; font-weight: 650; }
.quiz-answer { margin-block-start: 0.5rem; color: var(--muted); }
:lang(ar) { font-family: system-ui, "Geeza Pro", Tahoma, Arial, sans-serif; line-height: 1.8; }
:lang(ar) h1, :lang(ar) h2, :lang(ar) h3 { font-family: "Geeza Pro", Tahoma, Arial, sans-serif; }

@media (max-width: 1100px) {
  .page-shell { grid-template-columns: minmax(12rem, 15rem) minmax(0, 72ch); grid-template-areas: "sidebar main"; }
  .page-toc { display: none; }
}

@media (max-width: 760px) {
  .menu-button { display: inline-block; }
  .site-title span, .text-button kbd { display: none; }
  .page-shell { display: block; padding: 1.35rem 1rem 3rem; }
  .sidebar { display: none; position: fixed; inset: var(--header-height) 0 auto 0; z-index: 15; max-height: calc(100vh - var(--header-height)); padding: 1rem; border-bottom: 1px solid var(--line); background: var(--bg); }
  body.nav-open .sidebar { display: block; }
  h1 { font-size: clamp(2rem, 11vw, 2.85rem); }
  table { min-width: 38rem; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; transition-duration: 0.01ms !important; }
}
'''


APP_JS = r'''(() => {
  "use strict";
  const root = document.documentElement;
  const themeButton = document.getElementById("theme-toggle");
  const menuButton = document.getElementById("menu-toggle");
  const languageToggle = document.getElementById("language-toggle");

  let savedLanguage = null;
  try { savedLanguage = localStorage.getItem("karmadock-language"); } catch (_) { savedLanguage = null; }
  const counterpart = root.dataset.counterpart || "";
  if (counterpart && (savedLanguage === "en" || savedLanguage === "ar") && savedLanguage !== root.lang) {
    window.location.replace(counterpart + window.location.hash);
    return;
  }
  if (languageToggle) {
    languageToggle.href = languageToggle.getAttribute("href") + window.location.hash;
    languageToggle.addEventListener("click", () => {
      try { localStorage.setItem("karmadock-language", languageToggle.dataset.language); } catch (_) { /* storage is optional */ }
    });
  }

  let savedTheme = null;
  try { savedTheme = localStorage.getItem("karmadock-theme"); } catch (_) { savedTheme = null; }
  if (savedTheme === "light" || savedTheme === "dark") root.dataset.theme = savedTheme;

  const effectiveDark = () => root.dataset.theme === "dark" ||
    (!root.dataset.theme && window.matchMedia("(prefers-color-scheme: dark)").matches);
  const updateThemeButton = () => {
    const dark = effectiveDark();
    themeButton.setAttribute("aria-pressed", String(dark));
    themeButton.setAttribute("aria-label", dark ? themeButton.dataset.labelLight : themeButton.dataset.labelDark);
  };
  updateThemeButton();
  themeButton.addEventListener("click", () => {
    const next = effectiveDark() ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("karmadock-theme", next); } catch (_) { /* storage is optional */ }
    updateThemeButton();
  });

  menuButton.addEventListener("click", () => {
    const open = document.body.classList.toggle("nav-open");
    menuButton.setAttribute("aria-expanded", String(open));
  });
  document.querySelectorAll(".sidebar a").forEach((link) => link.addEventListener("click", () => {
    document.body.classList.remove("nav-open");
    menuButton.setAttribute("aria-expanded", "false");
  }));
})();
'''


SEARCH_JS = r'''(() => {
  "use strict";
  const dialog = document.getElementById("search-dialog");
  const trigger = document.getElementById("search-trigger");
  const closeButton = document.getElementById("search-close");
  const input = document.getElementById("search-input");
  const status = document.getElementById("search-status");
  const list = document.getElementById("search-results");
  const payload = window.KD_SEARCH_INDEX;
  if (!payload || payload.lang !== document.documentElement.lang || !Array.isArray(payload.entries)) {
    throw new Error("Search index language mismatch");
  }
  const index = payload.entries;
  let priorFocus = null;
  let matches = [];
  let active = -1;

  const normalize = (value) => value.toLocaleLowerCase().normalize("NFKD")
    .replace(/\p{M}/gu, "").replace(/ـ/g, "").replace(/[أإآ]/g, "ا").trim().replace(/\s+/g, " ");
  const fixtures = Array.isArray(window.KD_NORMALIZATION_FIXTURES) ? window.KD_NORMALIZATION_FIXTURES : [];
  fixtures.forEach((fixture) => {
    if (normalize(fixture.input) !== fixture.expected) throw new Error("Search normalization fixture mismatch");
  });
  const editable = (target) => target instanceof HTMLElement &&
    (target.matches("input, textarea, select") || target.isContentEditable);

  const open = () => {
    priorFocus = document.activeElement;
    if (!dialog.open) dialog.showModal();
    input.focus();
    input.select();
  };
  const close = () => {
    if (dialog.open) dialog.close();
    if (priorFocus instanceof HTMLElement) priorFocus.focus();
  };

  const draw = () => {
    list.replaceChildren();
    matches.forEach((entry, position) => {
      const item = document.createElement("li");
      const link = document.createElement("a");
      const title = document.createElement("span");
      const excerpt = document.createElement("span");
      link.href = entry.url;
      link.dataset.active = String(position === active);
      link.tabIndex = position === active ? 0 : -1;
      title.className = "result-title";
      excerpt.className = "result-excerpt";
      title.textContent = `${entry.title} — ${entry.heading}`;
      excerpt.textContent = entry.excerpt;
      link.append(title, excerpt);
      item.append(link);
      list.append(item);
    });
    if (active >= 0) list.querySelectorAll("a")[active]?.focus();
  };

  const search = () => {
    const query = normalize(input.value);
    active = -1;
    if (!query) {
      matches = [];
      status.textContent = dialog.dataset.empty;
      draw();
      return;
    }
    const terms = query.split(/\s+/);
    matches = index.filter((entry) => terms.every((term) => entry.text.includes(term))).slice(0, 24);
    status.textContent = matches.length
      ? `${matches.length} ${matches.length === 1 ? dialog.dataset.resultSingular : dialog.dataset.resultPlural}.`
      : dialog.dataset.none;
    draw();
  };

  trigger.addEventListener("click", open);
  closeButton.addEventListener("click", close);
  input.addEventListener("input", search);
  dialog.addEventListener("click", (event) => { if (event.target === dialog) close(); });
  dialog.addEventListener("cancel", (event) => { event.preventDefault(); close(); });
  input.addEventListener("keydown", (event) => {
    if (!matches.length || !["ArrowDown", "ArrowUp", "Enter"].includes(event.key)) return;
    event.preventDefault();
    if (event.key === "ArrowDown") active = (active + 1) % matches.length;
    if (event.key === "ArrowUp") active = (active - 1 + matches.length) % matches.length;
    if (event.key === "Enter") {
      const choice = matches[active >= 0 ? active : 0];
      if (choice) window.location.href = choice.url;
      return;
    }
    draw();
  });
  list.addEventListener("keydown", (event) => {
    if (!["ArrowDown", "ArrowUp", "Enter"].includes(event.key)) return;
    event.preventDefault();
    if (event.key === "ArrowDown") active = (active + 1) % matches.length;
    if (event.key === "ArrowUp") active = (active - 1 + matches.length) % matches.length;
    if (event.key === "Enter" && active >= 0) window.location.href = matches[active].url;
    draw();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && dialog.open) { event.preventDefault(); close(); return; }
    if ((event.key === "/" && !editable(event.target)) || ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k")) {
      event.preventDefault();
      open();
    }
  });
  search();
})();
'''


class PageInspector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.h1_count = 0
        self.pre_code_count = 0
        self._in_pre = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(values["id"] or "")
        if tag == "h1":
            self.h1_count += 1
        if tag == "pre":
            self._in_pre = True
        if tag == "code" and self._in_pre:
            self.pre_code_count += 1
        for attribute in ("href", "src"):
            if values.get(attribute):
                self.links.append((attribute, values[attribute] or ""))

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre":
            self._in_pre = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_generated(
    pages: list[Page],
    resolvers: dict[str, Resolver],
    search_payloads: dict[str, dict],
) -> None:
    html_files = sorted(path.name for path in OUTPUT_ROOT.glob("*.html"))
    if html_files != sorted(HTML_MANIFEST):
        raise ValueError(f"HTML manifest mismatch: {html_files}")
    page_ids: dict[str, set[str]] = {}
    inspectors: dict[str, PageInspector] = {}
    for filename in HTML_MANIFEST:
        inspector = PageInspector()
        inspector.feed(read_text(OUTPUT_ROOT / filename))
        inspector.close()
        if inspector.h1_count != 1:
            raise ValueError(f"Expected exactly one h1 in {filename}, got {inspector.h1_count}")
        if len(inspector.ids) != len(set(inspector.ids)):
            raise ValueError(f"Duplicate id in {filename}")
        page_ids[filename] = set(inspector.ids)
        inspectors[filename] = inspector

    for filename, inspector in inspectors.items():
        for attribute, target in inspector.links:
            parts = urlsplit(target)
            if parts.scheme in {"http", "https", "mailto"}:
                continue
            if parts.scheme:
                raise ValueError(f"Unsupported generated scheme in {filename}: {target}")
            target_file = unquote(parts.path) if parts.path else filename
            target_path = OUTPUT_ROOT / target_file
            if not target_path.exists():
                raise ValueError(f"Broken local {attribute} in {filename}: {target}")
            if parts.fragment and target_file.endswith(".html"):
                if parts.fragment not in page_ids.get(Path(target_file).name, set()):
                    raise ValueError(f"Broken fragment in {filename}: {target}")

    selected_code = sum(1 for page in pages for block in iter_blocks(page.blocks) if block.kind == "code")
    generated_code = sum(inspector.pre_code_count for inspector in inspectors.values())
    if selected_code != generated_code:
        raise ValueError(f"Code fence count mismatch: selected={selected_code}, generated={generated_code}")

    expected_by_lang = {
        lang: {page.filename for page in pages if page.lang == lang}
        for lang in ("en", "ar")
    }
    for lang, payload in search_payloads.items():
        if payload.get("lang") != lang:
            raise ValueError(f"Search payload language mismatch: {lang}")
        entries = payload.get("entries", [])
        if {entry["url"].split("#", 1)[0] for entry in entries} != expected_by_lang[lang]:
            raise ValueError(f"Search index does not cover every {lang} page")
        for entry in entries:
            filename, fragment = entry["url"].split("#", 1)
            if fragment not in page_ids[filename]:
                raise ValueError(f"Search fragment missing: {entry['url']}")

    for output_name, source in {**resolvers["en"].downloads, **resolvers["en"].images}.items():
        target = OUTPUT_ROOT / output_name
        if not target.is_file() or sha256(target) != sha256(source):
            raise ValueError(f"Copied asset mismatch: {output_name}")

    combined = "\n".join(read_text(OUTPUT_ROOT / filename) for filename in HTML_MANIFEST)
    for needle in ("82.2", "88.3", "10.3", "80.9", "94.1", "95.6", "pos_r"):
        if needle not in combined:
            raise ValueError(f"Content spot check missing: {needle}")
    for subfile in sorted((REPO_ROOT / "condor").glob("*.sub")):
        anchor = subfile.stem.replace("_", "-") + "-sub"
        if f'id="{anchor}"' not in read_text(OUTPUT_ROOT / "walkthrough-condor.html"):
            raise ValueError(f"Condor heading missing: {subfile.name}")


def parser_self_test() -> None:
    fixture = """# Fixture

```python
# heading | table * emphasis [link] <tag>
```

| left | `a|b` | escaped \\| pipe |
|:---|:---:|---:|
| **bold** | value | 3 |

> quoted
> text

3. ordered
4. second

- unordered

Unknown <script>alert(1)</script> stays text.
"""
    blocks = parse_blocks(fixture, REPO_ROOT / "README.md")
    kinds = [block.kind for block in blocks]
    if kinds != ["heading", "code", "table", "blockquote", "list", "list", "paragraph"]:
        raise ValueError(f"Parser self-test block mismatch: {kinds}")
    if blocks[1].text != "# heading | table * emphasis [link] <tag>":
        raise ValueError("Code fence masking self-test failed")
    if blocks[2].rows[0][1] != "`a|b`" or len(blocks[2].rows[0]) != 3:
        raise ValueError("Table cell splitting self-test failed")
    if blocks[4].start != 3:
        raise ValueError("Ordered-list start self-test failed")


def assign_headings(
    pages: list[Page],
) -> tuple[dict[tuple[str, str], tuple[str, str]], dict[tuple[str, str], tuple[str, str]]]:
    routes: dict[tuple[str, str], tuple[str, str]] = {}
    global_routes: dict[tuple[str, str], tuple[str, str]] = {}
    for page in pages:
        used: dict[str, int] = {}
        for block in iter_blocks(page.blocks):
            if block.kind != "heading":
                continue
            if page.lang == "ar" and not block.explicit_id:
                raise ValueError(f"Arabic heading requires an explicit id in {relpath(block.source)}: {block.text}")
            special = re.fullmatch(r"`?condor/([A-Za-z0-9_]+)\.sub`?", block.text)
            base = block.explicit_id or (special.group(1).replace("_", "-") + "-sub" if special else slugify(block.text))
            count = used.get(base, 0)
            if block.explicit_id and count:
                raise ValueError(f"Duplicate explicit heading id {base} in {page.filename}")
            block.heading_id = base if count == 0 else f"{base}-{count}"
            used[base] = count + 1
            original_fragment = slugify(block.text)
            key = (relpath(block.source), original_fragment)
            routes.setdefault(key, (page.filename, block.heading_id))
            global_routes.setdefault((page.lang, block.heading_id), (page.filename, block.heading_id))
    return routes, global_routes


def page_source_routes(pages: list[Page]) -> dict[str, str]:
    routes = {
        "README.md": "index.html",
        "results/README.md": "results.html",
        "scripts/README.md": "reference.html",
    }
    for source in WALKTHROUGH_FILES:
        routes[relpath(source)] = walkthrough_output(source)
    return routes


def clean_generated() -> None:
    for filename in HTML_MANIFEST:
        path = OUTPUT_ROOT / filename
        if path.exists():
            path.unlink()
    if ASSETS_DIR.exists():
        shutil.rmtree(ASSETS_DIR)


def write_generated(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def build() -> None:
    parser_self_test()
    pages = build_pages()
    for page in pages:
        page.blocks = []
        for segment in page.segments:
            page.blocks.extend(parse_blocks(segment.markdown, segment.source_path, segment.link_base))
        if page.stem in CORE_STEMS:
            page.quizzes = load_core_quizzes(page)
        elif page.stem.startswith("walkthrough"):
            page.quizzes = load_walkthrough_quizzes(page)

    heading_routes, global_heading_routes = assign_headings(pages)
    for page in pages:
        if not page.stem.startswith("walkthrough"):
            continue
        content_h2_ids = [
            block.heading_id
            for block in iter_blocks(page.blocks)
            if block.kind == "heading" and block.level == 2
        ]
        if list(page.quizzes) != content_h2_ids:
            raise ValueError(
                f"Walkthrough quiz keys must match content h2 order in {page.filename}: "
                f"{list(page.quizzes)} != {content_h2_ids}"
            )
    page_pairs = {
        stem: {page.lang: page for page in pages if page.stem == stem}
        for stem in CORE_STEMS + ARABIC_WALKTHROUGH_STEMS
    }
    for stem, pair in page_pairs.items():
        if set(pair) != {"en", "ar"}:
            raise ValueError(f"Missing bilingual core pair: {stem}")
        en_ids = [(block.level, block.heading_id) for block in iter_blocks(pair["en"].blocks) if block.kind == "heading" and block.level in {1, 2, 3}]
        ar_ids = [(block.level, block.heading_id) for block in iter_blocks(pair["ar"].blocks) if block.kind == "heading" and block.level in {1, 2, 3}]
        if en_ids != ar_ids:
            raise ValueError(f"Arabic heading structure differs from English for {stem}: {en_ids} != {ar_ids}")
        if list(pair["en"].quizzes) != list(pair["ar"].quizzes):
            raise ValueError(f"Quiz section keys differ between languages for {stem}")
        for key in pair["en"].quizzes:
            en_ids = [question.qid for question in pair["en"].quizzes[key].questions]
            ar_ids = [question.qid for question in pair["ar"].quizzes[key].questions]
            if en_ids != ar_ids:
                raise ValueError(f"Quiz question ids differ between languages for {stem}#{key}")

    resolvers = {
        lang: Resolver(lang, page_source_routes(pages), heading_routes, global_heading_routes)
        for lang in ("en", "ar")
    }

    clean_generated()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    write_generated(ASSETS_DIR / "styles.css", STYLES_CSS.rstrip() + "\n")
    write_generated(ASSETS_DIR / "app.js", APP_JS.rstrip() + "\n")
    write_generated(ASSETS_DIR / "search.js", SEARCH_JS.rstrip() + "\n")
    fixture_json = json.dumps(normalization_fixtures(), ensure_ascii=False, indent=2, sort_keys=True)
    write_generated(
        ASSETS_DIR / "search-normalization-fixtures.js",
        "window.KD_NORMALIZATION_FIXTURES = " + fixture_json + ";\n",
    )

    search_entries: dict[str, list[dict[str, str]]] = {"en": [], "ar": []}
    for page in pages:
        write_generated(OUTPUT_ROOT / page.filename, render_page(page, pages, resolvers[page.lang]))
        search_entries[page.lang].extend(make_search_entries(page))

    search_payloads: dict[str, dict] = {}
    for lang in ("en", "ar"):
        payload = {"lang": lang, "entries": search_entries[lang]}
        search_payloads[lang] = payload
        search_json = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        write_generated(ASSETS_DIR / f"search-index.{lang}.json", search_json)
        write_generated(ASSETS_DIR / f"search-index.{lang}.js", "window.KD_SEARCH_INDEX = " + search_json.rstrip() + ";\n")

    for output_name, source in sorted(resolvers["en"].downloads.items()):
        target = OUTPUT_ROOT / output_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for output_name, source in sorted(resolvers["en"].images.items()):
        target = OUTPUT_ROOT / output_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    validate_generated(pages, resolvers, search_payloads)
    total_entries = sum(len(entries) for entries in search_entries.values())
    print(f"Built {len(pages)} pages and {total_entries} bilingual search sections in {OUTPUT_ROOT}")


if __name__ == "__main__":
    build()
