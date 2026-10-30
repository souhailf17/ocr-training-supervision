"""Resolve OCR annotation table references without changing the source annotation.

The result is Markdown with complete inline HTML tables. Markdown placement is the
reading order; blocks validate that order when their metadata is available. This
module never fetches a URL, reads a prediction, or runs annotation content.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from html.parser import HTMLParser

TARGET_FORMAT = "markdown-inline-html-tables-v2"
MANIFEST_VERSION = "ocr-manifest-v2-tables"
TABLE_OCR_PROMPT = (
    "Transcribe all visible text in this image into Markdown, preserving its language, "
    "mathematical notation and reading order. Include every table at its position as "
    "a complete inline HTML <table>, with all headers, rows, cells and merged-cell spans. "
    "Do not replace tables with file links or placeholders. Do not translate, summarize, "
    "explain or invent content. If the page is blank, output nothing."
)
TABLE_ID = re.compile(r"(?:tbl|table)[-_]\d+\.html", re.I)
TABLE_LINK = re.compile(
    r"(?P<image>!?)\[(?P<label>[^\]\r\n]*)\]\(\s*<?(?P<id>(?:\./)?(?:tbl|table)[-_]\d+\.html)>?\s*\)",
    re.I,
)


class TableTargetError(ValueError):
    """An ambiguous or incomplete reference must be quarantined, never omitted."""


@dataclass(frozen=True)
class ResolvedTarget:
    text: str
    audit: dict


class TableHTML(HTMLParser):
    """Strict table structure and cell text for validation/scoring, not a renderer."""
    VOID = {"br", "hr", "img", "col", "wbr"}
    FORBIDDEN = {"script", "style", "iframe", "object", "embed"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.tables = []
        self.table = None
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag in self.FORBIDDEN:
            raise TableTargetError("active_or_hidden_html:" + tag)
        attributes = dict(attrs)
        if tag == "table":
            if self.table is not None:
                raise TableTargetError("nested_table_requires_explicit_policy")
            self.table = {"rows": [], "caption": ""}
            self.tables.append(self.table)
        elif tag == "tr":
            if self.table is None or self.row is not None:
                raise TableTargetError("row_outside_table_or_unclosed")
            self.row = []
            self.table["rows"].append(self.row)
        elif tag in {"td", "th"}:
            if self.row is None or self.cell is not None:
                raise TableTargetError("cell_outside_row_or_unclosed")
            spans = {}
            for name in ("rowspan", "colspan"):
                value = attributes.get(name, "1")
                minimum=0 if name=='rowspan' else 1  # HTML rowspan=0 means the remaining rows.
                if not isinstance(value, str) or not value.isdecimal() or not minimum <= int(value) <= 1000:
                    raise TableTargetError("invalid_" + name)
                spans[name] = int(value)
            self.cell = {"tag": tag, "text": "", **spans}
            self.row.append(self.cell)
        if tag == "br" and self.cell is not None:
            self.cell["text"] += " "
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            raise TableTargetError("unbalanced_html:" + tag)
        self.stack.pop()
        if tag in {"td", "th"}:
            self.cell = None
        elif tag == "tr":
            self.row = None
        elif tag == "table":
            self.table = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell["text"] += data
        elif self.table is not None and "caption" in self.stack:
            self.table["caption"] += data


def parse_tables(text: str) -> list[dict]:
    parser = TableHTML()
    parser.feed(text)
    parser.close()
    if parser.stack:
        raise TableTargetError("unclosed_html:" + parser.stack[-1])
    for table in parser.tables:
        if not table["rows"] or not any(table["rows"]):
            raise TableTargetError("table_has_no_cells")
    return parser.tables


def _code_ranges(text):
    ranges = []
    opened = None
    offset = 0
    for line in text.splitlines(keepends=True):
        fence = re.match(r"^[ ]{0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if fence:
            marker = fence.group(1)
            if opened is None:
                opened = (offset, marker[0], len(marker))
            elif marker[0] == opened[1] and len(marker) >= opened[2] and not fence.group(2).strip():
                ranges.append((opened[0], offset + len(line)))
                opened = None
        offset += len(line)
    if opened is not None:
        ranges.append((opened[0], len(text)))
    # Protect inline code as well. Fenced-code ranges already take precedence.
    at = 0
    while at < len(text):
        start = re.search(r"`+", text[at:])
        if not start:
            break
        first, last = at + start.start(), at + start.end()
        width = last - first
        close = re.search(r"(?<!`)`{" + str(width) + r"}(?!`)", text[last:])
        if close:
            end = last + close.end()
            ranges.append((first, end))
            at = end
        else:
            at = last
    return ranges


def table_references(text: str):
    protected = _code_ranges(text)
    for match in TABLE_LINK.finditer(text):
        before = text[:match.start()]
        escaped = (len(before) - len(before.rstrip("\\"))) % 2
        if escaped or any(a <= match.start() < b for a,b in protected):
            continue
        yield match


def assert_no_table_placeholders(text: str):
    ids = [m.group("id") for m in table_references(text)]
    if ids:
        raise TableTargetError("unresolved_table_references:" + ",".join(ids))


def resolve_annotation(annotation: dict) -> ResolvedTarget:
    if not isinstance(annotation, dict):
        raise TableTargetError("annotation_is_not_an_object")
    markdown = annotation.get("markdown")
    if not isinstance(markdown, str) or not markdown.strip():
        raise TableTargetError("empty_or_missing_markdown")
    tables = annotation.get("tables", [])
    if not isinstance(tables, list):
        raise TableTargetError("tables_is_not_a_list")
    by_id = {}
    structures = {}
    for table in tables:
        if not isinstance(table, dict):
            raise TableTargetError("table_is_not_an_object")
        ident = table.get("id")
        if not isinstance(ident, str) or not TABLE_ID.fullmatch(ident):
            raise TableTargetError("invalid_table_id")
        if ident in by_id:
            raise TableTargetError("duplicate_table_id:" + ident)
        content = table.get("content")
        if table.get("format", "html") != "html" or not isinstance(content, str) or not content.strip():
            raise TableTargetError("missing_or_non_html_table:" + ident)
        parsed = parse_tables(content)
        if len(parsed) != 1:
            raise TableTargetError("expected_one_table:" + ident)
        by_id[ident] = content
        structures[ident] = parsed[0]
    pieces = []
    order = []
    positions = {}
    end = 0
    for match in table_references(markdown):
        ident = match.group("id").removeprefix("./")
        if ident not in by_id:
            raise TableTargetError("missing_table_id:" + ident)
        if ident in order:
            raise TableTargetError("duplicate_table_reference:" + ident)
        pieces.append(markdown[end:match.start()])
        positions[ident]=sum(len(piece) for piece in pieces)
        pieces.append(by_id[ident])
        order.append(ident)
        end = match.end()
    pieces.append(markdown[end:])
    text = "".join(pieces)
    # Already inlined annotations are supported without duplicating their tables.
    expected_counts=Counter(by_id.values())
    inline_count={content:markdown.count(content) for content in expected_counts}
    for ident, content in by_id.items():
        if ident not in order and inline_count[content] > 0:
            at=0
            while True:
                at=text.find(content,at)
                if at<0:raise TableTargetError("ambiguous_inline_table:" + ident)
                if at not in positions.values():break
                at+=len(content)
            positions[ident]=at
            order.append(ident)
            inline_count[content] -= 1
        if text.count(content) != expected_counts[content]:
            raise TableTargetError("unreferenced_or_duplicated_table:" + ident)
    if by_id:
        order.sort(key=lambda ident: positions[ident])
    assert_no_table_placeholders(text)
    blocks = annotation.get("blocks", [])
    if not isinstance(blocks, list):
        raise TableTargetError("blocks_is_not_a_list")
    block_matches = defaultdict(list)
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "table":
            continue
        ident = [ident for ident, content in by_id.items() if content == block.get("content")]
        if not ident:
            raise TableTargetError("table_block_missing_or_ambiguous_in_tables")
        index = block.get("reading_index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise TableTargetError("table_block_without_reading_index")
        block_matches[block['content']].append(block)
    for content,matches in block_matches.items():
        if len(matches)>expected_counts[content]:
            raise TableTargetError('duplicate_table_block')
        matches.sort(key=lambda b:b['reading_index'])
    table_audit = []
    reading_order = []
    for ident in order:
        table = structures[ident]
        cell_count = sum(len(row) for row in table["rows"])
        audit = {"id": ident, "rows": len(table["rows"]), "cells": cell_count,
                 "content_chars": len(by_id[ident]),
                 "content_sha256": hashlib.sha256(by_id[ident].encode()).hexdigest()}
        matches = block_matches[by_id[ident]]
        if matches:
            block = matches.pop(0)
            audit["reading_index"] = block["reading_index"]
            audit["bbox"] = [block.get(k) for k in ("top_left_x","top_left_y","bottom_right_x","bottom_right_y")]
            reading_order.append(block["reading_index"])
        table_audit.append(audit)
    if len(reading_order) != len(set(reading_order)) or reading_order != sorted(reading_order):
        raise TableTargetError("markdown_table_order_conflicts_with_blocks")
    return ResolvedTarget(text, {"target_format": TARGET_FORMAT, "table_count": len(by_id),
                                "cell_count": sum(t["cells"] for t in table_audit),
                                "resolved_table_ids": order, "tables": table_audit,
                                "source_markdown_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
                                "blocks_order_checked": len(reading_order) == len(order),
                                "reading_order_policy": "Replace at each Markdown reference; do not reorder columns or append blocks twice."})
