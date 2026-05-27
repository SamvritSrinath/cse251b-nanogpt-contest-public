#!/usr/bin/env python3
"""Apply source-specific filters and normalize parquet rows for corpus-prep."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq


NormalizedRows = list[dict[str, Any]]
ProfileFn = Callable[[dict[str, Any], str, Path, int], NormalizedRows]


GUTENBERG_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
GUTENBERG_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter parquet corpora into normalized rows.")
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES), help="Filter profile name.")
    parser.add_argument("--input-dir", required=True, help="Recursive input parquet directory.")
    parser.add_argument("--output-dir", required=True, help="Output parquet directory.")
    parser.add_argument("--text-column", default="text", help="Input text column name.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Rows per parquet read batch.")
    return parser.parse_args()


def first_present(row: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value not in {None, ""}:
            return value
    return None


def base_row(
    row: dict[str, Any],
    *,
    text: str,
    source: str,
    input_path: Path,
    row_index: int,
    section: str | None = None,
) -> dict[str, Any]:
    doc_id = first_present(row, ["doc_id", "id", "paper_id", "book_id", "url"])
    title = first_present(row, ["title", "book_title", "paper_title"])
    out = {
        "text": text.strip(),
        "doc_id": str(doc_id) if doc_id is not None else f"{input_path.name}:{row_index}",
        "title": "" if title is None else str(title),
        "source": source,
    }
    if section is not None:
        out["section"] = section
    return out


def require_columns(columns: set[str], required: list[str], *, profile: str, path: Path) -> None:
    missing = [name for name in required if name not in columns]
    if missing:
        raise SystemExit(
            f"{profile}: {path} is missing required column(s): {', '.join(missing)}"
        )


def require_any_column(
    columns: set[str],
    choices: list[str],
    *,
    profile: str,
    path: Path,
) -> str:
    for name in choices:
        if name in columns:
            return name
    raise SystemExit(
        f"{profile}: {path} is missing one of required column(s): {', '.join(choices)}"
    )


def filter_fineweb_edu_hi(
    row: dict[str, Any],
    text_column: str,
    input_path: Path,
    row_index: int,
) -> NormalizedRows:
    text = str(row.get(text_column) or "")
    score_raw = first_present(row, ["score", "int_score", "quality_score"])
    score = float(score_raw)
    threshold = 3.0 if score > 1.0 else 0.7
    if len(text.strip()) < 200 or score < threshold:
        return []
    language = str(first_present(row, ["language", "lang"]) or "en").lower()
    if language not in {"en", "english"}:
        return []
    return [base_row(row, text=text, source="fineweb_edu_hi", input_path=input_path, row_index=row_index)]


def filter_dclm_clean(
    row: dict[str, Any],
    text_column: str,
    input_path: Path,
    row_index: int,
) -> NormalizedRows:
    text = str(row.get(text_column) or "")
    if len(text.strip()) < 200:
        return []
    language = str(first_present(row, ["language", "lang", "fasttext_language"]) or "en").lower()
    if language not in {"en", "eng", "english"}:
        return []
    score_raw = first_present(row, ["language_score", "quality_score", "score"])
    if score_raw is not None and float(score_raw) < 0.6:
        return []
    return [base_row(row, text=text, source="dclm_clean", input_path=input_path, row_index=row_index)]


def strip_gutenberg_boilerplate(text: str) -> str:
    start = GUTENBERG_START_RE.search(text)
    if start is not None:
        text = text[start.end() :]
    end = GUTENBERG_END_RE.search(text)
    if end is not None:
        text = text[: end.start()]
    lines = [
        line
        for line in text.splitlines()
        if not line.strip().lower().startswith(("produced by", "transcribed by"))
    ]
    return "\n".join(lines).strip()


def filter_pg19_books(
    row: dict[str, Any],
    text_column: str,
    input_path: Path,
    row_index: int,
) -> NormalizedRows:
    text = strip_gutenberg_boilerplate(str(row.get(text_column) or ""))
    if len(text) < 200:
        return []
    return [base_row(row, text=text, source="pg19_books", input_path=input_path, row_index=row_index)]


def s2orc_section_rows(
    row: dict[str, Any],
    text_column: str,
    input_path: Path,
    row_index: int,
) -> NormalizedRows:
    if isinstance(row.get("body_text"), list):
        out = []
        for section_index, section_row in enumerate(row["body_text"]):
            if not isinstance(section_row, dict):
                continue
            text = str(section_row.get("text") or "").strip()
            section = first_present(section_row, ["section", "section_title", "section_name"])
            if len(text) < 100:
                continue
            out.append(
                base_row(
                    row,
                    text=text,
                    source="s2orc_sections",
                    input_path=input_path,
                    row_index=row_index * 100_000 + section_index,
                    section="" if section is None else str(section),
                )
            )
        return out

    text = str(row.get(text_column) or "").strip()
    section = first_present(row, ["section", "section_title", "section_name"])
    if len(text) < 100:
        return []
    return [
        base_row(
            row,
            text=text,
            source="s2orc_sections",
            input_path=input_path,
            row_index=row_index,
            section="" if section is None else str(section),
        )
    ]


def filter_openwebmath(
    row: dict[str, Any],
    text_column: str,
    input_path: Path,
    row_index: int,
) -> NormalizedRows:
    text = str(row.get(text_column) or "")
    if len(text.strip()) < 120:
        return []
    return [base_row(row, text=text, source="openwebmath", input_path=input_path, row_index=row_index)]


PROFILES: dict[str, ProfileFn] = {
    "fineweb_edu_hi": filter_fineweb_edu_hi,
    "dclm_clean": filter_dclm_clean,
    "pg19_books": filter_pg19_books,
    "s2orc_sections": s2orc_section_rows,
    "openwebmath": filter_openwebmath,
}


def validate_profile_columns(profile: str, columns: set[str], text_column: str, path: Path) -> None:
    if profile == "fineweb_edu_hi":
        require_columns(columns, [text_column], profile=profile, path=path)
        require_any_column(columns, ["score", "int_score", "quality_score"], profile=profile, path=path)
    elif profile == "s2orc_sections":
        if "body_text" not in columns:
            require_columns(columns, [text_column], profile=profile, path=path)
            require_any_column(columns, ["section", "section_title", "section_name"], profile=profile, path=path)
    else:
        require_columns(columns, [text_column], profile=profile, path=path)


def write_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    include_section = any("section" in row for row in rows)
    normalized = []
    for row in rows:
        item = {
            "text": row["text"],
            "doc_id": row["doc_id"],
            "title": row["title"],
            "source": row["source"],
        }
        if include_section:
            item["section"] = row.get("section", "")
        normalized.append(item)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(normalized), output_path)


def filter_file(
    path: Path,
    *,
    input_root: Path,
    output_root: Path,
    profile: str,
    text_column: str,
    batch_size: int,
) -> int:
    parquet_file = pq.ParquetFile(path)
    columns = set(parquet_file.schema_arrow.names)
    validate_profile_columns(profile, columns, text_column, path)

    rows: list[dict[str, Any]] = []
    profile_fn = PROFILES[profile]
    row_index = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            rows.extend(profile_fn(row, text_column, path, row_index))
            row_index += 1

    relative = path.relative_to(input_root)
    output_path = output_root / relative
    write_rows(rows, output_path)
    return len(rows)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    if not input_root.is_dir():
        raise SystemExit(f"input directory not found: {input_root}")
    parquet_paths = sorted(input_root.rglob("*.parquet"))
    if not parquet_paths:
        raise SystemExit(f"no parquet files found under {input_root}")

    total_rows = 0
    for path in parquet_paths:
        total_rows += filter_file(
            path,
            input_root=input_root,
            output_root=output_root,
            profile=args.profile,
            text_column=args.text_column,
            batch_size=args.batch_size,
        )
    print(f"filter-parquet: wrote {total_rows} row(s) to {output_root}")


if __name__ == "__main__":
    main()
