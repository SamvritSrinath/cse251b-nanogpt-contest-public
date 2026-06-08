#!/usr/bin/env python3
"""Apply source-specific filters and normalize parquet rows for corpus-prep."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq


NormalizedRows = list[dict[str, Any]]
ProfileFn = Callable[[dict[str, Any], str, Path, int], NormalizedRows]
FILTER_OPTIONS: dict[str, Any] = {
    "min_chars": None,
    "max_chars": None,
    "line_quality_threshold": 0.5,
    "rewrite_low_quality_lines": False,
}
FILTER_STATS: dict[str, Any] = {
    "kept": 0,
    "dropped_by_reason": Counter(),
}


GUTENBERG_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
GUTENBERG_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
BOILERPLATE_RE = re.compile(
    r"(cookie policy|accept cookies|enable javascript|javascript is disabled|"
    r"privacy policy|terms of service|subscribe to continue|advertisement)",
    re.IGNORECASE,
)


def record_drop(reason: str) -> None:
    FILTER_STATS["dropped_by_reason"][reason] += 1


def heuristic_quality_filter(text: str, profile: str = "conservative_v1") -> str | None:
    """Return a conservative drop reason for low-quality text, or ``None`` to keep it."""

    if profile != "conservative_v1":
        raise ValueError(f"Unsupported heuristic quality filter profile: {profile}")

    stripped = text.strip()
    if len(stripped) < min_chars(80):
        return "too_short"

    non_space = [char for char in stripped if not char.isspace()]
    if non_space:
        alnum_ratio = sum(char.isalnum() for char in non_space) / len(non_space)
        if alnum_ratio < 0.35:
            return "low_alnum_ratio"

    lower = stripped.lower()
    if BOILERPLATE_RE.search(lower):
        if len(stripped) < 2_000 or lower.count("cookie") + lower.count("javascript") >= 2:
            return "boilerplate_cookie_javascript"

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) >= 6:
        counts = Counter(lines)
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count / len(lines) >= 0.5:
            return "repeated_lines"

    words = re.findall(r"\w+", lower)
    if len(words) >= 80:
        ngrams = [" ".join(words[index : index + 5]) for index in range(len(words) - 4)]
        counts = Counter(ngrams)
        if counts and counts.most_common(1)[0][1] >= 8:
            return "repeated_ngrams"

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter parquet corpora into normalized rows.")
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES), help="Filter profile name.")
    parser.add_argument("--input-dir", required=True, help="Recursive input parquet directory.")
    parser.add_argument("--output-dir", required=True, help="Output parquet directory.")
    parser.add_argument("--text-column", default="text", help="Input text column name.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Rows per parquet read batch.")
    parser.add_argument("--min-chars", type=int, default=None, help="Override profile minimum text length.")
    parser.add_argument("--max-chars", type=int, default=None, help="Optional maximum text length.")
    parser.add_argument(
        "--line-quality-threshold",
        type=float,
        default=0.5,
        help="Minimum mean line_quality for finerweb_line_quality.",
    )
    parser.add_argument(
        "--rewrite-low-quality-lines",
        action="store_true",
        help="For finerweb_line_quality, remove lines below --line-quality-threshold instead of dropping the row.",
    )
    return parser.parse_args()


def first_present(row: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value not in {None, ""}:
            return value
    return None


def min_chars(default: int) -> int:
    return int(FILTER_OPTIONS["min_chars"] or default)


def max_chars() -> int | None:
    value = FILTER_OPTIONS["max_chars"]
    return None if value is None else int(value)


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
    if len(text.strip()) < min_chars(256):
        return []
    language = str(first_present(row, ["language", "lang"]) or "en").lower()
    if language not in {"en", "english"}:
        return []
    language_score = first_present(row, ["language_score", "lang_score"])
    if language_score is not None and float(language_score) < 0.90:
        return []
    int_score = first_present(row, ["int_score", "edu_int_score"])
    score = first_present(row, ["score", "edu_score", "quality_score"])
    int_ok = int_score is not None and int(float(int_score)) >= 3
    score_ok = score is not None and float(score) >= 3.0
    if not (int_ok or score_ok):
        return []
    return [base_row(row, text=text, source="fineweb_edu_hi", input_path=input_path, row_index=row_index)]


def filter_dclm_edu_hi(
    row: dict[str, Any],
    text_column: str,
    input_path: Path,
    row_index: int,
) -> NormalizedRows:
    text = str(row.get(text_column) or "")
    limit = max_chars() or 100_000
    if len(text.strip()) < min_chars(256) or len(text) > limit:
        return []
    language = str(first_present(row, ["language", "lang"]) or "en").lower()
    if language not in {"en", "english"}:
        return []
    language_score = row.get("language_score")
    if language_score is not None and float(language_score) < 0.90:
        return []
    edu_int_score = row.get("edu_int_score")
    edu_score = row.get("edu_score")
    int_ok = edu_int_score is not None and int(float(edu_int_score)) >= 4
    score_ok = edu_score is not None and float(edu_score) >= 3.0
    if not (int_ok or score_ok):
        return []
    return [base_row(row, text=text, source="dclm_edu_hi", input_path=input_path, row_index=row_index)]


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
    if len(text) < min_chars(200):
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
            if len(text) < min_chars(100):
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
    if len(text) < min_chars(100):
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
    if len(text.strip()) < min_chars(120):
        return []
    return [base_row(row, text=text, source="openwebmath", input_path=input_path, row_index=row_index)]


def filter_finerweb_line_quality(
    row: dict[str, Any],
    text_column: str,
    input_path: Path,
    row_index: int,
) -> NormalizedRows:
    text = str(row.get(text_column) or "")
    if len(text.strip()) < min_chars(256):
        return []
    quality = row.get("line_quality")
    if not isinstance(quality, list) or not quality:
        return [base_row(row, text=text, source="finerweb_line_quality", input_path=input_path, row_index=row_index)]
    scores = [float(value) for value in quality if value is not None]
    if not scores:
        return []
    threshold = float(FILTER_OPTIONS["line_quality_threshold"])
    if bool(FILTER_OPTIONS["rewrite_low_quality_lines"]):
        lines = text.splitlines()
        kept = [line for line, score in zip(lines, scores) if score >= threshold]
        text = "\n".join(kept).strip()
        if len(text) < min_chars(256):
            return []
    elif sum(scores) / len(scores) < threshold:
        return []
    return [base_row(row, text=text, source="finerweb_line_quality", input_path=input_path, row_index=row_index)]


def filter_conservative_v1(
    row: dict[str, Any],
    text_column: str,
    input_path: Path,
    row_index: int,
) -> NormalizedRows:
    text = str(row.get(text_column) or "")
    reason = heuristic_quality_filter(text, profile="conservative_v1")
    if reason is not None:
        record_drop(reason)
        return []
    return [base_row(row, text=text, source="conservative_v1", input_path=input_path, row_index=row_index)]


PROFILES: dict[str, ProfileFn] = {
    "fineweb_edu_hi": filter_fineweb_edu_hi,
    "dclm_edu_hi": filter_dclm_edu_hi,
    "dclm_clean": filter_dclm_clean,
    "pg19_books": filter_pg19_books,
    "s2orc_sections": s2orc_section_rows,
    "openwebmath": filter_openwebmath,
    "finerweb_line_quality": filter_finerweb_line_quality,
    "conservative_v1": filter_conservative_v1,
}


def validate_profile_columns(profile: str, columns: set[str], text_column: str, path: Path) -> None:
    if profile == "fineweb_edu_hi":
        require_columns(columns, [text_column], profile=profile, path=path)
        require_any_column(columns, ["score", "int_score", "edu_score", "edu_int_score", "quality_score"], profile=profile, path=path)
    elif profile == "dclm_edu_hi":
        require_columns(columns, [text_column], profile=profile, path=path)
        require_any_column(columns, ["edu_int_score", "edu_score"], profile=profile, path=path)
    elif profile == "s2orc_sections":
        if "body_text" not in columns:
            require_columns(columns, [text_column], profile=profile, path=path)
            require_any_column(columns, ["section", "section_title", "section_name"], profile=profile, path=path)
    else:
        require_columns(columns, [text_column], profile=profile, path=path)


def rows_to_table(rows: list[dict[str, Any]], *, include_section: bool) -> pa.Table | None:
    if not rows:
        return None
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
    return pa.Table.from_pylist(normalized)


def filter_file(
    path: Path,
    *,
    input_root: Path,
    output_root: Path,
    profile: str,
    text_column: str,
    batch_size: int,
) -> int:
    print(f"filter-parquet: reading {path}", flush=True)
    parquet_file = pq.ParquetFile(path)
    columns = set(parquet_file.schema_arrow.names)
    validate_profile_columns(profile, columns, text_column, path)

    profile_fn = PROFILES[profile]
    row_index = 0
    written = 0
    batch_index = 0
    writer: pq.ParquetWriter | None = None
    relative = path.relative_to(input_root)
    output_path = output_root / relative
    include_section = profile == "s2orc_sections"
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        rows: list[dict[str, Any]] = []
        for row in batch.to_pylist():
            rows.extend(profile_fn(row, text_column, path, row_index))
            row_index += 1
        FILTER_STATS["kept"] += len(rows)
        table = rows_to_table(rows, include_section=include_section)
        if table is None:
            batch_index += 1
            if batch_index % 100 == 0:
                print(
                    f"filter-parquet: {path.name}: scanned {row_index:,} rows, kept {written:,}",
                    flush=True,
                )
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
        written += table.num_rows
        batch_index += 1
        if batch_index % 100 == 0:
            print(
                f"filter-parquet: {path.name}: scanned {row_index:,} rows, kept {written:,}",
                flush=True,
            )

    if writer is not None:
        writer.close()
    print(
        f"filter-parquet: wrote {written:,} row(s) from {path} to {output_path}",
        flush=True,
    )
    return written


def main() -> None:
    args = parse_args()
    FILTER_OPTIONS.update(
        {
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "line_quality_threshold": args.line_quality_threshold,
            "rewrite_low_quality_lines": args.rewrite_low_quality_lines,
        }
    )
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
    print(f"filter-parquet: kept={FILTER_STATS['kept']:,}")
    dropped_by_reason: Counter[str] = FILTER_STATS["dropped_by_reason"]
    if dropped_by_reason:
        print("filter-parquet: top drop reasons:")
        for reason, count in dropped_by_reason.most_common(10):
            print(f"  {reason}: {count:,}")


if __name__ == "__main__":
    main()
