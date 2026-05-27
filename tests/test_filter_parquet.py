from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "filter_parquet.py"


class FilterParquetTests(unittest.TestCase):
    def write_parquet(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path)

    def run_filter(
        self,
        profile: str,
        input_dir: Path,
        output_dir: Path,
        *,
        text_column: str = "text",
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--profile",
                profile,
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--text-column",
                text_column,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_pg19_profile_strips_gutenberg_boilerplate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = "This is the body text. " * 20
            text = (
                "Header\n*** START OF THE PROJECT GUTENBERG EBOOK TINY ***\n"
                "Produced by someone\n"
                f"{body}\n"
                "*** END OF THE PROJECT GUTENBERG EBOOK TINY ***\nFooter"
            )
            self.write_parquet(root / "in" / "pg.parquet", [{"text": text, "book_id": "book-1"}])

            result = self.run_filter("pg19_books", root / "in", root / "out")

            self.assertEqual(result.returncode, 0, result.stderr)
            table = pq.read_table(root / "out" / "pg.parquet")
            row = table.to_pylist()[0]
            self.assertEqual(row["doc_id"], "book-1")
            self.assertNotIn("PROJECT GUTENBERG", row["text"])
            self.assertNotIn("Produced by", row["text"])
            self.assertIn("This is the body text.", row["text"])

    def test_s2orc_profile_keeps_section_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_parquet(
                root / "in" / "s2.parquet",
                [
                    {
                        "text": "Methods content with enough words. " * 8,
                        "paper_id": "paper-1",
                        "title": "Paper One",
                        "section": "Methods",
                    }
                ],
            )

            result = self.run_filter("s2orc_sections", root / "in", root / "out")

            self.assertEqual(result.returncode, 0, result.stderr)
            row = pq.read_table(root / "out" / "s2.parquet").to_pylist()[0]
            self.assertEqual(row["doc_id"], "paper-1")
            self.assertEqual(row["title"], "Paper One")
            self.assertEqual(row["section"], "Methods")
            self.assertEqual(row["source"], "s2orc_sections")

    def test_s2orc_profile_fails_fast_without_section_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_parquet(
                root / "in" / "bad.parquet",
                [{"text": "Text without section metadata. " * 8, "paper_id": "paper-1"}],
            )

            result = self.run_filter("s2orc_sections", root / "in", root / "out")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing one of required column", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
