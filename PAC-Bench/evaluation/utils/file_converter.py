from pathlib import Path
import pandas as pd
import pypandoc
import re
import shutil


class FileConverter:
    def __init__(self, input_dir: str, output_dir: str, to_txt: bool = False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.to_txt = to_txt

        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_lines: list[str] = []

    def convert(self) -> None:
        # 1. Write directory tree
        self._write_directory_tree()
        self.report_lines.append("")

        # 2. Process files
        for src_path in sorted(self.input_dir.rglob("*")):
            if src_path.is_dir():
                continue

            rel_path = src_path.relative_to(self.input_dir)
            dst_path = self.output_dir / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            suffix = src_path.suffix.lower()

            if suffix == ".xlsx":
                self._handle_excel(src_path, dst_path.parent, rel_path)
            elif suffix == ".docx":
                self._handle_word(src_path, dst_path.parent, rel_path)
            else:
                shutil.copy2(src_path, dst_path)
                self._record_plain_file(rel_path, dst_path)

        if self.to_txt:
            self._all_files_to_txt(self.output_dir)

        self._write_report()

    # ---------- tree ----------

    def _write_directory_tree(self) -> None:
        self.report_lines.append("DIRECTORY STRUCTURE")
        self._build_tree(self.input_dir, prefix="")

    def _build_tree(self, base: Path, prefix: str) -> None:
        entries = sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name))
        for i, path in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            self.report_lines.append(f"{prefix}{connector}{path.name}")
            if path.is_dir():
                extension = "    " if i == len(entries) - 1 else "│   "
                self._build_tree(path, prefix + extension)

    # ---------- handlers ----------

    def _handle_excel(self, src: Path, result_dir: Path, rel_path: Path) -> None:
        xls = pd.ExcelFile(src)
        base_name = src.stem

        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            safe_sheet = re.sub(r"[^\w\-]+", "_", sheet_name).strip("_")
            out_file = result_dir / f"{base_name}_{safe_sheet}.csv"
            df.to_csv(out_file, index=False)

            self._begin_file_block(
                f"{rel_path} (sheet: {sheet_name}) (converted to csv)"
            )
            self._append_file_content(out_file)
            self._end_file_block()

    def _handle_word(self, src: Path, result_dir: Path, rel_path: Path) -> None:
        out_file = result_dir / f"{src.stem}.md"

        pypandoc.convert_file(
            source_file=str(src),
            to="md",
            outputfile=str(out_file),
            extra_args=["--standalone"]
        )

        self._begin_file_block(
            f"{rel_path} (converted to md)"
        )
        self._append_file_content(out_file)
        self._end_file_block()

    def _record_plain_file(self, rel_path: Path, dst_path: Path) -> None:
        self._begin_file_block(str(rel_path))
        self._append_file_content(dst_path)
        self._end_file_block()

    # ---------- report helpers ----------

    def _begin_file_block(self, header: str) -> None:
        self.report_lines.append("\n========== BEGIN FILE ==========")
        self.report_lines.append(f"[FILE] {header}")
        self.report_lines.append("========== CONTENT =============")

    def _end_file_block(self) -> None:
        self.report_lines.append("========== END FILE ============")

    def _append_file_content(self, path: Path) -> None:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                self.report_lines.append("(empty)")
            else:
                self.report_lines.append(text)
        except Exception as e:
            self.report_lines.append(f"[ERROR reading file: {e}]")

    # ---------- utils ----------

    def _all_files_to_txt(self, base_dir: Path) -> None:
        for path in base_dir.rglob("*"):
            if path.is_dir() or path.suffix == ".txt":
                continue
            path.rename(path.with_suffix(".txt"))

    def _write_report(self) -> None:
        report_path = self.output_dir / "conversion_report.txt"
        report_path.write_text("\n".join(self.report_lines), encoding="utf-8")


if __name__ == "__main__":
    converter = FileConverter(input_dir="shared", output_dir="shared_converted")
    converter.convert()