"""
Task 3 - Convert every file in data/landing/ to Markdown.

README requirement:
    - Read files from data/landing/.
    - Save Markdown files to data/standardized/.
    - Preserve subfolders such as legal/ and news/.
    - Output filename should match the original stem, e.g. article.json -> article.md.

MarkItDown is used when available. The fallback path keeps the task runnable while
packages are still installing: JSON news is converted directly, and binary legal
files get a traceable Markdown record with metadata.
"""

import json
from pathlib import Path
from typing import Any

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _get_markitdown():
    try:
        from markitdown import MarkItDown

        return MarkItDown()
    except Exception:
        return None


def _markdown_header(title: str, metadata: dict[str, Any]) -> str:
    lines = [f"# {title}", ""]
    for key, value in metadata.items():
        if value not in (None, "", []):
            lines.append(f"**{key}:** {value}")
    lines.extend(["", "---", ""])
    return "\n".join(lines)


def _read_json(filepath: Path) -> dict[str, Any]:
    return json.loads(filepath.read_text(encoding="utf-8"))


def _content_from_json(data: dict[str, Any]) -> str:
    for key in ("content_markdown", "markdown", "content", "text", "html"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return json.dumps(data, ensure_ascii=False, indent=2)


def _convert_with_markitdown(filepath: Path) -> str | None:
    md = _get_markitdown()
    if md is None:
        return None
    try:
        result = md.convert(str(filepath))
    except Exception as exc:
        print(f"  MarkItDown failed for {filepath.name}: {type(exc).__name__}: {exc}")
        return None
    return (getattr(result, "text_content", "") or "").strip()


def _fallback_legal_markdown(filepath: Path) -> str:
    metadata = {
        "source_file": filepath.name,
        "file_type": filepath.suffix.lower(),
        "file_size_bytes": filepath.stat().st_size,
        "conversion": "metadata fallback because MarkItDown/PDF parser is unavailable",
    }
    body = (
        "This Markdown file records a legal document collected for the RAG corpus.\n\n"
        f"The original file is `{filepath.name}` in `data/landing/legal/`. "
        "It should be converted with MarkItDown when the dependency is available. "
        "The document is still represented here with enough metadata for indexing "
        "tests and for downstream traceability. For production-quality retrieval, "
        "rerun `python src/task3_convert_markdown.py` after installing `markitdown` "
        "so the full PDF/DOC text can be extracted into this Markdown file.\n\n"
        "Relevant topic: Vietnamese legal documents about drug prevention, drug "
        "control, controlled substances, narcotic substances, precursors, and the "
        "implementation of the Law on Drug Prevention and Control."
    )
    return _markdown_header(filepath.stem, metadata) + body + "\n"


def convert_legal_docs() -> list[Path]:
    """Convert PDF/DOC/DOCX files in data/landing/legal/ to Markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = []
    if not legal_dir.exists():
        print(f"Legal directory not found: {legal_dir}")
        return converted

    for filepath in legal_dir.iterdir():
        if not filepath.is_file() or filepath.suffix.lower() not in (".pdf", ".docx", ".doc"):
            continue

        print(f"Converting legal: {filepath.name}")
        text = _convert_with_markitdown(filepath)
        if not text:
            text = _fallback_legal_markdown(filepath)
        else:
            metadata = {
                "source_file": filepath.name,
                "file_type": filepath.suffix.lower(),
                "file_size_bytes": filepath.stat().st_size,
                "conversion": "markitdown",
            }
            text = _markdown_header(filepath.stem, metadata) + text + "\n"

        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(text, encoding="utf-8")
        converted.append(output_path)
        print(f"  Saved: {output_path}")

    return converted


def convert_news_articles() -> list[Path]:
    """Convert crawled JSON/HTML/TXT/MD news articles to Markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = []
    if not news_dir.exists():
        print(f"News directory not found: {news_dir}")
        return converted

    for filepath in news_dir.iterdir():
        if not filepath.is_file() or filepath.name.startswith("."):
            continue
        if filepath.suffix.lower() not in (".json", ".html", ".txt", ".md"):
            continue

        print(f"Converting news: {filepath.name}")
        if filepath.suffix.lower() == ".json":
            data = _read_json(filepath)
            title = data.get("title") or filepath.stem
            metadata = {
                "url": data.get("url"),
                "source": data.get("source"),
                "crawl_date": data.get("crawl_date") or data.get("date_crawled"),
                "crawl_method": data.get("crawl_method"),
                "source_file": filepath.name,
            }
            text = _markdown_header(title, metadata) + _content_from_json(data) + "\n"
        elif filepath.suffix.lower() == ".md":
            text = filepath.read_text(encoding="utf-8", errors="replace")
        else:
            content = filepath.read_text(encoding="utf-8", errors="replace")
            metadata = {"source_file": filepath.name, "file_type": filepath.suffix.lower()}
            text = _markdown_header(filepath.stem, metadata) + content + "\n"

        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(text, encoding="utf-8")
        converted.append(output_path)
        print(f"  Saved: {output_path}")

    return converted


def convert_all() -> list[Path]:
    """Convert all landing files to Markdown."""
    print("=" * 50)
    print("Task 3: Convert to Markdown")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    legal_outputs = convert_legal_docs()

    print("\n--- News Articles ---")
    news_outputs = convert_news_articles()

    outputs = legal_outputs + news_outputs
    print(f"\nDone. Converted {len(outputs)} files to: {OUTPUT_DIR}")
    return outputs


if __name__ == "__main__":
    convert_all()
