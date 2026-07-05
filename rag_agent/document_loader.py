"""
document_loader.py
-------------------
Loads and normalizes text content from heterogeneous document sources:
PDF, Word (.docx), HTML, and plain text.

Each loader returns a `Document` dataclass with normalized text plus
source metadata, so downstream chunking/retrieval never has to care
what format the content originally came from.
"""

from __future__ import annotations

import logging
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Normalized representation of any loaded source document."""

    source_path: str
    source_type: str  # "pdf" | "docx" | "html" | "text"
    text: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Document(type={self.source_type}, path={self.source_path}, text='{preview}...')"


def _load_pdf(path: Path) -> Document:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages_text.append(text)

    full_text = "\n\n".join(pages_text)
    return Document(
        source_path=str(path),
        source_type="pdf",
        text=full_text,
        metadata={"num_pages": len(reader.pages)},
    )


def _load_docx(path: Path) -> Document:
    import docx  # python-docx

    doc = docx.Document(str(path))
    parts = []

    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)

    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells)
            if row_text.strip(" |"):
                parts.append(row_text)

    full_text = "\n".join(parts)
    return Document(
        source_path=str(path),
        source_type="docx",
        text=full_text,
        metadata={"num_paragraphs": len(doc.paragraphs)},
    )


def _load_html(path: Path) -> Document:
    from bs4 import BeautifulSoup

    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "lxml")

    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    full_text = "\n".join(lines)

    return Document(
        source_path=str(path),
        source_type="html",
        text=full_text,
        metadata={"title": title},
    )


def _load_text(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return Document(source_path=str(path), source_type="text", text=raw, metadata={})


_LOADERS = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".htm": _load_html,
    ".html": _load_html,
    ".txt": _load_text,
    ".md": _load_text,
}


def load_document(path: str | Path) -> Document:
    """Load a single document, auto-detecting type from its extension."""
    path = Path(path)
    suffix = path.suffix.lower()

    loader = _LOADERS.get(suffix)
    if loader is None:
        mime, _ = mimetypes.guess_type(str(path))
        raise ValueError(
            f"Unsupported file type '{suffix}' (guessed mime: {mime}) for {path}. "
            f"Supported: {sorted(_LOADERS.keys())}"
        )

    logger.info("Loading %s as %s", path, suffix)
    doc = loader(path)

    if not doc.text.strip():
        logger.warning("Document %s produced no extractable text", path)

    return doc


def load_directory(directory: str | Path, recursive: bool = True) -> Iterable[Document]:
    """Load every supported document found in a directory."""
    directory = Path(directory)
    pattern = "**/*" if recursive else "*"

    for path in sorted(directory.glob(pattern)):
        if path.is_file() and path.suffix.lower() in _LOADERS:
            try:
                yield load_document(path)
            except Exception as exc:  # noqa: BLE001 - log and continue ingesting the rest
                logger.error("Failed to load %s: %s", path, exc)
