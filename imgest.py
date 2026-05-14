# ingest.py
import argparse
import re
import sqlite3
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DB_PATH,
    EMBEDDING_MODEL,
    REPORT_PATH,
    VECTORSTORE_PATH,
)


def safe_name(value: str, fallback: str) -> str:
    name = re.sub(r"\W+", "_", value).lower().strip("_")
    return name or fallback


def convert_pdf_to_markdown(pdf_path: Path, markdown_path: Path) -> Path:
    from docling.document_converter import DocumentConverter

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    result = DocumentConverter().convert(str(pdf_path))
    markdown_path.write_text(result.document.export_to_markdown(), encoding="utf-8")
    print(f"Docling: converted {pdf_path} -> {markdown_path}")
    return markdown_path


def extract_tables_to_sqlite(markdown: str, db_path: str, source_name: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)

    pattern = re.compile(r"(?:#{1,3}\s*(.+?)\n)?((?:\|.+\n)+)", re.MULTILINE)
    saved, skipped = 0, 0
    source_prefix = safe_name(Path(source_name).stem, "source")

    for i, match in enumerate(pattern.finditer(markdown)):
        heading = (match.group(1) or f"table_{i}").strip()
        table_md = match.group(2)

        lines = [line.strip() for line in table_md.strip().split("\n") if line.strip()]
        lines = [line for line in lines if not re.match(r"^\|[\s\-|]+\|$", line)]
        rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]

        if len(rows) < 2:
            skipped += 1
            continue

        headers = [
            safe_name(col, f"col_{j}")
            for j, col in enumerate(rows[0])
        ]

        seen, deduped = {}, []
        for header in headers:
            seen[header] = seen.get(header, 0) + 1
            deduped.append(header if seen[header] == 1 else f"{header}_{seen[header]}")

        df = pd.DataFrame(rows[1:], columns=deduped)
        table_slug = safe_name(heading[:50], f"table_{i}")
        table_name = f"{source_prefix}_{table_slug}"
        df.to_sql(table_name, con, if_exists="replace", index=False)
        saved += 1

    con.close()
    print(f"SQLite: saved {saved} tables from {source_name}, skipped {skipped}")


def reset_sqlite(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    for (table_name,) in tables:
        escaped_name = table_name.replace('"', '""')
        con.execute(f'DROP TABLE IF EXISTS "{escaped_name}"')
    con.commit()
    con.close()


def markdown_to_documents(markdown: str, source_name: str) -> list[Document]:
    clean_text = re.sub(r"((?:\|.+\n)+)", "", markdown)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return [
        Document(page_content=chunk, metadata={"source": source_name})
        for chunk in splitter.split_text(clean_text)
    ]


def build_vectorstore(markdown_files: list[Path]) -> None:
    Path(VECTORSTORE_PATH).mkdir(parents=True, exist_ok=True)

    docs = []
    for markdown_file in markdown_files:
        text = markdown_file.read_text(encoding="utf-8")
        docs.extend(markdown_to_documents(text, markdown_file.name))

    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(docs, embedding)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"FAISS: saved {len(docs)} chunks to {VECTORSTORE_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PDFs to Markdown, then ingest Markdown into SQLite and FAISS."
    )
    parser.add_argument(
        "--pdf",
        action="append",
        type=Path,
        default=[],
        help="PDF file to convert with Docling. Can be passed multiple times.",
    )
    parser.add_argument(
        "--md",
        action="append",
        type=Path,
        default=[],
        help="Markdown file to ingest. Can be passed multiple times.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(REPORT_PATH).parent,
        help="Where converted Markdown files are written.",
    )
    parser.add_argument(
        "--db",
        default=DB_PATH,
        help="SQLite database path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    markdown_files = [path for path in args.md]

    for pdf_path in args.pdf:
        output_path = args.out_dir / f"{pdf_path.stem}.md"
        markdown_files.append(convert_pdf_to_markdown(pdf_path, output_path))

    if not markdown_files:
        markdown_files = [Path(REPORT_PATH)]

    print("Extracting tables to SQLite...")
    reset_sqlite(args.db)
    for markdown_file in markdown_files:
        text = markdown_file.read_text(encoding="utf-8")
        extract_tables_to_sqlite(text, args.db, markdown_file.name)

    print("Building vectorstore...")
    build_vectorstore(markdown_files)

    print("Ingestion complete.")


if __name__ == "__main__":
    main()