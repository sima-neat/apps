#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_text_splitters import MarkdownHeaderTextSplitter


DEFAULT_COLLECTION = "demo_collection"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def create_markdown_vectordb(
    input_path: str,
    output_db: str,
    embedding_model: str,
    collection: str = DEFAULT_COLLECTION,
) -> None:
    source = Path(input_path).expanduser().resolve()
    db_path = Path(output_db).expanduser().resolve()
    model_path = Path(embedding_model).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Markdown file does not exist: {source}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Embedding model directory does not exist: {model_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_text = source.read_text(encoding="utf-8", errors="ignore")
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
            ("####", "Header_4"),
        ]
    )
    documents = splitter.split_text(markdown_text)
    if not documents:
        raise ValueError(f"No Markdown chunks were created from: {source}")

    embedding = HuggingFaceEmbeddings(
        model_name=str(model_path),
        model_kwargs={"device": "cpu"},
    )
    Milvus.from_documents(
        documents=documents,
        embedding=embedding,
        collection_name=collection,
        connection_args={"uri": str(db_path)},
        index_params={"index_type": "FLAT", "metric_type": "IP"},
        drop_old=True,
    )

    metadata_path = db_path.with_suffix(".meta.json")
    metadata_path.write_text(
        json.dumps(
            {
                "input": str(source),
                "collection": collection,
                "embedding_model": str(model_path),
                "chunks": len(documents),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logging.info("Created %s with %d Markdown chunks", db_path, len(documents))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Milvus VectorDB from a Markdown file.")
    parser.add_argument("--input", required=True, help="Markdown input file")
    parser.add_argument("--output", required=True, help="Output milvus.db path")
    parser.add_argument("--embedding-model", required=True, help="Local embedding model directory")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Milvus collection name")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    create_markdown_vectordb(
        input_path=args.input,
        output_db=args.output,
        embedding_model=args.embedding_model,
        collection=args.collection,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
