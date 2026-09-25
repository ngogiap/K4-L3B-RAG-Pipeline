"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from .contracts import validate_document


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

COLLECTION_NAME = "rag_documents"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed text bằng provider cấu hình trong .env."""
    if not texts:
        return []

    load_dotenv()
    provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").strip()
    model_name = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL).strip() or EMBEDDING_MODEL

    if provider == "sentence_transformers":
        from sentence_transformers import SentenceTransformer

        if not hasattr(embed_texts, "_sentence_model"):
            embed_texts._sentence_model = SentenceTransformer(model_name)  # type: ignore[attr-defined]
        model = embed_texts._sentence_model  # type: ignore[attr-defined]
        return model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=True,
        ).tolist()

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI()
        vectors: list[list[float]] = []
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = client.embeddings.create(model=model_name, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER={provider!r}")


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        relative_path = path.relative_to(STANDARDIZED_DIR)
        doc_type = "legal" if "legal" in relative_path.parts else "news"
        document = {
            "id": relative_path.as_posix(),
            "content": content,
            "metadata": {
                "source": path.name,
                "title": path.stem.replace("_", " ").replace("-", " ").strip(),
                "doc_type": doc_type,
                "url": None,
            },
        }
        validate_document(document)
        documents.append(document)
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    for document in documents:
        validate_document(document)
        texts = [text.strip() for text in splitter.split_text(document["content"])]
        for index, text in enumerate(text for text in texts if text):
            chunk = {
                "id": f"{document['id']}::chunk-{index}",
                "content": text,
                "metadata": {**document["metadata"], "chunk_index": index},
            }
            validate_document(chunk, require_chunk=True)
            chunks.append(chunk)
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    if not chunks:
        return []

    vectors = embed_texts([chunk["content"] for chunk in chunks])
    embedded_chunks = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        validate_document(chunk, require_chunk=True)
        embedded_chunks.append({**chunk, "embedding": vector})
    return embedded_chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    if not chunks:
        return

    for chunk in chunks:
        validate_document(chunk, require_chunk=True)
        if "embedding" not in chunk:
            raise ValueError(f"chunk {chunk['id']} is missing embedding")

    collection = get_collection()
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks")


if __name__ == "__main__":
    run_pipeline()
