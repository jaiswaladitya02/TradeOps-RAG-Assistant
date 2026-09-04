from pathlib import Path

from pdf_loader import load_pdf
from chunker import chunk_documents
from embeddings import LocalEmbeddingModel
from vector_store import create_vector_store


RAW_DATA_DIR = Path("data/raw")
PERSIST_DIRECTORY = "vector_store"


def main():

    print("=" * 60)
    print("TRADE OPERATIONS RAG - DOCUMENT INGESTION")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Find all PDFs
    # ---------------------------------------------------------

    pdf_files = sorted(
        RAW_DATA_DIR.glob("*.pdf")
    )

    if not pdf_files:
        print("\nNo PDF files found.")
        print(f"Please place PDF files inside: {RAW_DATA_DIR}")
        return

    print(f"\nPDF files found: {len(pdf_files)}")

    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")

    # ---------------------------------------------------------
    # 2. Load all PDFs
    # ---------------------------------------------------------

    print("\nLoading PDFs...")

    all_documents = []

    for pdf_file in pdf_files:

        print(f"\nLoading: {pdf_file.name}")

        documents = load_pdf(
            str(pdf_file)
        )

        # Add source filename to metadata
        # so the RAG system can identify
        # which PDF the answer came from.
        for document in documents:
            document.metadata["source"] = pdf_file.name

        all_documents.extend(documents)

        print(
            f"Pages loaded: {len(documents)}"
        )

    print(
        f"\nTotal pages/documents loaded: "
        f"{len(all_documents)}"
    )

    # ---------------------------------------------------------
    # 3. Create chunks
    # ---------------------------------------------------------

    print("\nCreating chunks...")

    chunks = chunk_documents(
        all_documents
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    # ---------------------------------------------------------
    # 4. Load embedding model
    # ---------------------------------------------------------

    print("\nLoading embedding model...")

    embedding_model = LocalEmbeddingModel()

    print("Embedding model loaded!")

    # ---------------------------------------------------------
    # 5. Create vector store
    # ---------------------------------------------------------

    print("\nCreating vector store...")

    collection = create_vector_store(
        chunks=chunks,
        embedding_model=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )

    print("\nVector store created successfully!")

    print(
        f"Total chunks stored: "
        f"{collection.count()}"
    )

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()