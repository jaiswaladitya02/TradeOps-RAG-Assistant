from pdf_loader import load_pdf
from chunker import chunk_documents
from embeddings import LocalEmbeddingModel
from vector_store import create_vector_store


PDF_PATH = "data/raw/TradeOps_Enterprise_Manual_Improved.pdf"


def main():

    print("=" * 60)
    print("TRADE OPERATIONS RAG - DOCUMENT INGESTION")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load PDF
    # ---------------------------------------------------------

    print("\nLoading PDF...")

    documents = load_pdf(PDF_PATH)

    print(f"Documents loaded: {len(documents)}")

    # ---------------------------------------------------------
    # 2. Create chunks
    # ---------------------------------------------------------

    print("\nCreating chunks...")

    chunks = chunk_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    # ---------------------------------------------------------
    # 3. Load embedding model
    # ---------------------------------------------------------

    print("\nLoading embedding model...")

    embedding_model = LocalEmbeddingModel()

    print("Embedding model loaded!")

    # ---------------------------------------------------------
    # 4. Create vector store
    # ---------------------------------------------------------

    print("\nCreating vector store...")

    collection = create_vector_store(
        chunks=chunks,
        embedding_model=embedding_model,
        persist_directory="vector_store"
    )

    print("\nVector store created successfully!")

    print(f"Total chunks stored: {collection.count()}")

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()