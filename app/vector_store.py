import chromadb


def create_vector_store(
    chunks,
    embedding_model,
    persist_directory="vector_store"
):
    """
    Creates a fresh persistent ChromaDB vector store.

    Existing TradeOps collection is deleted before
    creating the new collection.

    Args:
        chunks: List of LangChain Document chunks.
        embedding_model: Local embedding model adapter.
        persist_directory: Directory where ChromaDB stores data.

    Returns:
        ChromaDB collection.
    """

    client = chromadb.PersistentClient(
        path=persist_directory
    )

    # Remove the existing collection so ingestion
    # can safely be rerun without duplicate IDs.
    try:
        client.delete_collection(
            name="tradeops"
        )
        print("Existing TradeOps collection deleted.")
    except Exception:
        print("No existing TradeOps collection found.")

    collection = client.create_collection(
        name="tradeops"
    )

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    metadatas = [
        chunk.metadata
        for chunk in chunks
    ]

    ids = [
        f"chunk_{index}"
        for index in range(len(chunks))
    ]

    print("Generating embeddings...")

    embeddings = embedding_model.embed_documents(
        texts
    )

    print("Embeddings generated.")

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print(
        f"Stored {len(chunks)} chunks in ChromaDB."
    )

    return collection