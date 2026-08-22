from app.embeddings import LocalEmbeddingModel
from app.retriever import get_chroma_collection, search_collection


def main():

    print("=" * 70)
    print("TRADEOPS RAG - RETRIEVAL QUALITY TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load embedding model
    # ---------------------------------------------------------

    print("\nLoading embedding model...")

    embedding_model = LocalEmbeddingModel()

    print("Embedding model loaded!")

    # ---------------------------------------------------------
    # 2. Connect to ChromaDB
    # ---------------------------------------------------------

    print("\nConnecting to ChromaDB...")

    collection = get_chroma_collection()

    print("Collection connected!")
    print(f"Documents in collection: {collection.count()}")

    # ---------------------------------------------------------
    # 3. Test query
    # ---------------------------------------------------------

    query = "How are trade breaks investigated?"

    print("\nQuery:")
    print(query)

    # ---------------------------------------------------------
    # 4. Generate query embedding
    # ---------------------------------------------------------

    print("\nGenerating query embedding...")

    query_embedding = embedding_model.embed_query(query)

    print("Query embedding generated!")

    # ---------------------------------------------------------
    # 5. Search ChromaDB
    # ---------------------------------------------------------

    print("\nSearching ChromaDB...")

    results = search_collection(
        collection=collection,
        query_embedding=query_embedding,
        top_k=5
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print(f"\nResults returned: {len(documents)}")

    # ---------------------------------------------------------
    # 6. Inspect every result
    # ---------------------------------------------------------

    for i, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1
    ):

        print("\n" + "=" * 70)
        print(f"RESULT {i}")
        print("=" * 70)

        print("\nMetadata:")
        print(metadata)

        print("\nDistance:")
        print(distance)

        print("\nContent:")
        print(document)

    # ---------------------------------------------------------
    # 7. Check for duplicate content
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("DUPLICATE CHECK")
    print("=" * 70)

    unique_documents = set(documents)

    print(f"Total retrieved chunks: {len(documents)}")
    print(f"Unique chunks: {len(unique_documents)}")

    duplicates = len(documents) - len(unique_documents)

    print(f"Duplicate chunks: {duplicates}")

    if duplicates > 0:
        print("\nWARNING:")
        print("Some retrieved chunks are identical.")

    else:
        print("\nNo identical chunks detected.")


if __name__ == "__main__":
    main()