from app.embeddings import LocalEmbeddingModel
from app.retriever import get_chroma_collection


def main():

    print("=" * 70)
    print("TRADEOPS - MARKET DEADLINE RETRIEVAL DEBUG")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load embedding model
    # ---------------------------------------------------------

    print("\nLoading embedding model...")

    embedding_model = LocalEmbeddingModel()

    print("Embedding model loaded!")

    # ---------------------------------------------------------
    # Connect to ChromaDB
    # ---------------------------------------------------------

    print("\nConnecting to ChromaDB...")

    collection = get_chroma_collection()

    print("Collection connected!")
    print(
        f"Documents in collection: {collection.count()}"
    )

    # ---------------------------------------------------------
    # Query
    # ---------------------------------------------------------

    query = "Why are market deadlines important?"

    print("\nQuery:")
    print(query)

    # ---------------------------------------------------------
    # Generate embedding
    # ---------------------------------------------------------

    print("\nGenerating query embedding...")

    query_embedding = embedding_model.embed_query(
        query
    )

    print("Embedding generated!")

    print(
        f"Embedding dimensions: "
        f"{len(query_embedding)}"
    )

    # ---------------------------------------------------------
    # RAW CHROMA SEARCH
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("RAW CHROMADB RESULTS")
    print("=" * 70)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (
        document,
        metadata,
        distance
    ) in enumerate(
        zip(
            documents,
            metadatas,
            distances
        ),
        start=1
    ):

        print("\n" + "-" * 70)

        print(f"RESULT {i}")

        print("-" * 70)

        print(
            f"Page: "
            f"{metadata.get('page', 'Unknown')}"
        )

        print(
            f"Distance: {distance}"
        )

        print("\nContent:")

        print(document)

    print("\n" + "=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()