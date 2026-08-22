from app.embeddings import LocalEmbeddingModel
from app.retriever import get_chroma_collection, search_collection


def main():

    print("=" * 60)
    print("TRADEOPS RAG - RETRIEVAL TEST")
    print("=" * 60)

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

    print(f"Collection connected!")
    print(f"Documents in collection: {collection.count()}")

    # ---------------------------------------------------------
    # 3. Create query
    # ---------------------------------------------------------

    query = "How are trade breaks investigated?"

    print(f"\nQuery:")
    print(query)

    # ---------------------------------------------------------
    # 4. Convert query into an embedding
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

    print(f"Results returned: {len(results['documents'][0])}")

    # ---------------------------------------------------------
    # 6. Display results
    # ---------------------------------------------------------

    for i, document in enumerate(
        results["documents"][0],
        start=1
    ):

        print("\n" + "=" * 60)
        print(f"RESULT {i}")
        print("=" * 60)

        print(document)

        print("\nMetadata:")
        print(results["metadatas"][0][i - 1])

        print("\nDistance:")
        print(results["distances"][0][i - 1])


if __name__ == "__main__":
    main()