from app.embeddings import LocalEmbeddingModel
from app.retriever import get_chroma_collection, search_collection


def main():

    print("=" * 70)
    print("TRADEOPS RAG - RETRIEVAL DEBUG")
    print("=" * 70)

    query = "How are trade breaks investigated?"

    # ---------------------------------------------------------
    # 1. Load embedding model
    # ---------------------------------------------------------

    print("\nLoading embedding model...")

    embedding_model = LocalEmbeddingModel()

    print("Embedding model loaded!")

    # ---------------------------------------------------------
    # 2. Connect to ChromaDB
    # ---------------------------------------------------------

    collection = get_chroma_collection()

    print("\nCollection:")
    print(collection.name)

    print("Document count:")
    print(collection.count())

    # ---------------------------------------------------------
    # 3. Generate embedding
    # ---------------------------------------------------------

    print("\nGenerating query embedding...")

    query_embedding = embedding_model.embed_query(query)

    print("Embedding generated!")

    print(
        f"Embedding dimensions: {len(query_embedding)}"
    )

    # ---------------------------------------------------------
    # 4. RAW ChromaDB query
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("RAW CHROMADB QUERY")
    print("=" * 70)

    raw_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=15
    )

    raw_documents = raw_results["documents"][0]
    raw_metadatas = raw_results["metadatas"][0]
    raw_distances = raw_results["distances"][0]

    print(
        f"\nRaw results returned: "
        f"{len(raw_documents)}"
    )

    for i, (document, metadata, distance) in enumerate(
        zip(
            raw_documents,
            raw_metadatas,
            raw_distances
        ),
        start=1
    ):

        print(
            f"\nRAW RESULT {i}"
        )

        print(
            f"Page: {metadata.get('page')}"
        )

        print(
            f"Distance: {distance}"
        )

        print(
            f"Content preview: "
            f"{document[:150].replace(chr(10), ' ')}"
        )

    # ---------------------------------------------------------
    # 5. search_collection()
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SEARCH_COLLECTION RESULT")
    print("=" * 70)

    filtered_results = search_collection(
        collection=collection,
        query_embedding=query_embedding,
        top_k=5
    )

    documents = filtered_results["documents"][0]
    metadatas = filtered_results["metadatas"][0]
    distances = filtered_results["distances"][0]

    print(
        f"\nFiltered results returned: "
        f"{len(documents)}"
    )

    for i, (document, metadata, distance) in enumerate(
        zip(
            documents,
            metadatas,
            distances
        ),
        start=1
    ):

        print(
            f"\nFILTERED RESULT {i}"
        )

        print(
            f"Page: {metadata.get('page')}"
        )

        print(
            f"Distance: {distance}"
        )

        print(
            f"Content preview: "
            f"{document[:150].replace(chr(10), ' ')}"
        )

    # ---------------------------------------------------------
    # 6. Compare
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    print(
        f"\nRaw Chroma results: "
        f"{len(raw_documents)}"
    )

    print(
        f"search_collection results: "
        f"{len(documents)}"
    )

    if raw_documents:

        print(
            f"\nBest RAW result:"
        )

        print(
            f"Page: "
            f"{raw_metadatas[0].get('page')}"
        )

        print(
            f"Distance: "
            f"{raw_distances[0]}"
        )

    if documents:

        print(
            f"\nBest search_collection result:"
        )

        print(
            f"Page: "
            f"{metadatas[0].get('page')}"
        )

        print(
            f"Distance: "
            f"{distances[0]}"
        )


if __name__ == "__main__":
    main()