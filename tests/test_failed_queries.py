from app.embeddings import LocalEmbeddingModel
from app.retriever import get_chroma_collection, search_collection


FAILED_QUERIES = [
    "What causes a trade break?",
    "What is the role of clearing in trade operations?",
    "Why are market deadlines important?",
    "Why is documenting root causes important when investigating trade breaks?",
]


def main():

    print("=" * 70)
    print("TRADEOPS RAG - FAILED QUERY DIAGNOSTIC")
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
        f"Documents in collection: "
        f"{collection.count()}"
    )

    # ---------------------------------------------------------
    # Test each failed query
    # ---------------------------------------------------------

    for query_number, query in enumerate(
        FAILED_QUERIES,
        start=1
    ):

        print("\n\n")
        print("=" * 70)
        print(f"FAILED QUERY {query_number}")
        print("=" * 70)

        print("\nQuery:")
        print(query)

        # -----------------------------------------------------
        # Generate embedding
        # -----------------------------------------------------

        print("\nGenerating query embedding...")

        query_embedding = (
            embedding_model.embed_query(
                query
            )
        )

        print("Embedding generated!")

        # -----------------------------------------------------
        # Raw ChromaDB retrieval
        # -----------------------------------------------------

        print("\nSearching ChromaDB...")

        raw_results = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=10
        )

        documents = raw_results[
            "documents"
        ][0]

        metadatas = raw_results[
            "metadatas"
        ][0]

        distances = raw_results[
            "distances"
        ][0]

        print(
            f"\nRaw results returned: "
            f"{len(documents)}"
        )

        # -----------------------------------------------------
        # Display raw results
        # -----------------------------------------------------

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
            print(f"RAW RESULT {i}")
            print("-" * 70)

            print(
                f"Page: "
                f"{metadata.get('page', 'Unknown')}"
            )

            print(
                f"Distance: "
                f"{distance}"
            )

            print(
                f"Source: "
                f"{metadata.get('source', 'Unknown')}"
            )

            print("\nContent:")
            print(document)

        # -----------------------------------------------------
        # Current search_collection result
        # -----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("SEARCH_COLLECTION RESULTS")
        print("=" * 70)

        filtered_results = search_collection(
            collection=collection,
            query_embedding=query_embedding,
            top_k=5
        )

        filtered_documents = (
            filtered_results[
                "documents"
            ][0]
        )

        filtered_metadatas = (
            filtered_results[
                "metadatas"
            ][0]
        )

        filtered_distances = (
            filtered_results[
                "distances"
            ][0]
        )

        print(
            f"\nFiltered results returned: "
            f"{len(filtered_documents)}"
        )

        for i, (
            document,
            metadata,
            distance
        ) in enumerate(
            zip(
                filtered_documents,
                filtered_metadatas,
                filtered_distances
            ),
            start=1
        ):

            print("\n" + "-" * 70)
            print(f"FILTERED RESULT {i}")
            print("-" * 70)

            print(
                f"Page: "
                f"{metadata.get('page', 'Unknown')}"
            )

            print(
                f"Distance: "
                f"{distance}"
            )

            print("\nContent:")
            print(document)

    print("\n")
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()