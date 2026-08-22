from app.embeddings import LocalEmbeddingModel
from app.retriever import get_chroma_collection


def print_result(
    number,
    document,
    metadata,
    distance
):
    print("\n" + "=" * 70)
    print(f"RESULT {number}")
    print("=" * 70)

    print("\nMetadata:")
    print(metadata)

    print("\nDistance:")
    print(distance)

    print("\nContent:")
    print(document)


def run_debug(
    collection,
    embedding_model,
    query,
    top_k=10
):
    print("\n")
    print("=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    print("\nGenerating query embedding...")

    query_embedding = embedding_model.embed_query(
        query
    )

    print("Embedding generated.")

    print("\nRunning RAW ChromaDB search...")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print(
        f"\nRaw results returned: "
        f"{len(documents)}"
    )

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

        print_result(
            i,
            document,
            metadata,
            distance
        )


def main():

    print("=" * 70)
    print("TRADEOPS RAG DEBUG")
    print("=" * 70)

    print("\nLoading embedding model...")

    embedding_model = LocalEmbeddingModel()

    print("Embedding model loaded!")

    print("\nConnecting to ChromaDB...")

    collection = get_chroma_collection()

    print(
        f"Collection: {collection.name}"
    )

    print(
        f"Document count: {collection.count()}"
    )

    # ---------------------------------------------------------
    # Queries specifically targeting our failed evaluations
    # ---------------------------------------------------------

    queries = [

        "Why are market deadlines important?",

        "What are market deadlines?",

        "How do market deadlines affect trade operations?",

        "What happens when market deadlines are missed?",

        "Why is documenting root causes important when "
        "investigating trade breaks?",

        "Why do analysts document root causes?",

        "What is the purpose of documenting root causes?",

    ]

    for query in queries:

        run_debug(
            collection=collection,
            embedding_model=embedding_model,
            query=query,
            top_k=10
        )

    print("\n")
    print("=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()