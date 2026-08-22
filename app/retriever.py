import re
import chromadb


def get_chroma_collection(
    persist_directory="vector_store"
):
    """
    Connects to the persistent ChromaDB
    collection used by the TradeOps RAG system.
    """

    client = chromadb.PersistentClient(
        path=persist_directory
    )

    collection = client.get_collection(
        name="tradeops"
    )

    return collection


def search_collection(
    collection,
    query_embedding,
    top_k=5
):
    """
    Searches ChromaDB using the query embedding.

    Retrieves extra candidates so exact duplicate chunks
    can be removed without reducing the number of useful
    results.

    Returns results in original Chroma relevance order.
    """

    candidate_count = max(
        top_k * 3,
        10
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    result_ids = results.get(
        "ids",
        [[]]
    )[0]

    seen_documents = set()

    unique_documents = []
    unique_metadatas = []
    unique_distances = []
    unique_ids = []

    for index, document in enumerate(documents):

        normalized_document = re.sub(
            r"\s+",
            " ",
            document.strip().lower()
        )

        if normalized_document in seen_documents:
            continue

        seen_documents.add(
            normalized_document
        )

        unique_documents.append(
            document
        )

        unique_metadatas.append(
            metadatas[index]
        )

        unique_distances.append(
            distances[index]
        )

        if result_ids:
            unique_ids.append(
                result_ids[index]
            )

        if len(unique_documents) >= top_k:
            break

    filtered_results = {
        "documents": [
            unique_documents
        ],
        "metadatas": [
            unique_metadatas
        ],
        "distances": [
            unique_distances
        ]
    }

    if result_ids:
        filtered_results["ids"] = [
            unique_ids
        ]

    return filtered_results