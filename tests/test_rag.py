import re

from app.embeddings import LocalEmbeddingModel
from app.retriever import get_chroma_collection, search_collection
from app.llm import generate_response


def main():

    print("=" * 60)
    print("TRADEOPS RAG - END-TO-END TEST")
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

    print("Collection connected!")
    print(f"Documents in collection: {collection.count()}")

    # ---------------------------------------------------------
    # 3. User question
    # ---------------------------------------------------------

    query = "How are trade breaks investigated?"

    print("\nUser Question:")
    print(query)

    # ---------------------------------------------------------
    # 4. Generate query embedding
    # ---------------------------------------------------------

    print("\nGenerating query embedding...")

    query_embedding = embedding_model.embed_query(query)

    print("Query embedding generated!")

    # ---------------------------------------------------------
    # 5. Retrieve relevant chunks
    # ---------------------------------------------------------

    print("\nSearching ChromaDB...")

    results = search_collection(
        collection=collection,
        query_embedding=query_embedding,
        top_k=5
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    print(f"Retrieved chunks: {len(documents)}")

    # ---------------------------------------------------------
    # 6. Build source-aware context
    # ---------------------------------------------------------

    context_parts = []

    for i, document in enumerate(documents):

        metadata = metadatas[i]

        page = metadata.get("page", "Unknown")
        source = metadata.get("source", "Unknown")

        source_number = i + 1

        context_parts.append(
            f"""
SOURCE {source_number}
Page: {page}
Document: {source}

Content:
{document}
"""
        )

    context = "\n".join(context_parts)

    # ---------------------------------------------------------
    # 7. Build strict RAG prompt
    # ---------------------------------------------------------

    prompt = f"""
You are a Trade Operations assistant.

Answer the user's question using ONLY the information
contained in the provided sources.

STRICT RULES:

1. Do not use outside knowledge.

2. Do not invent facts.

3. Do not invent sources or page numbers.

4. Only cite a source if that source directly supports
   the specific statement you are making.

5. Do NOT cite every source simply because it was retrieved.

6. Use the smallest number of sources necessary.

7. Prefer ONE source when one source is sufficient.

8. Use a maximum of TWO sources in your entire answer.

9. If multiple sources contain the same information,
   cite only the most relevant source.

10. Citations must appear immediately after the statement
    they support.

11. Use this exact citation format:
    [Source 1]

    or:

    [Source 1, Source 2]

12. Only use source numbers that actually exist in the
    provided sources.

13. Never create a citation such as [Source 6] if only
    four sources were provided.

14. Do not create a separate "Sources" section in your
    answer.

15. Keep the answer concise: one short paragraph unless
    the question requires a list.

16. If the provided sources do not contain enough
    information to answer the question, say:

    "I could not find enough information in the provided
    documents to answer this question."

IMPORTANT:

The retrieved sources may contain related information
that does not directly answer the question.

Do NOT cite a source merely because it is related.

USER QUESTION:

{query}

PROVIDED SOURCES:

{context}

ANSWER:
"""

    # ---------------------------------------------------------
    # 8. Send context + question to Qwen
    # ---------------------------------------------------------

    print("\nSending context to Qwen...")

    response = generate_response(prompt)

    # ---------------------------------------------------------
    # 9. Extract cited source numbers
    # ---------------------------------------------------------

    cited_source_numbers = sorted(
        {
            int(number)
            for number in re.findall(
                r"Source\s+(\d+)",
                response,
                flags=re.IGNORECASE
            )
            if 1 <= int(number) <= len(metadatas)
        }
    )


    # ---------------------------------------------------------
    # 11. Build verified source list
    # ---------------------------------------------------------

    source_lines = []

    for source_number in cited_source_numbers:

        metadata = metadatas[source_number - 1]

        page = metadata.get("page", "Unknown")
        source = metadata.get("source", "Unknown")

        source_lines.append(
            f"[Source {source_number}] {source} — Page {page}"
        )

    # ---------------------------------------------------------
    # 12. Display final answer
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL RAG ANSWER")
    print("=" * 60)

    print(response)

    # ---------------------------------------------------------
    # 13. Display verified sources
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("VERIFIED SOURCES")
    print("=" * 60)

    if source_lines:
        print("\n".join(source_lines))
    else:
        print("No valid sources were cited by the model.")


if __name__ == "__main__":
    main()