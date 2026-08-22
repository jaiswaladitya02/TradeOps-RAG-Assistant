from app.embeddings import LocalEmbeddingModel
from app.retriever import get_chroma_collection, search_collection
from app.llm import generate_response


class TradeOpsRAG:
    """
    Main RAG pipeline for the TradeOps Assistant.

    Handles:
    - Query embedding
    - ChromaDB retrieval
    - Context construction
    - LLM response generation
    - Source metadata
    """

    def __init__(self):

        print("Loading embedding model...")

        self.embedding_model = LocalEmbeddingModel()

        print("Embedding model loaded!")

        print("\nConnecting to ChromaDB...")

        self.collection = get_chroma_collection()

        print("Collection connected!")
        print(
            f"Documents in collection: "
            f"{self.collection.count()}"
        )

    # ---------------------------------------------------------
    # RETRIEVAL
    # ---------------------------------------------------------

    def retrieve(self, query, top_k=5):

        query_embedding = self.embedding_model.embed_query(
            query
        )

        results = search_collection(
            collection=self.collection,
            query_embedding=query_embedding,
            top_k=top_k
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results.get("distances", [[]])[0]

        return documents, metadatas, distances

    # ---------------------------------------------------------
    # CONTEXT
    # ---------------------------------------------------------

    def build_context(
        self,
        documents,
        metadatas,
        max_context_chunks=None
    ):
        """
        Builds source-aware context.

        max_context_chunks allows chat.py to control how many
        retrieved chunks are sent to the LLM.
        """

        if max_context_chunks is not None:

            documents = documents[
                :max_context_chunks
            ]

            metadatas = metadatas[
                :max_context_chunks
            ]

        context_parts = []

        for i, document in enumerate(documents):

            metadata = metadatas[i]

            page = metadata.get(
                "page",
                "Unknown"
            )

            source = metadata.get(
                "source",
                "Unknown"
            )

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

        return "\n".join(context_parts)

    # ---------------------------------------------------------
    # ANSWER GENERATION
    # ---------------------------------------------------------

    def generate_answer(
        self,
        query,
        documents,
        metadatas,
        max_context_chunks=None
    ):
        """
        Generate a grounded answer using only the
        retrieved TradeOps documentation.
        """

        context = self.build_context(
            documents=documents,
            metadatas=metadatas,
            max_context_chunks=max_context_chunks
        )

        prompt = f"""
You are the TradeOps RAG Assistant.

Your job is to answer questions using ONLY the
information contained in the provided TradeOps sources.

STRICT RULES:

1. Use ONLY the provided sources.

2. Do NOT use outside knowledge.

3. Do NOT invent facts.

4. Answer the user's exact question.

5. If the sources do not contain enough information
to answer the question, say exactly:

"I could not find this information in the provided documents."

6. Keep the answer concise and professional.

7. When the source directly supports an answer,
include a citation immediately after the relevant
statement.

Use this citation format:

[Source 1]

or:

[Source 1, Source 2]

8. Only cite source numbers that actually exist in
the PROVIDED SOURCES.

9. Do not invent page numbers.

10. Do not create a separate Sources section.

11. Prefer the most relevant source rather than
citing every retrieved source.

12. If the question asks "why", explain the reason
using the source content.

13. If the question asks "what role", explain the
specific role described by the sources.

14. Do not merely repeat unrelated TradeOps concepts.
Stay focused on the question.

USER QUESTION:

{query}

PROVIDED SOURCES:

{context}

ANSWER:
"""

        response = generate_response(prompt)

        return response.strip()

    # ---------------------------------------------------------
    # COMPLETE RAG PIPELINE
    # ---------------------------------------------------------

    def ask(
        self,
        query,
        top_k=5,
        max_context_chunks=None
    ):
        """
        Run the complete RAG pipeline.

        max_context_chunks is optional so existing code that
        only passes query/top_k continues to work.
        """

        documents, metadatas, distances = self.retrieve(
            query=query,
            top_k=top_k
        )

        answer = self.generate_answer(
            query=query,
            documents=documents,
            metadatas=metadatas,
            max_context_chunks=max_context_chunks
        )

        sources = []

        for i, metadata in enumerate(
            metadatas,
            start=1
        ):

            distance = None

            if i - 1 < len(distances):
                distance = distances[i - 1]

            sources.append(
                {
                    "source_number": i,
                    "source": metadata.get(
                        "source",
                        "Unknown"
                    ),
                    "page": metadata.get(
                        "page",
                        "Unknown"
                    ),
                    "distance": distance
                }
            )

        return answer, sources