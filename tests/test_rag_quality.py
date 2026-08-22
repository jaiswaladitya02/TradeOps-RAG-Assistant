from app.embeddings import LocalEmbeddingModel
from app.retriever import get_chroma_collection, search_collection


# =============================================================
# TEST QUESTIONS
# =============================================================

TEST_CASES = [

    # ---------------------------------------------------------
    # Trade Breaks
    # ---------------------------------------------------------

    {
        "category": "Trade Breaks",
        "question": "How are trade breaks investigated?"
    },

    {
        "category": "Trade Breaks",
        "question": "What causes a trade break?"
    },

    # ---------------------------------------------------------
    # Trade Lifecycle
    # ---------------------------------------------------------

    {
        "category": "Trade Lifecycle",
        "question": "What is involved in the trade lifecycle?"
    },

    # ---------------------------------------------------------
    # Reconciliation
    # ---------------------------------------------------------

    {
        "category": "Reconciliation",
        "question": "What are reconciliation controls?"
    },

    # ---------------------------------------------------------
    # Reference Data
    # ---------------------------------------------------------

    {
        "category": "Reference Data",
        "question": "What role does reference data play in trade operations?"
    },

    # ---------------------------------------------------------
    # Clearing
    # ---------------------------------------------------------

    {
        "category": "Clearing",
        "question": "What is the role of clearing in trade operations?"
    },

    # ---------------------------------------------------------
    # Settlement
    # ---------------------------------------------------------

    {
        "category": "Settlement",
        "question": "What factors are considered during settlement?"
    },

    # ---------------------------------------------------------
    # Client Communication
    # ---------------------------------------------------------

    {
        "category": "Client Communication",
        "question": "How should trade operations teams handle client queries?"
    },

    # ---------------------------------------------------------
    # Regulatory Reporting
    # ---------------------------------------------------------

    {
        "category": "Regulatory Reporting",
        "question": "What is the role of regulatory reporting in trade operations?"
    },

    # ---------------------------------------------------------
    # Exchange Traded Derivatives
    # ---------------------------------------------------------

    {
        "category": "Exchange Traded Derivatives",
        "question": "How are exchange traded derivatives handled?"
    },

    # ---------------------------------------------------------
    # STP
    # ---------------------------------------------------------

    {
        "category": "STP",
        "question": "How can operational controls improve straight-through processing?"
    },

    # ---------------------------------------------------------
    # Operational Controls
    # ---------------------------------------------------------

    {
        "category": "Operational Controls",
        "question": "Why are strong operational controls important?"
    },

    # ---------------------------------------------------------
    # Exception Queues
    # ---------------------------------------------------------

    {
        "category": "Exception Queues",
        "question": "What role do exception queues play in trade operations?"
    },

    # ---------------------------------------------------------
    # Audit
    # ---------------------------------------------------------

    {
        "category": "Audit",
        "question": "Why are audit requirements important in trade operations?"
    },

    # ---------------------------------------------------------
    # Market Deadlines
    # ---------------------------------------------------------

    {
        "category": "Market Deadlines",
        "question": "Why are market deadlines important?"
    },

    # ---------------------------------------------------------
    # Clearing Broker
    # ---------------------------------------------------------

    {
        "category": "Clearing Broker",
        "question": "What information about clearing brokers must operations teams verify?"
    },

    # ---------------------------------------------------------
    # Client Instructions
    # ---------------------------------------------------------

    {
        "category": "Client Instructions",
        "question": "Why are client instructions important in trade processing?"
    },

    # ---------------------------------------------------------
    # Upstream / Downstream Systems
    # ---------------------------------------------------------

    {
        "category": "Systems",
        "question": "Why do analysts validate upstream and downstream data?"
    },

    # ---------------------------------------------------------
    # Root Cause
    # ---------------------------------------------------------

    {
        "category": "Root Cause",
        "question": "Why is documenting root causes important when investigating trade breaks?"
    },

    # ---------------------------------------------------------
    # OUT OF SCOPE
    # ---------------------------------------------------------

    {
        "category": "OUT OF SCOPE",
        "question": "What is the capital of France?"
    },

    {
        "category": "OUT OF SCOPE",
        "question": "Who won the FIFA World Cup in 2022?"
    },
]


# =============================================================
# CONFIGURATION
# =============================================================

TOP_K = 5

# Lower Chroma distance = stronger semantic match.

STRONG_THRESHOLD = 0.50
WEAK_THRESHOLD = 0.60


# =============================================================
# HELPERS
# =============================================================

def classify_distance(distance):
    """
    Classify retrieval quality based on ChromaDB distance.
    """

    if distance <= STRONG_THRESHOLD:

        return "STRONG"

    elif distance <= WEAK_THRESHOLD:

        return "WEAK"

    else:

        return "POOR"


def print_separator():
    print("\n" + "=" * 70)


# =============================================================
# MAIN
# =============================================================

def main():

    print("=" * 70)
    print("TRADEOPS RAG - RETRIEVAL QUALITY EVALUATION")
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
    # Statistics
    # ---------------------------------------------------------

    strong_count = 0
    weak_count = 0
    poor_count = 0

    total_tests = len(TEST_CASES)

    results_summary = []

    # ---------------------------------------------------------
    # Run tests
    # ---------------------------------------------------------

    for test_number, test_case in enumerate(
        TEST_CASES,
        start=1
    ):

        category = test_case["category"]
        question = test_case["question"]

        print_separator()

        print(
            f"TEST {test_number}/{total_tests}"
        )

        print(
            f"Category: {category}"
        )

        print(
            f"Question: {question}"
        )

        # -----------------------------------------------------
        # Generate embedding
        # -----------------------------------------------------

        query_embedding = (
            embedding_model.embed_query(
                question
            )
        )

        # -----------------------------------------------------
        # Search ChromaDB
        # -----------------------------------------------------

        results = search_collection(
            collection=collection,
            query_embedding=query_embedding,
            top_k=TOP_K
        )

        documents = results[
            "documents"
        ][0]

        metadatas = results[
            "metadatas"
        ][0]

        distances = results[
            "distances"
        ][0]

        # -----------------------------------------------------
        # Handle no results
        # -----------------------------------------------------

        if not distances:

            print(
                "\nResult: NO RESULTS"
            )

            results_summary.append(
                {
                    "category": category,
                    "question": question,
                    "status": "NO RESULTS",
                    "distance": None
                }
            )

            continue

        # -----------------------------------------------------
        # Best result
        # -----------------------------------------------------

        best_distance = min(
            distances
        )

        best_index = distances.index(
            best_distance
        )

        best_document = documents[
            best_index
        ]

        best_metadata = metadatas[
            best_index
        ]

        classification = (
            classify_distance(
                best_distance
            )
        )

        # -----------------------------------------------------
        # Statistics
        # -----------------------------------------------------

        if classification == "STRONG":

            strong_count += 1

        elif classification == "WEAK":

            weak_count += 1

        else:

            poor_count += 1

        # -----------------------------------------------------
        # Display
        # -----------------------------------------------------

        print(
            f"\nBest retrieval:"
        )

        print(
            f"Distance: {best_distance:.6f}"
        )

        print(
            f"Classification: {classification}"
        )

        print(
            f"Page: "
            f"{best_metadata.get('page', 'Unknown')}"
        )

        print(
            f"Source: "
            f"{best_metadata.get('source', 'Unknown')}"
        )

        print(
            "\nContent preview:"
        )

        preview = (
            best_document
            .replace("\n", " ")
            .strip()
        )

        if len(preview) > 350:

            preview = preview[:350] + "..."

        print(preview)

        # -----------------------------------------------------
        # Store summary
        # -----------------------------------------------------

        results_summary.append(
            {
                "category": category,
                "question": question,
                "status": classification,
                "distance": best_distance
            }
        )

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    print_separator()

    print(
        "FINAL EVALUATION"
    )

    print_separator()

    print(
        f"Total tests:   {total_tests}"
    )

    print(
        f"Strong:        {strong_count}"
    )

    print(
        f"Weak:          {weak_count}"
    )

    print(
        f"Poor:          {poor_count}"
    )

    # ---------------------------------------------------------
    # Percentages
    # ---------------------------------------------------------

    completed_tests = (
        strong_count
        + weak_count
        + poor_count
    )

    if completed_tests > 0:

        strong_percentage = (
            strong_count
            / completed_tests
            * 100
        )

        weak_percentage = (
            weak_count
            / completed_tests
            * 100
        )

        poor_percentage = (
            poor_count
            / completed_tests
            * 100
        )

        print(
            f"\nStrong retrieval: "
            f"{strong_percentage:.1f}%"
        )

        print(
            f"Weak retrieval:   "
            f"{weak_percentage:.1f}%"
        )

        print(
            f"Poor retrieval:    "
            f"{poor_percentage:.1f}%"
        )

    # =========================================================
    # OUT-OF-SCOPE TESTS
    # =========================================================

    print_separator()

    print(
        "OUT-OF-SCOPE TEST RESULTS"
    )

    print_separator()

    out_of_scope_results = [
        result
        for result in results_summary
        if result["category"]
        == "OUT OF SCOPE"
    ]

    for result in out_of_scope_results:

        print(
            f"\nQuestion: "
            f"{result['question']}"
        )

        print(
            f"Distance: "
            f"{result['distance']}"
        )

        print(
            f"Classification: "
            f"{result['status']}"
        )

    # =========================================================
    # IMPORTANT DIAGNOSTIC
    # =========================================================

    print_separator()

    print(
        "DIAGNOSTIC NOTES"
    )

    print_separator()

    print(
        "STRONG <= 0.50"
    )

    print(
        "WEAK   = 0.50 - 0.60"
    )

    print(
        "POOR   > 0.60"
    )

    print(
        "\nThese thresholds are initial diagnostic thresholds."
    )

    print(
        "They are NOT a final measure of answer correctness."
    )

    print(
        "We will use the results to decide whether the"
    )

    print(
        "embedding/retrieval configuration needs tuning."
    )

    print()


if __name__ == "__main__":

    main()