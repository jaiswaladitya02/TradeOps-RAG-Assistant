from app.rag import TradeOpsRAG


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [

    {
        "category": "Trade Breaks",
        "question": "How are trade breaks investigated?",
        "expected_keywords": [
            "upstream",
            "downstream",
            "counterparties",
            "root cause"
        ],
        "expected_page": 14,
        "out_of_scope": False,
    },

    {
        "category": "Trade Breaks",
        "question": "What causes a trade break?",
        "expected_keywords": [
            "data",
            "reference",
            "mapping",
            "instruction"
        ],
        "expected_page": 14,
        "out_of_scope": False,
    },

    {
        "category": "Trade Lifecycle",
        "question": "What is involved in the trade lifecycle?",
        "expected_keywords": [
            "account",
            "exchange",
            "product",
            "settlement",
            "reconciliation"
        ],
        "expected_page": 2,
        "out_of_scope": False,
    },

    {
        "category": "Reconciliation",
        "question": "What are reconciliation controls?",
        "expected_keywords": [
            "trade breaks",
            "upstream",
            "downstream",
            "service"
        ],
        "expected_page": 36,
        "out_of_scope": False,
    },

    {
        "category": "Reference Data",
        "question": "What role does reference data play in trade operations?",
        "expected_keywords": [
            "reference data",
            "product",
            "processing"
        ],
        "expected_page": 50,
        "out_of_scope": False,
    },

    {
        "category": "Clearing",
        "question": "What is the role of clearing in trade operations?",
        "expected_keywords": [
            "clearing",
            "trade",
            "processing"
        ],
        "expected_page": 10,
        "out_of_scope": False,
    },

    {
        "category": "Settlement",
        "question": "What factors are considered during settlement?",
        "expected_keywords": [
            "settlement",
            "timeline",
            "processing"
        ],
        "expected_page": 13,
        "out_of_scope": False,
    },

    {
        "category": "Client Communication",
        "question": "How should trade operations teams handle client queries?",
        "expected_keywords": [
            "client",
            "communication",
            "queries"
        ],
        "expected_page": 8,
        "out_of_scope": False,
    },

    {
        "category": "Regulatory Reporting",
        "question": "What is the role of regulatory reporting in trade operations?",
        "expected_keywords": [
            "regulatory",
            "reporting",
            "operations"
        ],
        "expected_page": 36,
        "out_of_scope": False,
    },

    {
        "category": "Exchange Traded Derivatives",
        "question": "How are exchange traded derivatives handled?",
        "expected_keywords": [
            "exchange traded derivatives",
            "clearing",
            "trade"
        ],
        "expected_page": 53,
        "out_of_scope": False,
    },

    {
        "category": "STP",
        "question": "How can operational controls improve straight-through processing?",
        "expected_keywords": [
            "straight-through processing",
            "STP",
            "manual intervention"
        ],
        "expected_page": 87,
        "out_of_scope": False,
    },

    {
        "category": "Operational Controls",
        "question": "Why are strong operational controls important?",
        "expected_keywords": [
            "STP",
            "manual intervention",
            "operational risk",
            "client satisfaction"
        ],
        "expected_page": 25,
        "out_of_scope": False,
    },

    {
        "category": "Exception Queues",
        "question": "What role do exception queues play in trade operations?",
        "expected_keywords": [
            "exception",
            "investigation",
            "issue resolution"
        ],
        "expected_page": 16,
        "out_of_scope": False,
    },

    {
        "category": "Audit",
        "question": "Why are audit requirements important in trade operations?",
        "expected_keywords": [
            "audit",
            "investigation",
            "issue resolution"
        ],
        "expected_page": 48,
        "out_of_scope": False,
    },

    {
        "category": "Market Deadlines",
        "question": "Why are market deadlines important?",
        "expected_keywords": [
            "market",
            "deadlines",
            "processing"
        ],
        "expected_page": 72,
        "out_of_scope": False,
    },

    {
        "category": "Clearing Broker",
        "question": "What information about clearing brokers must operations teams verify?",
        "expected_keywords": [
            "clearing broker",
            "verify",
            "trade"
        ],
        "expected_page": 10,
        "out_of_scope": False,
    },

    {
        "category": "Client Instructions",
        "question": "Why are client instructions important in trade processing?",
        "expected_keywords": [
            "client instructions",
            "processing",
            "trade"
        ],
        "expected_page": 80,
        "out_of_scope": False,
    },

    {
        "category": "Systems",
        "question": "Why do analysts validate upstream and downstream data?",
        "expected_keywords": [
            "upstream",
            "downstream",
            "investigation",
            "issue resolution"
        ],
        "expected_page": 33,
        "out_of_scope": False,
    },

    {
        "category": "Root Cause",
        "question": "Why is documenting root causes important when investigating trade breaks?",
        "expected_keywords": [
            "root cause",
            "trade break",
            "investigation"
        ],
        "expected_page": 14,
        "out_of_scope": False,
    },

    # --------------------------------------------------------
    # OUT OF SCOPE
    # --------------------------------------------------------

    {
        "category": "OUT OF SCOPE",
        "question": "What is the capital of France?",
        "expected_keywords": [],
        "expected_page": None,
        "out_of_scope": True,
    },

    {
        "category": "OUT OF SCOPE",
        "question": "Who won the FIFA World Cup in 2022?",
        "expected_keywords": [],
        "expected_page": None,
        "out_of_scope": True,
    },
]


# ============================================================
# CONFIGURATION
# ============================================================

# Distance below this value is considered useful retrieval.
STRONG_THRESHOLD = 0.50

# Distance between these values is considered borderline.
WEAK_THRESHOLD = 0.60


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def classify_distance(distance):
    """
    Classify retrieval quality using the same diagnostic
    thresholds used by the retrieval evaluation.
    """

    if distance <= STRONG_THRESHOLD:
        return "STRONG"

    if distance <= WEAK_THRESHOLD:
        return "WEAK"

    return "POOR"


def answer_contains_keywords(answer, keywords):
    """
    Check whether the generated answer contains the
    expected concepts.

    We use a simple keyword check rather than an LLM
    judge at this stage so that the evaluation itself
    remains deterministic.
    """

    if not keywords:
        return True, []

    answer_lower = answer.lower()

    found = []
    missing = []

    for keyword in keywords:

        if keyword.lower() in answer_lower:
            found.append(keyword)
        else:
            missing.append(keyword)

    return len(found) > 0, missing


def is_refusal(answer):
    """
    Detect whether the assistant correctly refused an
    out-of-scope question.
    """

    answer_lower = answer.lower()

    refusal_phrases = [
        "could not find",
        "cannot find",
        "not found in the provided documents",
        "not available in the provided documents",
        "do not contain this information",
        "does not contain this information",
    ]

    return any(
        phrase in answer_lower
        for phrase in refusal_phrases
    )


def citation_exists(answer):
    """
    Check whether the answer contains a citation such as:

        [Source 1]

        [Source 1, Source 2]
    """

    return "[source" in answer.lower()


def citation_is_valid(answer, sources):
    """
    Check that cited source numbers actually exist in the
    retrieved source list.
    """

    import re

    matches = re.findall(
        r"Source\s+(\d+)",
        answer,
        flags=re.IGNORECASE
    )

    if not matches:
        return False

    valid_numbers = {
        source["source_number"]
        for source in sources
    }

    for number in matches:

        if int(number) not in valid_numbers:
            return False

    return True


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("TRADEOPS RAG - END-TO-END EVALUATION")
    print("=" * 70)

    print("\nInitializing RAG system...\n")

    rag = TradeOpsRAG()

    print("\n" + "=" * 70)
    print("STARTING EVALUATION")
    print("=" * 70)

    total = len(TEST_CASES)

    retrieval_strong = 0
    retrieval_weak = 0
    retrieval_poor = 0

    answer_pass = 0
    citation_pass = 0
    overall_pass = 0

    out_of_scope_total = 0
    out_of_scope_pass = 0

    results = []

    # ========================================================
    # RUN EVERY TEST
    # ========================================================

    for index, test in enumerate(TEST_CASES, start=1):

        question = test["question"]

        print("\n")
        print("=" * 70)
        print(f"TEST {index}/{total}")
        print(f"Category: {test['category']}")
        print(f"Question: {question}")
        print("=" * 70)

        try:

            # ------------------------------------------------
            # Run RAG
            # ------------------------------------------------

            answer, sources = rag.ask(
                query=question,
                top_k=5
            )

            answer = answer.strip()

            # ------------------------------------------------
            # Retrieval evaluation
            # ------------------------------------------------

            if sources:

                best_source = min(
                    sources,
                    key=lambda x: x["distance"]
                    if x["distance"] is not None
                    else float("inf")
                )

                best_distance = best_source["distance"]

                retrieval_classification = classify_distance(
                    best_distance
                )

                if retrieval_classification == "STRONG":
                    retrieval_strong += 1

                elif retrieval_classification == "WEAK":
                    retrieval_weak += 1

                else:
                    retrieval_poor += 1

            else:

                best_source = None
                best_distance = None
                retrieval_classification = "POOR"

                retrieval_poor += 1

            # ------------------------------------------------
            # Print retrieval result
            # ------------------------------------------------

            print("\nBest retrieval:")

            print(
                f"Distance: "
                f"{best_distance}"
            )

            print(
                f"Classification: "
                f"{retrieval_classification}"
            )

            if best_source:

                print(
                    f"Page: "
                    f"{best_source['page']}"
                )

                print(
                    f"Source: "
                    f"{best_source['source']}"
                )

            # ------------------------------------------------
            # Answer evaluation
            # ------------------------------------------------

            if test["out_of_scope"]:

                out_of_scope_total += 1

                answer_ok = is_refusal(answer)

                if answer_ok:
                    out_of_scope_pass += 1

                print("\nGenerated answer:")

                print(answer)

                print(
                    "\nOut-of-scope handling: "
                    f"{'PASS' if answer_ok else 'FAIL'}"
                )

                citation_ok = True

            else:

                answer_ok, missing_keywords = (
                    answer_contains_keywords(
                        answer,
                        test["expected_keywords"]
                    )
                )

                if answer_ok:
                    answer_pass += 1

                print("\nGenerated answer:")

                print(answer)

                if missing_keywords:

                    print(
                        "\nMissing expected concepts:"
                    )

                    print(
                        ", ".join(missing_keywords)
                    )

                print(
                    "\nAnswer quality: "
                    f"{'PASS' if answer_ok else 'FAIL'}"
                )

                # ------------------------------------------------
                # Citation evaluation
                # ------------------------------------------------

                citation_ok = citation_is_valid(
                    answer,
                    sources
                )

                if citation_ok:
                    citation_pass += 1

                print(
                    "Citation quality: "
                    f"{'PASS' if citation_ok else 'FAIL'}"
                )

            # ------------------------------------------------
            # Overall result
            # ------------------------------------------------

            if test["out_of_scope"]:

                overall_ok = (
                    retrieval_classification == "POOR"
                    and answer_ok
                )

            else:

                overall_ok = (
                    answer_ok
                    and citation_ok
                )

            if overall_ok:
                overall_pass += 1

            print(
                "Overall result: "
                f"{'PASS' if overall_ok else 'FAIL'}"
            )

            results.append(
                {
                    "question": question,
                    "retrieval": retrieval_classification,
                    "distance": best_distance,
                    "answer_pass": answer_ok,
                    "citation_pass": citation_ok,
                    "overall_pass": overall_ok,
                }
            )

        except Exception as error:

            print("\nERROR:")
            print(error)

            results.append(
                {
                    "question": question,
                    "retrieval": "ERROR",
                    "distance": None,
                    "answer_pass": False,
                    "citation_pass": False,
                    "overall_pass": False,
                }
            )

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    print(
        f"\nTotal tests:        {total}"
    )

    print(
        f"Retrieval STRONG:   {retrieval_strong}"
    )

    print(
        f"Retrieval WEAK:     {retrieval_weak}"
    )

    print(
        f"Retrieval POOR:     {retrieval_poor}"
    )

    retrieval_percentage = (
        retrieval_strong / total * 100
    )

    answer_percentage = (
        answer_pass
        / (total - out_of_scope_total)
        * 100
    )

    citation_percentage = (
        citation_pass
        / (total - out_of_scope_total)
        * 100
    )

    overall_percentage = (
        overall_pass
        / total
        * 100
    )

    out_of_scope_percentage = (
        out_of_scope_pass
        / out_of_scope_total
        * 100
    )

    print(
        f"\nStrong retrieval:   "
        f"{retrieval_percentage:.1f}%"
    )

    print(
        f"Answer accuracy:    "
        f"{answer_percentage:.1f}%"
    )

    print(
        f"Citation accuracy:  "
        f"{citation_percentage:.1f}%"
    )

    print(
        f"Out-of-scope:       "
        f"{out_of_scope_percentage:.1f}%"
    )

    print(
        f"Overall RAG score:  "
        f"{overall_percentage:.1f}%"
    )

    # ========================================================
    # OUT OF SCOPE SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("OUT-OF-SCOPE EVALUATION")
    print("=" * 70)

    print(
        f"\nPassed: "
        f"{out_of_scope_pass}/{out_of_scope_total}"
    )

    # ========================================================
    # FAILED TESTS
    # ========================================================

    failed_tests = [
        result
        for result in results
        if not result["overall_pass"]
    ]

    print("\n")
    print("=" * 70)
    print("FAILED TESTS")
    print("=" * 70)

    if not failed_tests:

        print(
            "\nAll tests passed."
        )

    else:

        for result in failed_tests:

            print(
                f"\nQuestion: "
                f"{result['question']}"
            )

            print(
                f"Retrieval: "
                f"{result['retrieval']}"
            )

            print(
                f"Distance: "
                f"{result['distance']}"
            )

            print(
                f"Answer: "
                f"{'PASS' if result['answer_pass'] else 'FAIL'}"
            )

            print(
                f"Citation: "
                f"{'PASS' if result['citation_pass'] else 'FAIL'}"
            )

    print("\n")
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()