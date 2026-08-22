from app.llm import generate_response


def main():

    print("=" * 60)
    print("TRADEOPS RAG - LLM TEST")
    print("=" * 60)

    prompt = "Explain what a trade break is in one short paragraph."

    print("\nSending prompt to Qwen...")
    print(prompt)

    response = generate_response(prompt)

    print("\nResponse:")
    print("=" * 60)
    print(response)
    print("=" * 60)


if __name__ == "__main__":
    main()