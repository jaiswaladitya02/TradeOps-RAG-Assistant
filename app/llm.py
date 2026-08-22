import ollama


def generate_response(
    prompt: str,
    model: str = "qwen2.5:7b"
):
    """
    Sends a prompt to the local Ollama model
    and returns the generated response.
    """

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]