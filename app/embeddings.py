from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


class LocalEmbeddingModel(Embeddings):
    """
    Adapter that makes a SentenceTransformer model
    compatible with LangChain's embedding interface.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True
        )

        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            convert_to_numpy=True
        )

        return embedding.tolist()