from app.pdf_loader import load_pdf
from app.chunker import chunk_documents
from app.embeddings import LocalEmbeddingModel
from app.vector_store import create_vector_store


PDF_PATH = "data/raw/TradeOps_Enterprise_Manual_50_Pages.pdf"


print("Loading PDF...")

documents = load_pdf(PDF_PATH)

print(f"Documents loaded: {len(documents)}")


print("\nCreating chunks...")

chunks = chunk_documents(documents)

print(f"Chunks created: {len(chunks)}")


print("\nLoading embedding model...")

embedding_model = LocalEmbeddingModel()

print("Embedding model loaded!")


print("\nCreating test vector store...")

test_chunks = chunks[:5]

collection = create_vector_store(
    chunks=test_chunks,
    embedding_model=embedding_model,
    persist_directory="vector_store_test"
)

print("Vector store created successfully!")


print("\nRunning similarity search...")

query = "What is the trade lifecycle?"

query_embedding = embedding_model.embed_query(query)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print(f"\nResults returned: {len(results['documents'][0])}")


for i, document in enumerate(results["documents"][0], start=1):

    print("\n" + "=" * 50)
    print(f"RESULT {i}")
    print("=" * 50)

    print(document)

    print("\nMetadata:")
    print(results["metadatas"][0][i - 1])

    print("\nDistance:")
    print(results["distances"][0][i - 1])