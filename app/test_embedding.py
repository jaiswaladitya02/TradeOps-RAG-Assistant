from embeddings import LocalEmbeddingModel


print("Loading embedding model...")

embedding_model = LocalEmbeddingModel()

print("Model loaded successfully!")


sample_text = "Trade lifecycle begins with trade capture."

print("\nGenerating document embedding...")

document_embedding = embedding_model.embed_documents(
    [sample_text]
)

print("Document embedding generated!")

print(f"\nNumber of embeddings: {len(document_embedding)}")
print(f"Embedding dimensions: {len(document_embedding[0])}")

print("\nFirst 10 values:")
print(document_embedding[0][:10])


print("\nGenerating query embedding...")

query_embedding = embedding_model.embed_query(
    "What is the trade lifecycle?"
)

print("Query embedding generated!")

print(f"\nQuery embedding dimensions: {len(query_embedding)}")

print("\nFirst 10 query values:")
print(query_embedding[:10])