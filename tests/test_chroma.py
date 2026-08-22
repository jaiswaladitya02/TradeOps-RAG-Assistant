import chromadb


print("Creating Chroma client...")

client = chromadb.PersistentClient(
    path="vector_store_test_direct"
)

print("Chroma client created successfully!")


collection = client.get_or_create_collection(
    name="test_collection"
)

print("Collection created successfully!")


collection.add(
    ids=["1", "2"],
    documents=[
        "Trade lifecycle begins with trade capture.",
        "Trade breaks occur when expected trade information does not match."
    ],
    embeddings=[
        [0.1] * 384,
        [0.2] * 384
    ]
)

print("Documents added successfully!")


results = collection.query(
    query_embeddings=[[0.1] * 384],
    n_results=1
)

print("\nSearch completed!")

print(results)