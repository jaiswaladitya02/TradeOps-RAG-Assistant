from pdf_loader import load_pdf
from chunker import chunk_documents


def main():
    """
    Main entry point for the document ingestion pipeline.
    """

    pdf_path = "data/raw/TradeOps_Enterprise_Manual_50_Pages.pdf"

    print("=" * 60)
    print("TRADE OPERATIONS RAG - DOCUMENT INGESTION")
    print("=" * 60)

    # Step 1: Load the PDF
    documents = load_pdf(pdf_path)

    print(f"\nDocuments Loaded : {len(documents)}")

    # Step 2: Chunk the documents
    chunks = chunk_documents(documents)

    print(f"Chunks Created  : {len(chunks)}")

    # Step 3: Inspect a sample chunk
    print("\n" + "=" * 60)
    print("CHUNK INSPECTION")
    print("=" * 60)

    print(f"Total Chunks: {len(chunks)}")

    print("\nFirst Chunk")
    print("-" * 40)
    print(chunks[0].page_content)

    print("\nMetadata")
    print(chunks[0].metadata)

    print("\nLength")
    print(len(chunks[0].page_content))

    print("\nSecond Chunk")
    print("-" * 40)
    print(chunks[1].page_content[:300])


if __name__ == "__main__":
    main()