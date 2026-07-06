from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100
):
    """
    Splits LangChain Documents into smaller chunks.

    Args:
        documents (list[Document]): Documents to split.
        chunk_size (int): Maximum characters per chunk.
        chunk_overlap (int): Number of overlapping characters.

    Returns:
        list[Document]: Chunked LangChain Documents.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    return chunks