import fitz
from langchain_core.documents import Document


def load_pdf(pdf_path: str):
    """
    Reads a PDF file and converts each page into a LangChain Document.

    Args:
        pdf_path (str): Path to the PDF.

    Returns:
        list[Document]: List of LangChain Document objects.
    """

    document = fitz.open(pdf_path)

    documents = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text()

        doc = Document(
            page_content=text,
            metadata={
                "page": page_number,
                "source": pdf_path
            }
        )

        documents.append(doc)

    return documents