"""Load documents from specified file paths.

Raises:
    ValueError: If the file type is unsupported.

Returns:
    List[Document]: A list of loaded documents.
"""

from typing import List
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain_community.docstore.document import Document


def load_documents(file_paths: List[str]) -> List[Document]:
    """Load documents from the specified file paths.

    Args:
        file_paths (List[str]): List of file paths to load documents from.

    Raises:
        ValueError: If the file type is unsupported.

    Returns:
        List[Document]: A list of loaded documents.
    """
    print(f"📂 Loading documents from {len(file_paths)} files...")
    docs = []
    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyMuPDFLoader(path)
        elif path.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(path)
        else:
            raise ValueError(f"Unsupported file type: {path}")
        docs.extend(loader.load())
    return docs
