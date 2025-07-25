"""Main entry point for the RAG pipeline application."""

from rag_pipeline.document_loader import load_documents
from rag_pipeline.embedder import build_vector_store
from rag_pipeline.query_pipeline import run_query_pipeline


def main():
    """Main function to execute the RAG pipeline."""
    docs = load_documents([r"rag_pipeline\docs\policy.pdf"])
    build_vector_store(docs)

    query = (
        "What is the grace period for premium payment under the "
        "National Parivar Mediclaim Plus Policy?"
    )

    response = run_query_pipeline(query)

    print(response)


if __name__ == "__main__":
    main()
