from rag_pipeline.document_loader import load_documents
from rag_pipeline.embedder import build_vector_store
from rag_pipeline.query_pipeline import run_batch_query_pipeline


def main():
    """Main function to execute the RAG pipeline."""
    docs = load_documents([r"rag_pipeline\docs\BAJHLIP23020V012223.pdf"])
    build_vector_store(docs)

    questions = [
        "What is the maximum duration for which post-hospitalization expenses are covered under the domestic plan?",
  
    ]

    answers = run_batch_query_pipeline(questions)

    print({"answers": answers})

if __name__ == "__main__":
    main()
