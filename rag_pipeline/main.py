from rag_pipeline.document_loader import load_documents
from rag_pipeline.embedder import build_vector_store
from rag_pipeline.query_pipeline import run_batch_query_pipeline
import time


def main():
    """Main function to execute the RAG pipeline."""
    start = time.time()
    docs = load_documents([r"rag_pipeline\docs\BAJHLIP23020V012223.pdf"])
    build_vector_store(docs)

    questions = [
    "What is the definition of an Accident under this policy?",
    "What does the policy define as a Pre-Existing Disease?",
    "What are the conditions for coverage of Mental Illness Treatment?",
    "What expenses are covered under In-patient Hospitalization Treatment within India?",
    "What are the criteria for an AYUSH Hospital to be covered under the policy?",
    "What does the policy state about the Grace Period for premium payment?",
    "What is covered under Living Donor Medical Costs?",
    "How does the policy define Emergency Treatment outside area of cover?",
    "What benefits are included under Out-patient Treatment in the Imperial Plus Plan?",
    "What are the exclusions for Mental Illness Treatment according to the policy?"
  ]

    answers = run_batch_query_pipeline(questions)
    end = time.time()

    print({"answers": answers})
    print(f"⏱️ Total time: {end - start:.2f}s")

if __name__ == "__main__":
    main()
