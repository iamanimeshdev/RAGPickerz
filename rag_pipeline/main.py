import time
import threading



from rag_pipeline.document_loader import load_documents
from rag_pipeline.embedder import build_vector_store
from rag_pipeline.query_pipeline import run_batch_query_pipeline

def preload_config():
    global config
    import rag_pipeline.config  # expensive
    config = rag_pipeline.config
    config.embeddings.embed_query("warmup")

def main():
    preload_thread = threading.Thread(target=preload_config)
    preload_thread.start()

    start = time.perf_counter()

    docs = load_documents([r"rag_pipeline/docs/policy.pdf"])

    preload_thread.join() 

    build_vector_store(docs)

    questions = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]

    answers = run_batch_query_pipeline(questions)
    end = time.perf_counter()

    print({"answers": answers})
    print(f"⏱️ Total time: {end - start:.2f}s")

if __name__ == "__main__":
    main()
