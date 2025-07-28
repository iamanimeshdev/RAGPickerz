import time
import threading

start = time.perf_counter()
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
    docs = load_documents([r"rag_pipeline\docs\BAJHLIP23020V012223.pdf"])

    preload_thread.join() 

    build_vector_store(docs)

    questions = [
        "46M, knee surgery, Pune, 3-month policy"
    ]

    answers = run_batch_query_pipeline(questions)
    end = time.perf_counter()    
    print({"answers": answers})
    print(f"⏱️ Total time: {end - start:.2f}s")

if __name__ == "__main__":
    main()
