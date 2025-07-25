from document_loader import load_documents
from embedder import build_vector_store
from query_pipeline import run_query_pipeline


def main():
    docs = load_documents(
        [
            r"docs\BAJHLIP23020V012223.pdf",
            r"docs\CHOTGDP23004V012223.pdf",
            r"docs\EDLHLGA23009V012223.pdf",
            r"docs\HDFHLIP23024V072223.pdf",
            r"docs\ICIHLIP22012V012223.pdf",
        ]
    )
    build_vector_store(docs)

    query = "46M, knee surgery in Pune, 3-month-old policy"
    response = run_query_pipeline(query)

    print(response)


if __name__ == "__main__":
    main()
