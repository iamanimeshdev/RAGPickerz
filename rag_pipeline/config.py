from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

VECTOR_DB_PATH=r".\faiss_index"
CHUNK_SIZE=350
CHUNK_OVERLAP=50

print("started loading config")

embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
print("loaded embeddings in config")
splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
print("loaded splitter in config")