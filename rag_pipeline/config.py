from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


VECTOR_DB_PATH=r".\faiss_index"
CHUNK_SIZE=350
CHUNK_OVERLAP=50

print("started loading config")

embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
print("loaded embeddings in config")
splitter = SemanticChunker(embeddings, breakpoint_threshold_type="gradient", breakpoint_threshold_amount=0.5)
print("loaded splitter in config")

LLM = ChatGoogleGenerativeAI(
        model="gemma-3n-e2b-it",
        temperature=0.2,
    )
print("loaded LLM in config")