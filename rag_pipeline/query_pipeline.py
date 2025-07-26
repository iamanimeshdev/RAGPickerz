from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from rag_pipeline.llm_engine import get_gemini_llm
from rag_pipeline.retriever import retrieve_documents
from concurrent.futures import ThreadPoolExecutor
import time

TEMPLATE = """You are a helpful insurance policy assistant.

Here are relevant policy clauses:
{context}

User Question:
{question}

Instructions:
- Answer the question in clear, concise natural language.
- Be factual, citing specifics from the context if available.
-Respond with one Line.
- Include any explanations outside the answer on if necessary and provided.
- Return ONLY the answer as plain text — no bullet points, no JSON, no labels.
"""

def run_query_pipeline(question: str) -> str:
    start = time.time()
    documents = retrieve_documents(question)
    print(f"🔍 Retrieval time: {time.time() - start:.2f}s")

    context = "\n".join(f"[{i+1}] {doc.page_content}" for i, doc in enumerate(documents))

    prompt = PromptTemplate.from_template(TEMPLATE)
    llm = get_gemini_llm()

    prepare_inputs = RunnableLambda(lambda _: {"context": context, "question": question})
    chain = prepare_inputs | prompt | llm

    start = time.time()
    response=chain.invoke({})
    print(f"🤖 LLM generation time: {time.time() - start:.2f}s")
    return response.content if hasattr(response, 'content') else response

def run_batch_query_pipeline(questions: list[str]) -> list[str]:
    """Processes multiple questions one by one using the query pipeline.

    Args:
        questions (list[str]): List of user questions.

    Returns:
        list[str]: List of natural language answers in order.
    """
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(run_query_pipeline, questions))
    return results
