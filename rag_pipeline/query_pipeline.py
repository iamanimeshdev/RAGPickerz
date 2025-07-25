from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from rag_pipeline.llm_engine import get_gemini_llm
from rag_pipeline.retriever import retrieve_documents

TEMPLATE = """You are a helpful insurance policy assistant.

Here are relevant policy clauses:
{context}

User Question:
{question}

Instructions:
- Answer the question in clear, concise natural language.
- Be factual, citing specifics from the context if available.
-Respond with atleast one Line.
- If the context does not contain enough information, respond with:
  "This information is not available in the provided text."
- Include any explanations outside the answer on if necessary and provided.
- Return ONLY the answer as plain text — no bullet points, no JSON, no labels.
"""

def run_query_pipeline(question: str) -> str:
    documents = retrieve_documents(question)
    context = "\n".join(f"[{i+1}] {doc.page_content}" for i, doc in enumerate(documents))

    prompt = PromptTemplate.from_template(TEMPLATE)
    llm = get_gemini_llm()

    prepare_inputs = RunnableLambda(lambda _: {"context": context, "question": question})
    chain = prepare_inputs | prompt | llm

    response=chain.invoke({})
    return response.content if hasattr(response, 'content') else response

def run_batch_query_pipeline(questions: list[str]) -> list[str]:
    """Processes multiple questions one by one using the query pipeline.

    Args:
        questions (list[str]): List of user questions.

    Returns:
        list[str]: List of natural language answers in order.
    """
    return [run_query_pipeline(q) for q in questions]
