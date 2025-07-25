"""Query Pipeline Module

Returns:
    ClaimResponse: The response containing the decision,
                    amount, and justification.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser

from rag_pipeline.llm_engine import get_gemini_llm
from rag_pipeline.retriever import retrieve_documents
from rag_pipeline.schemas import ClaimResponse


TEMPLATE = """You are a claims processing assistant.

Here are relevant policy clauses:
{context}

User Query:
{question}

Instructions:
- You must respond with only a valid JSON object.
- Do not include any explanations outside the JSON.
- Follow this format exactly:
{{
    "decision": "approved or rejected",
    "amount": number (0 if rejected),
    "justification": [{{ "clause_id": int, "text": str }}]
}}
"""


def run_query_pipeline(query: str) -> ClaimResponse:
    """Run the query pipeline to process the user's query.

    Args:
        query (str): The user's query regarding the claim.

    Returns:
        ClaimResponse: The response containing the decision,
                        amount, and justification.
    """
    documents = retrieve_documents(query)
    context = "\n".join(
        f"[{i+1}] {doc.page_content}" for i, doc in enumerate(documents)
    )

    prompt = PromptTemplate.from_template(TEMPLATE)
    llm = get_gemini_llm()
    output_parser = JsonOutputParser()

    prepare_inputs = RunnableLambda(lambda _: {"context": context, "question": query})

    chain = prepare_inputs | prompt | llm | output_parser

    parsed_output = chain.invoke({})
    return ClaimResponse(**parsed_output)
