from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from retriever import retrieve_documents
from llm_engine import get_gemini_llm
from schemas import ClaimResponse


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
    documents = retrieve_documents(query)
    context = "\n".join(
        f"[{i+1}] {doc.page_content}" for i, doc in enumerate(documents)
    )

    prompt = PromptTemplate.from_template(TEMPLATE)
    llm = get_gemini_llm()
    output_parser = JsonOutputParser()

    # Create input dict using RunnableLambda
    prepare_inputs = RunnableLambda(lambda _: {"context": context, "question": query})

    # Compose chain: input preparation → prompt → llm → parse json
    chain = prepare_inputs | prompt | llm | output_parser

    parsed_output = chain.invoke({})
    return ClaimResponse(**parsed_output)


# Example usage
if __name__ == "__main__":
    query = "46M, knee surgery in Pune, 3-month-old policy"
    response = run_query_pipeline(query)
    print(response)
