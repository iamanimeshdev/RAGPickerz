"""Provides a function to get a Google Gemini LLM instance.

Returns:
    ChatGoogleGeneraticeAi: An instance of the Google Gemini LLM.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

_llm=ChatGoogleGenerativeAI(
        model="gemma-3n-e2b-it",
        temperature=0.2,
    )

def get_gemini_llm() -> ChatGoogleGenerativeAI:
    """Get an instance of the Google Gemini LLM.

    Returns:
        ChatGoogleGenerativeAI: An instance of the Google Gemini LLM with
                                specified parameters.
    """
    return _llm