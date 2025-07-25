"""Provides a function to get a Google Gemini LLM instance.

Returns:
    ChatGoogleGeneraticeAi: An instance of the Google Gemini LLM.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


def get_gemini_llm() -> ChatGoogleGenerativeAI:
    """Get an instance of the Google Gemini LLM.

    Returns:
        ChatGoogleGenerativeAI: An instance of the Google Gemini LLM with
                                specified parameters.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
    )
