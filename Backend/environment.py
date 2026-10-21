import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


def get_groq_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Set GROQ_API_KEY or GROK_API_KEY in the .env file."
        )

    return api_key


def get_groq_chat_model() -> ChatGroq:
    api_key = get_groq_api_key()
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        api_key=api_key,
        temperature=0,
    )
