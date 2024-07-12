# openai_assistant/__init__.py
from .assistant_manager import *
from .thread_manager import *

_openai = None


def init(openai_client):
    global _openai
    _openai = openai_client


def get_openai_client():
    if _openai is None:
        raise ValueError("OpenAI client is not initialized. Call `init(openai_client)` first.")
    return _openai
