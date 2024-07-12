# openai_assistant/tool_functions_map.py
from typing import Callable, Dict

FUNCTION_MAP: Dict[str, Callable] = {}


def get_function(name: str) -> Callable:
    return FUNCTION_MAP.get(name)


def register_function(name: str, function: Callable):
    """
    Register a single function to be used in the assistant.
    :param name: Name of the function.
    :param function: The function to register.
    """
    FUNCTION_MAP[name] = function


def register_functions(functions: Dict[str, Callable]):
    """
    Register multiple functions to be used in the assistant.
    :param functions: A dictionary of function names and their corresponding functions.
    """
    FUNCTION_MAP.update(functions)
