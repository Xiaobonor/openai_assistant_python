# openai_assistant/tool_functions_map.py
from typing import Callable, Dict

_FUNCTION_MAP: Dict[str, Callable] = {}


def get_function(name: str) -> Callable:
    """
    Retrieve a function by its name.

    Args:
        name (str): The name of the function to retrieve.

    Returns:
        Callable: The function associated with the given name, or None if not found.
    """
    return _FUNCTION_MAP.get(name)


def list_functions() -> Dict[str, Callable]:
    """
    List all registered functions.

    Returns:
        Dict[str, Callable]: A dictionary of function names and their corresponding functions.
    """
    return _FUNCTION_MAP.copy()


def register_function(name: str, function: Callable):
    """
    Register a single function to be used in the assistant.

    Args:
        name (str): The name of the function.
        function (Callable): The function to register.
    """
    _FUNCTION_MAP[name] = function


def register_functions(functions: Dict[str, Callable]):
    """
    Register multiple functions to be used in the assistant.

    Args:
        functions (Dict[str, Callable]): A dictionary of function names and their corresponding functions.
    """
    _FUNCTION_MAP.update(functions)


def unregister_function(name: str):
    """
    Unregister a function by its name.

    Args:
        name (str): The name of the function to unregister.
    """
    _FUNCTION_MAP.pop(name, None)


def replace_function(name: str, function: Callable):
    """
    Replace an existing function with a new one.

    Args:
        name (str): The name of the function to replace.
        function (Callable): The new function to replace the old one.
    """
    _FUNCTION_MAP[name] = function
