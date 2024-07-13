# openai_assistant/assistant_manager.py
from typing import Optional, List, Dict, Any
from .utils import get_openai_client


async def list_assistants() -> Dict[str, str]:
    """
    List all assistants.

    Returns:
        Dict[str, str]: A dictionary of assistant names and their corresponding IDs.
    """
    assistants = await get_openai_client().beta.assistants.list()
    return {assistant.name: assistant.id for assistant in assistants.data}


async def retrieve_assistant(assistant_id: str) -> Any:
    """
    Retrieve a specific assistant by its ID.

    Args:
        assistant_id (str): The ID of the assistant to retrieve.

    Returns:
        Any: The retrieved assistant.
    """
    return await get_openai_client().beta.assistants.retrieve(assistant_id)


async def create_assistant(name: str, instructions: str, tools: List[str], model: str) -> Any:
    """
    Create a new assistant.

    Args:
        name (str): The name of the assistant.
        instructions (str): The instructions for the assistant.
        tools (List[str]): A list of tools the assistant can use.
        model (str): The model to be used by the assistant.

    Returns:
        Any: The created assistant.
    """
    return await get_openai_client().beta.assistants.create(
        name=name,
        instructions=instructions,
        tools=tools,
        model=model
    )


async def update_assistant(
        assistant_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[List[str]] = None
) -> Any:
    """
    Update an existing assistant.

    Args:
        assistant_id (str): The ID of the assistant to update.
        name (Optional[str]): The new name of the assistant. Defaults to None.
        description (Optional[str]): The new description of the assistant. Defaults to None.
        instructions (Optional[str]): The new instructions for the assistant. Defaults to None.
        tools (Optional[List[str]]): The new list of tools for the assistant. Defaults to None.

    Returns:
        Any: The updated assistant.
    """
    update_fields = {
        'name': name,
        'description': description,
        'instructions': instructions,
        'tools': tools
    }
    update_fields = {k: v for k, v in update_fields.items() if v is not None}
    return await get_openai_client().beta.assistants.update(assistant_id, **update_fields)


async def delete_assistant(assistant_id: str) -> Any:
    """
    Delete an assistant by its ID.

    Args:
        assistant_id (str): The ID of the assistant to delete.

    Returns:
        Any: The response from the delete operation.
    """
    return await get_openai_client().beta.assistants.delete(assistant_id)


async def create_assistant_file(assistant_id: str, file_id: str) -> Any:
    """
    Associate a file with an assistant.

    Args:
        assistant_id (str): The ID of the assistant.
        file_id (str): The ID of the file to associate.

    Returns:
        Any: The response from the create operation.
    """
    return await get_openai_client().beta.assistants.files.create(assistant_id=assistant_id, file_id=file_id)


async def delete_assistant_file(assistant_id: str, file_id: str) -> Any:
    """
    Disassociate a file from an assistant.

    Args:
        assistant_id (str): The ID of the assistant.
        file_id (str): The ID of the file to disassociate.

    Returns:
        Any: The response from the delete operation.
    """
    return await get_openai_client().beta.assistants.files.delete(assistant_id, file_id)


async def list_assistant_files(assistant_id: str) -> Any:
    """
    List all files associated with an assistant.

    Args:
        assistant_id (str): The ID of the assistant.

    Returns:
        Any: The list of files associated with the assistant.
    """
    return await get_openai_client().beta.assistants.files.list(assistant_id)


async def get_assistant_id_by_name(name: str) -> Optional[str]:
    """
    Get the ID of an assistant by its name.

    Args:
        name (str): The name of the assistant.

    Returns:
        Optional[str]: The ID of the assistant, or None if not found.
    """
    assistants = await list_assistants()
    return assistants.get(name)
