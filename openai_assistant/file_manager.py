# openai_assistant/file_manager.py
from typing import Any
from .utils import get_openai_client


async def upload_file(file, purpose="assistants") -> Any:
    """
    Upload a file to OpenAI.

    Args:
        file: The file to upload.
        purpose: The purpose of the file.

    Returns:
        Any: The response from the upload operation.
    """
    return await get_openai_client().files.create(file=file, purpose=purpose)


async def delete_file(file_id: str) -> Any:
    """
    Delete a file from OpenAI.

    Args:
        file_id (str): The ID of the file to delete.

    Returns:
        Any: The response from the delete operation.
    """
    return await get_openai_client().files.delete(file_id)


async def list_files() -> Any:
    """
    List all files in OpenAI.

    Returns:
        Any: The list of files.
    """
    return await get_openai_client().files.list()


async def retrieve_file(file_id: str) -> Any:
    """
    Retrieve a specific file from OpenAI.

    Args:
        file_id (str): The ID of the file to retrieve.

    Returns:
        Any: The retrieved file.
    """
    return await get_openai_client().files.retrieve(file_id)


async def update_file(file_id: str, file, purpose="assistants") -> Any:
    """
    Update a file in OpenAI by deleting the old file and uploading a new one.

    Args:
        file_id (str): The ID of the file to update.
        file: The new file to upload.

    Returns:
        Any: The response from the upload operation.
    """
    await delete_file(file_id)
    return await upload_file(file, purpose)
