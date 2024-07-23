# openai_assistant/vector_stores_manager.py
from typing import Any
from .utils import get_openai_client


async def create_vector_store(name: str) -> Any:
    """
    Create a new vector store.

    Args:
        name (str): The name of the vector store.

    Returns:
        Any: The created vector store.
    """
    return await get_openai_client().beta.vector_stores.create(name=name)


async def list_vector_stores() -> Any:
    """
    List all vector stores.

    Returns:
        Any: The list of vector stores.
    """
    return await get_openai_client().beta.vector_stores.list()


async def upload_and_poll_vector_store_data(vector_store_id: str, files_path: str) -> Any:
    """
    Upload and poll data to a vector store.

    Args:
        vector_store_id (str): The ID of the vector store.
        files_path (str): The path to the file to upload.

    Returns:
        Any: The response from the upload operation.
    """
    file_streams = [open(path, "rb") for path in files_path]
    return await get_openai_client().beta.vector_stores.upload_and_poll_data(vector_store_id, file_streams)


async def retrieve_vector_store(vector_store_id: str) -> Any:
    """
    Retrieve a specific vector store by its ID.

    Args:
        vector_store_id (str): The ID of the vector store to retrieve.

    Returns:
        Any: The retrieved vector store.
    """
    return await get_openai_client().beta.vector_stores.retrieve(vector_store_id)


async def modify_vector_store(vector_store_id: str, name: str) -> Any:
    """
    Modify a vector store's name and description.

    Args:
        vector_store_id (str): The ID of the vector store.
        name (str): The new name for the vector store.

    Returns:
        Any: The modified vector store.
    """
    return await get_openai_client().beta.vector_stores.update(vector_store_id, name=name)


async def delete_vector_store(vector_store_id: str) -> Any:
    """
    Delete a vector store.

    Args:
        vector_store_id (str): The ID of the vector store to delete.

    Returns:
        Any: The response from the delete operation.
    """
    return await get_openai_client().beta.vector_stores.delete(vector_store_id)
