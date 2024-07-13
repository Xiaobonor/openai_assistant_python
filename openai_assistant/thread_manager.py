# openai_assistant/thread_manager.py
from typing import Optional, List, Dict, Any
from .utils import get_openai_client


async def list_messages(
        thread_id: str,
        limit: int = 20,
        order: str = 'desc',
        after: Optional[str] = None,
        before: Optional[str] = None
) -> Optional[Any]:
    """
    List messages from a thread.

    Args:
        thread_id (str): The ID of the thread.
        limit (int): The maximum number of messages to retrieve. Defaults to 20.
        order (str): The order in which to retrieve messages. Defaults to 'desc'.
        after (Optional[str]): The timestamp after which to retrieve messages.
        before (Optional[str]): The timestamp before which to retrieve messages.

    Returns:
        Optional[Any]: The list of messages, or None if an error occurs.
    """
    try:
        return await get_openai_client().beta.threads.messages.list(
            thread_id=thread_id, limit=limit, order=order, after=after, before=before
        )
    except Exception as e:
        print(f"An error occurred while retrieving messages: {e}")
        return None


async def retrieve_message(thread_id: str, message_id: str) -> Any:
    """
    Retrieve a specific message from a thread.

    Args:
        thread_id (str): The ID of the thread.
        message_id (str): The ID of the message.

    Returns:
        Any: The retrieved message.
    """
    return await get_openai_client().beta.threads.messages.retrieve(thread_id=thread_id, message_id=message_id)


async def create_thread(messages: Optional[List[Dict[str, Any]]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Any:
    """
    Create a new thread.

    Args:
        messages (Optional[List[Dict[str, Any]]]): Initial messages for the thread. Defaults to None.
        metadata (Optional[Dict[str, Any]]): Metadata for the thread. Defaults to None.

    Returns:
        Any: The created thread.
    """
    return await get_openai_client().beta.threads.create(messages=messages, metadata=metadata)


async def retrieve_thread(thread_id: str) -> Any:
    """
    Retrieve a specific thread.

    Args:
        thread_id (str): The ID of the thread.

    Returns:
        Any: The retrieved thread.
    """
    return await get_openai_client().beta.threads.retrieve(thread_id)


async def modify_thread(thread_id: str, metadata: Dict[str, Any]) -> Any:
    """
    Modify a thread's metadata.

    Args:
        thread_id (str): The ID of the thread.
        metadata (Dict[str, Any]): The new metadata for the thread.

    Returns:
        Any: The modified thread.
    """
    return await get_openai_client().beta.threads.modify(thread_id, metadata=metadata)


async def delete_thread(thread_id: str) -> Any:
    """
    Delete a specific thread.

    Args:
        thread_id (str): The ID of the thread.

    Returns:
        Any: The response from the delete operation.
    """
    return await get_openai_client().beta.threads.delete(thread_id)


async def send_message(thread_id: str, content: str, role: str = "user") -> Any:
    """
    Send a message to a thread.

    Args:
        thread_id (str): The ID of the thread.
        content (str): The content of the message.
        role (str): The role of the message sender. Defaults to "user".

    Returns:
        Any: The sent message.
    """
    return await get_openai_client().beta.threads.messages.create(thread_id=thread_id, role=role, content=content)


async def send_image_with_id(thread_id: str, image_id: str, role: str = "user") -> Any:
    """
    Send an image to a thread using its ID.

    Args:
        thread_id (str): The ID of the thread.
        image_id (str): The ID of the image.
        role (str): The role of the message sender. Defaults to "user".

    Returns:
        Any: The sent image message.
    """
    return await get_openai_client().beta.threads.messages.create(
        thread_id=thread_id, role=role,
        content=[{"type": "image_file", "image_file": {"file_id": image_id}}]
    )


async def send_image_with_url(thread_id: str, image_url: str, role: str = "user") -> Any:
    """
    Send an image to a thread using its URL.

    Args:
        thread_id (str): The ID of the thread.
        image_url (str): The URL of the image.
        role (str): The role of the message sender. Defaults to "user".

    Returns:
        Any: The sent image message.
    """
    return await get_openai_client().beta.threads.messages.create(
        thread_id=thread_id, role=role,
        content=[{"type": "image_url", "image_url": {"url": image_url}}]
    )


async def create_run(thread_id: str, assistant_id: str) -> Any:
    """
    Create a new run for a thread.

    Args:
        thread_id (str): The ID of the thread.
        assistant_id (str): The ID of the assistant.

    Returns:
        Any: The created run.
    """
    return await get_openai_client().beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)


async def get_runs_by_thread(thread_id: str) -> Any:
    """
    Get all runs associated with a thread.

    Args:
        thread_id (str): The ID of the thread.

    Returns:
        Any: The list of runs.
    """
    return await get_openai_client().beta.threads.runs.list(thread_id=thread_id)


async def submit_tool_outputs_and_poll(thread_id: str, run_id: str, tool_outputs: List[Dict[str, Any]]) -> Any:
    """
    Submit tool outputs and poll for the result.

    Args:
        thread_id (str): The ID of the thread.
        run_id (str): The ID of the run.
        tool_outputs (List[Dict[str, Any]]): The tool outputs to submit.

    Returns:
        Any: The result of the submission and polling.
    """
    return await get_openai_client().beta.threads.runs.submit_tool_outputs_and_poll(
        thread_id=thread_id,
        run_id=run_id,
        tool_outputs=tool_outputs
    )
