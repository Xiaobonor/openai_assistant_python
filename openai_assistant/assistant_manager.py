# openai_assistant/assistant_manager.py
from typing import Optional
from . import get_openai_client


async def list_assistants():
    return {assistant.name: assistant.id for assistant in (await get_openai_client().beta.assistants.list()).data}


async def retrieve_assistant(assistant_id: str):
    return await get_openai_client().beta.assistants.retrieve(assistant_id)


async def create_assistant(name: str, instructions: str, tools: list, model: str):
    return await get_openai_client().beta.assistants.create(
        name=name,
        instructions=instructions,
        tools=tools,
        model=model
    )


async def update_assistant(assistant_id: str, name: Optional[str] = None, description: Optional[str] = None,
                           instructions: Optional[str] = None, tools: Optional[list] = None):
    update_fields = {k: v for k, v in {
        'name': name,
        'description': description,
        'instructions': instructions,
        'tools': tools
    }.items() if v is not None}
    return await get_openai_client().beta.assistants.update(assistant_id, **update_fields)


async def delete_assistant(assistant_id: str):
    return await get_openai_client().beta.assistants.delete(assistant_id)


async def create_assistant_file(assistant_id: str, file_id: str):
    return await get_openai_client().beta.assistants.files.create(assistant_id=assistant_id, file_id=file_id)


async def delete_assistant_file(assistant_id: str, file_id: str):
    return await get_openai_client().beta.assistants.files.delete(assistant_id, file_id)


async def list_assistant_files(assistant_id: str):
    return await get_openai_client().beta.assistants.files.list(assistant_id)


async def get_assistant_id_by_name(name: str):
    assistants = await list_assistants()
    return assistants.get(name)
