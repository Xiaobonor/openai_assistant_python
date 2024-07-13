# openai_assistant/file_manager.py
from .utils import get_openai_client


async def upload_image(image):
    return await get_openai_client().files.create(file=image, purpose="vision")


async def delete_file(file_id: str):
    return await get_openai_client().files.delete(file_id)


async def list_files():
    return await get_openai_client().files.list()


async def retrieve_file(file_id: str):
    return await get_openai_client().files.retrieve(file_id)


async def update_file(file_id: str, file):
    await delete_file(file_id)
    return await upload_image(file)