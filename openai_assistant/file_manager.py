# openai_assistant/file_manager.py
from .utils import get_openai_client


async def upload_image(image):
    return await get_openai_client().files.create(file=image, purpose="assistants")
