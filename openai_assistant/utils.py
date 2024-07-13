# openai_assistant/utils.py
import base64
import io

_assistant_vision_support = None
_openai_client = None


def init(openai_client, assistant_vision_support=True):
    """
    Initialize the OpenAI client and assistant vision support.

    Args:
        openai_client: The OpenAI client instance.
        assistant_vision_support (bool): Flag to support vision functionality. Defaults to True.
    """
    global _openai_client, _assistant_vision_support
    _openai_client = openai_client
    _assistant_vision_support = assistant_vision_support


def get_openai_client():
    """
    Get the initialized OpenAI client.
    Only for this package's internal use.

    Returns:
        The OpenAI client instance.

    Raises:
        ValueError: If the OpenAI client is not initialized.
    """
    if _openai_client is None:
        raise ValueError("OpenAI client is not initialized. Call `init(openai_client)` first.")
    return _openai_client


def convert_base64_image(base64_image):
    """
    Convert a base64 encoded image to a BytesIO object.

    Args:
        base64_image (str): The base64 encoded image string.

    Returns:
        tuple: A tuple containing the BytesIO object and the image mimetype.

    Raises:
        ValueError: If the image type is unsupported.
    """
    if base64_image.startswith('data:image/png;base64,'):
        base64_image_cleaned = base64_image[len('data:image/png;base64,'):]
        mimetype = 'image/png'
        extension = 'png'
    elif base64_image.startswith('data:image/jpeg;base64,') or base64_image.startswith('data:image/jpg;base64,'):
        base64_image_cleaned = base64_image.split('base64,')[1]
        mimetype = 'image/jpeg'
        extension = 'jpeg'
    elif base64_image.startswith('data:image/webp;base64,'):
        base64_image_cleaned = base64_image[len('data:image/webp;base64,'):]
        mimetype = 'image/webp'
        extension = 'webp'
    elif base64_image.startswith('data:image/gif;base64,'):
        base64_image_cleaned = base64_image[len('data:image/gif;base64,'):]
        mimetype = 'image/gif'
        extension = 'gif'
    else:
        raise ValueError("Unsupported image type")

    # Ensure base64 string is properly padded
    missing_padding = len(base64_image_cleaned) % 4
    if missing_padding:
        base64_image_cleaned += '=' * (4 - missing_padding)

    image_data = base64.b64decode(base64_image_cleaned)
    image_io = io.BytesIO(image_data)
    image_io.name = f'image.{extension}'
    return image_io, mimetype
