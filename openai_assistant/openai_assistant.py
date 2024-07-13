# openai_assistant/openai_assistant.py
import time
from .thread_manager import (
    create_thread, list_messages, send_message, delete_thread,
    create_run, get_runs_by_thread, submit_tool_outputs_and_poll,
    send_image_with_id, send_image_with_url
)
from .file_manager import upload_image
from .tool_functions_map import get_function
from .utils import convert_base64_image


class OpenAIAssistant:
    def __init__(self, assistant_id: str, thread_id: str = None, callback=None):
        """
        Initialize the OpenAIAssistant instance.

        Args:
            assistant_id (str): The ID of the assistant.
            thread_id (str, optional): The ID of the thread. Defaults to None.
            callback (callable, optional): Callback function. Defaults to None.
        """
        self.assistant_id = assistant_id
        self.thread_id = thread_id
        self.callback = callback

    async def initialize_thread_id(self):
        """Initialize the thread ID if it is not set."""
        if self.thread_id is None:
            self.thread_id = (await create_thread(messages=[])).id

    async def get_thread_id(self):
        """Get the current thread ID.

        Returns:
            str: The current thread ID.
        """
        return self.thread_id

    async def delete_current_thread(self):
        """Delete the current thread.

        Returns:
            Response: The response from the delete_thread call.
        """
        return await delete_thread(self.thread_id)

    async def send_request_with_base64(self, message_content: str, base64_images: list):
        """
        Send a request with base64 encoded images.

        Args:
            message_content (str): The content of the message.
            base64_images (list): A list of base64 encoded images.

        Raises:
            ValueError: If the number of images is not between 1 and 10.

        Returns:
            Response: The response from the send_request call.
        """
        if not (1 <= len(base64_images) <= 10):
            raise ValueError("Number of images should be between 1 and 10.")

        for base64_image in base64_images:
            image_io, mimetype = convert_base64_image(base64_image)
            image = await upload_image(image_io)
            await send_image_with_id(self.thread_id, image.id)

        return await self.send_request(message_content)

    async def send_request_with_url(self, message_content: str, urls: list):
        """
        Send a request with image URLs.

        Args:
            message_content (str): The content of the message.
            urls (list): A list of image URLs.

        Raises:
            ValueError: If the number of URLs is not between 1 and 10.

        Returns:
            Response: The response from the send_request call.
        """
        if not (1 <= len(urls) <= 10):
            raise ValueError("Number of images should be between 1 and 10.")

        for url in urls:
            await send_image_with_url(self.thread_id, url)

        return await self.send_request(message_content)

    async def send_request(self, message_content: str):
        """
        Send a request with the given message content.

        Args:
            message_content (str): The content of the message.

        Returns:
            str: The latest assistant response.
        """
        await send_message(self.thread_id, message_content)
        await create_run(self.thread_id, self.assistant_id)
        await self._wait_for_assistant_response()
        return await self._retrieve_latest_assistant_response()

    async def _wait_for_assistant_response(self):
        """Wait for the assistant's response, handling tool calls if required."""
        while True:
            runs = await get_runs_by_thread(self.thread_id)
            current_run = runs.data[0]
            if current_run.status == "requires_action":
                tool_outputs = await self._handle_tool_calls(current_run.required_action.submit_tool_outputs.tool_calls)
                await submit_tool_outputs_and_poll(self.thread_id, current_run.id, tool_outputs)
            elif current_run.status in ["completed", "failed"]:
                break
            time.sleep(2)

    async def _handle_tool_calls(self, tool_calls):
        """
        Handle tool calls required by the assistant.

        Args:
            tool_calls (list): A list of tool calls.

        Returns:
            list: A list of tool outputs.
        """
        output_list = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = tool_call.function.arguments

            if isinstance(function_args, str):
                function_args = eval(function_args.replace("true", "True").replace("false", "False"))

            function = get_function(function_name)
            if function:
                output = await function(**function_args)
                if output is None:
                    output = "Error processing tool call."
                output_list.append({
                    "tool_call_id": tool_call.id,
                    "output": output
                })
        return output_list

    async def _retrieve_latest_assistant_response(self):
        """Retrieve the latest response from the assistant.

        Returns:
            str: The latest assistant response, or None if not found.
        """
        messages = await list_messages(self.thread_id)
        for message in messages.data:
            if message.role == "assistant":
                return message.content[0].text.value
        return None
