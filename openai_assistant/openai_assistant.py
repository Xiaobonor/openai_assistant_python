# openai_assistant/openai_assistant.py
import time
from .thread_manager import create_thread, list_messages, send_message, delete_thread, \
    create_run, get_runs_by_thread, submit_tool_outputs_and_poll, send_image_with_id
from .file_manager import upload_image
from .tool_functions_map import get_function
from .utils import covert_base64_image


class OpenAIAssistant:
    def __init__(self, assistant_id: str, thread_id: str = None, callback=None):
        self.assistant_id = assistant_id
        self.thread_id = thread_id
        self.callback = callback

    async def initialize_thread_id(self):
        if self.thread_id is None:
            self.thread_id = (await create_thread(messages=[])).id

    async def get_thread_id(self):
        return self.thread_id

    async def delete_current_thread(self):
        return await delete_thread(self.thread_id)

    async def send_request_with_base64(self, message_content: str, base64s: list):
        if len(base64s) <= 0 or len(base64s) > 10:
            raise ValueError("Number of images should be between 1 and 10.")

        images = []
        for base64 in base64s:
            image_io = covert_base64_image(base64)
            images.append(await upload_image(image_io))
            await send_image_with_id(self.thread_id, images[-1].id)

        return await self.send_request(message_content)


    async def send_request(self, message_content: str):
        await self._send_message(message_content)
        await self._initiate_run()
        await self._wait_for_assistant_response()
        return await self._retrieve_latest_assistant_response()

    async def _send_message(self, message_content: str):
        return await send_message(self.thread_id, message_content)

    async def _initiate_run(self):
        return await create_run(self.thread_id, self.assistant_id)

    async def _wait_for_assistant_response(self):
        while True:
            runs = await get_runs_by_thread(self.thread_id)
            running = runs.data[0]
            if running.status == "requires_action":
                tool_outputs = await self._handle_tool_calls(running.required_action.submit_tool_outputs.tool_calls)
                await submit_tool_outputs_and_poll(self.thread_id, running.id, tool_outputs)
                await self._wait_for_assistant_response()
                break
            elif running.status in ["completed", "failed"]:
                break
            time.sleep(2)

    async def _handle_tool_calls(self, tool_calls):
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
                    output = "Got Error while processing this tool call."
                output_list.append({
                    "tool_call_id": tool_call.id,
                    "output": output
                })
        return output_list

    async def _retrieve_latest_assistant_response(self):
        messages = await list_messages(self.thread_id)
        for message in messages.data:
            if message.role == "assistant":
                return message.content[0].text.value
        return None
