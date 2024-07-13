# tests/test_thread_manager.py
import unittest
from unittest.mock import AsyncMock, patch
from openai_assistant.thread_manager import (
    list_messages, retrieve_message, create_thread, retrieve_thread,
    modify_thread, delete_thread, send_message, send_image_with_id,
    send_image_with_url, create_run, get_runs_by_thread, submit_tool_outputs_and_poll
)


class TestThreadManager(unittest.IsolatedAsyncioTestCase):
    @patch('openai_assistant.thread_manager.get_openai_client')
    async def test_list_messages(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await list_messages("thread_id")
        mock_client.beta.threads.messages.list.assert_called_once()
        self.assertIsNotNone(response)

    @patch('openai_assistant.thread_manager.get_openai_client')
    async def test_retrieve_message(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await retrieve_message("thread_id", "message_id")
        mock_client.beta.threads.messages.retrieve.assert_called_once_with(thread_id="thread_id",
                                                                           message_id="message_id")
        self.assertIsNotNone(response)

    @patch('openai_assistant.thread_manager.get_openai_client')
    async def test_create_thread(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await create_thread()
        mock_client.beta.threads.create.assert_called_once()
        self.assertIsNotNone(response)

    @patch('openai_assistant.thread_manager.get_openai_client')
    async def test_retrieve_thread(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await retrieve_thread("thread_id")
        mock_client.beta.threads.retrieve.assert_called_once_with("thread_id")
        self.assertIsNotNone(response)

    @patch('openai_assistant.thread_manager.get_openai_client')
    async def test_modify_thread(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await modify_thread("thread_id", {"metadata": "new"})
        mock_client.beta.threads.modify.assert_called_once_with("thread_id", metadata={"metadata": "new"})
        self.assertIsNotNone(response)

    @patch('openai_assistant.thread_manager.get_openai_client')
    async def test_delete_thread(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await delete_thread("thread_id")
        mock_client.beta.threads.delete.assert_called_once_with("thread_id")
        self.assertIsNotNone(response)

    @patch('openai_assistant.thread_manager.get_openai_client')
    async def test_send_message(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await send_message("thread_id", "content")
        mock_client.beta.threads.messages.create.assert_called_once_with(thread_id="thread_id", role="user",
                                                                         content="content")
        self.assertIsNotNone(response)

    @patch('openai_assistant.thread_manager.get_openai_client')
    async def test_send_image_with_id(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await send_image_with_id("thread_id", "image_id")
        mock_client.beta.threads.messages.create.assert_called_once()
        self.assertIsNotNone(response)

    @patch('openai_assistant.thread_manager.get_openai_client')
    async def test_send_image_with_url(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await send_image_with_url("thread_id", "image_url")
        mock_client.beta.threads.messages.create.assert_called_once()
        self.assertIsNotNone(response)

    @patch('openai_assistant.thread_manager.get_openai_client')
    async def test_create_run(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await create_run("thread_id", "assistant_id")
        mock_client.beta.threads.runs.create.assert_called_once_with(thread_id="thread_id", assistant_id="assistant_id")
        self.assertIsNotNone(response)

    @patch('openai_assistant.thread_manager.get_openai_client')
    async def test_get_runs_by_thread(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await get_runs_by_thread("thread_id")
        mock_client.beta.threads.runs.list.assert_called_once_with(thread_id="thread_id")
        self.assertIsNotNone(response)

    @patch('openai_assistant.thread_manager.get_openai_client')
    async def test_submit_tool_outputs_and_poll(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        tool_outputs = [{"tool_call_id": "id", "output": "output"}]
        response = await submit_tool_outputs_and_poll("thread_id", "run_id", tool_outputs)
        mock_client.beta.threads.runs.submit_tool_outputs_and_poll.assert_called_once_with(
            thread_id="thread_id", run_id="run_id", tool_outputs=tool_outputs
        )
        self.assertIsNotNone(response)


if __name__ == '__main__':
    unittest.main()
