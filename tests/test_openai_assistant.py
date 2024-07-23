# tests/test_openai_assistant.py
import unittest
from unittest.mock import AsyncMock, patch, mock_open, MagicMock
from openai_assistant.openai_assistant import OpenAIAssistant
from io import BytesIO


class TestOpenAIAssistant(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.assistant = OpenAIAssistant(assistant_id="test_assistant")

    @patch('openai_assistant.openai_assistant.create_thread', AsyncMock())
    async def test_initialize_thread_id(self):
        await self.assistant.initialize_thread_id()
        self.assertIsNotNone(self.assistant.thread_id)

    @patch('openai_assistant.openai_assistant.delete_thread', AsyncMock())
    async def test_delete_current_thread(self):
        self.assistant.thread_id = "test_thread"
        response = await self.assistant.delete_current_thread()
        self.assertIsNotNone(response)

    @patch('openai_assistant.openai_assistant.upload_file', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.send_image_with_id', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.send_message', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.create_run', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.OpenAIAssistant._wait_for_assistant_response', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.OpenAIAssistant._retrieve_latest_assistant_response', new_callable=AsyncMock, return_value="response")
    @patch('builtins.open', new_callable=mock_open, read_data=b'test data')
    async def test_send_request_with_file(self, mock_file, mock_retrieve, mock_wait, mock_create_run, mock_send_message, mock_send_image, mock_upload):
        self.assistant.thread_id = "test_thread"
        response = await self.assistant.send_request_with_file("message_content", "test_file_path")
        self.assertEqual(response, "response")

    @patch('openai_assistant.openai_assistant.upload_file', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.send_image_with_id', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.send_message', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.create_run', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.OpenAIAssistant._wait_for_assistant_response', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.OpenAIAssistant._retrieve_latest_assistant_response', new_callable=AsyncMock, return_value="response")
    async def test_send_request_image_io(self, mock_retrieve, mock_wait, mock_create_run, mock_send_message, mock_send_image, mock_upload):
        self.assistant.thread_id = "test_thread"
        response = await self.assistant.send_request_image_io("message_content", BytesIO(b"image data"))
        self.assertEqual(response, "response")

    @patch('openai_assistant.openai_assistant.upload_file', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.send_image_with_id', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.send_message', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.create_run', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.OpenAIAssistant._wait_for_assistant_response', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.OpenAIAssistant._retrieve_latest_assistant_response', new_callable=AsyncMock, return_value="response")
    async def test_send_request_image_base64(self, mock_retrieve, mock_wait, mock_create_run, mock_send_message, mock_send_image, mock_upload):
        self.assistant.thread_id = "test_thread"
        response = await self.assistant.send_request_image_base64("message_content", ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA"])
        self.assertEqual(response, "response")

    @patch('openai_assistant.openai_assistant.upload_file', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.send_image_with_url', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.send_message', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.create_run', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.OpenAIAssistant._wait_for_assistant_response', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.OpenAIAssistant._retrieve_latest_assistant_response', new_callable=AsyncMock, return_value="response")
    async def test_send_request_image_url(self, mock_retrieve, mock_wait, mock_create_run, mock_send_message, mock_send_image, mock_upload):
        self.assistant.thread_id = "test_thread"
        response = await self.assistant.send_request_image_url("message_content", ["http://example.com/image.png"])
        self.assertEqual(response, "response")

    @patch('openai_assistant.openai_assistant.send_message', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.create_run', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.OpenAIAssistant._wait_for_assistant_response', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.OpenAIAssistant._retrieve_latest_assistant_response', new_callable=AsyncMock, return_value="response")
    async def test_send_request(self, mock_retrieve, mock_wait, mock_create_run, mock_send_message):
        self.assistant.thread_id = "test_thread"
        response = await self.assistant.send_request("message_content")
        self.assertEqual(response, "response")

    @patch('openai_assistant.openai_assistant.upload_file', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.send_image_with_id', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.send_message', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.create_run', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.OpenAIAssistant._wait_for_assistant_response', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.OpenAIAssistant._retrieve_latest_assistant_response', new_callable=AsyncMock, return_value="response")
    @patch('builtins.open', new_callable=mock_open, read_data=b'test data')
    async def test_send_request_with_upload_file(self, mock_file, mock_retrieve, mock_wait, mock_create_run, mock_send_message, mock_send_image, mock_upload):
        self.assistant.thread_id = "test_thread"
        response = await self.assistant._send_request_with_upload("message_content", "test_file_path", "file")
        self.assertEqual(response, "response")

    @patch('openai_assistant.openai_assistant.upload_file', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.send_image_with_id', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.send_message', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.create_run', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.OpenAIAssistant._wait_for_assistant_response', new_callable=AsyncMock)
    @patch('openai_assistant.openai_assistant.OpenAIAssistant._retrieve_latest_assistant_response', new_callable=AsyncMock, return_value="response")
    async def test_send_request_with_upload_io(self, mock_retrieve, mock_wait, mock_create_run, mock_send_message, mock_send_image, mock_upload):
        self.assistant.thread_id = "test_thread"
        response = await self.assistant._send_request_with_upload("message_content", BytesIO(b"image data"), "vision")
        self.assertEqual(response, "response")


if __name__ == '__main__':
    unittest.main()
