# tests/test_file_manager.py
import unittest
from unittest.mock import AsyncMock, patch
from openai_assistant.file_manager import upload_file, delete_file, list_files, retrieve_file, update_file


class TestFileManager(unittest.IsolatedAsyncioTestCase):
    @patch('openai_assistant.file_manager.get_openai_client')
    async def test_upload_file(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        file = bytes(b"test data")
        response = await upload_file(file)
        mock_client.files.create.assert_called_once_with(file=file, purpose='assistants')
        self.assertIsNotNone(response)

    @patch('openai_assistant.file_manager.get_openai_client')
    async def test_delete_file(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await delete_file("file_id")
        mock_client.files.delete.assert_called_once_with("file_id")
        self.assertIsNotNone(response)

    @patch('openai_assistant.file_manager.get_openai_client')
    async def test_list_files(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await list_files()
        mock_client.files.list.assert_called_once()
        self.assertIsNotNone(response)

    @patch('openai_assistant.file_manager.get_openai_client')
    async def test_retrieve_file(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await retrieve_file("file_id")
        mock_client.files.retrieve.assert_called_once_with("file_id")
        self.assertIsNotNone(response)

    @patch('openai_assistant.file_manager.delete_file', AsyncMock())
    @patch('openai_assistant.file_manager.upload_file', AsyncMock())
    async def test_update_file(self):
        response = await update_file("file_id", bytes(b"new data"))
        self.assertIsNotNone(response)


if __name__ == '__main__':
    unittest.main()
