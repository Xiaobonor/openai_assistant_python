# tests/test_assistant_manager.py
import unittest
from unittest.mock import AsyncMock, patch, MagicMock
from openai_assistant.assistant_manager import (
    list_assistants, retrieve_assistant, create_assistant,
    update_assistant, delete_assistant, create_assistant_file,
    delete_assistant_file, list_assistant_files, get_assistant_id_by_name
)


class TestAssistantManager(unittest.IsolatedAsyncioTestCase):
    @patch('openai_assistant.assistant_manager.get_openai_client')
    async def test_list_assistants(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        assistant_mock = MagicMock()
        assistant_mock.name = "assistant1"
        assistant_mock.id = "id1"
        mock_client.beta.assistants.list.return_value = AsyncMock(data=[assistant_mock])
        response = await list_assistants()
        mock_client.beta.assistants.list.assert_called_once()
        self.assertEqual(response, {"assistant1": "id1"})

    @patch('openai_assistant.assistant_manager.get_openai_client')
    async def test_retrieve_assistant(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await retrieve_assistant("assistant_id")
        mock_client.beta.assistants.retrieve.assert_called_once_with("assistant_id")
        self.assertIsNotNone(response)

    @patch('openai_assistant.assistant_manager.get_openai_client')
    async def test_create_assistant(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await create_assistant("name", "instructions", ["tool1", "tool2"], "model")
        mock_client.beta.assistants.create.assert_called_once_with(
            name="name", instructions="instructions", tools=["tool1", "tool2"], model="model"
        )
        self.assertIsNotNone(response)

    @patch('openai_assistant.assistant_manager.get_openai_client')
    async def test_update_assistant(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await update_assistant("assistant_id", name="new_name")
        mock_client.beta.assistants.update.assert_called_once_with(
            "assistant_id", name="new_name"
        )
        self.assertIsNotNone(response)

    @patch('openai_assistant.assistant_manager.get_openai_client')
    async def test_delete_assistant(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await delete_assistant("assistant_id")
        mock_client.beta.assistants.delete.assert_called_once_with("assistant_id")
        self.assertIsNotNone(response)

    @patch('openai_assistant.assistant_manager.get_openai_client')
    async def test_create_assistant_file(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await create_assistant_file("assistant_id", "file_id")
        mock_client.beta.assistants.files.create.assert_called_once_with(
            assistant_id="assistant_id", file_id="file_id"
        )
        self.assertIsNotNone(response)

    @patch('openai_assistant.assistant_manager.get_openai_client')
    async def test_delete_assistant_file(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await delete_assistant_file("assistant_id", "file_id")
        mock_client.beta.assistants.files.delete.assert_called_once_with("assistant_id", "file_id")
        self.assertIsNotNone(response)

    @patch('openai_assistant.assistant_manager.get_openai_client')
    async def test_list_assistant_files(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await list_assistant_files("assistant_id")
        mock_client.beta.assistants.files.list.assert_called_once_with("assistant_id")
        self.assertIsNotNone(response)

    @patch('openai_assistant.assistant_manager.list_assistants', new_callable=AsyncMock)
    async def test_get_assistant_id_by_name(self, mock_list_assistants):
        mock_list_assistants.return_value = {"assistant1": "id1"}
        response = await get_assistant_id_by_name("assistant1")
        mock_list_assistants.assert_called_once()
        self.assertEqual(response, "id1")


if __name__ == '__main__':
    unittest.main()
