# tests/test_vector_stores_manager.py
import unittest
from unittest.mock import AsyncMock, patch
from openai_assistant.vector_stores_manager import (
    create_vector_store, list_vector_stores, retrieve_vector_store,
    modify_vector_store, delete_vector_store
)


class TestVectorStoresManager(unittest.IsolatedAsyncioTestCase):
    @patch('openai_assistant.vector_stores_manager.get_openai_client')
    async def test_create_vector_store(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await create_vector_store("test_store")
        mock_client.beta.vector_stores.create.assert_called_once_with(name="test_store")
        self.assertIsNotNone(response)

    @patch('openai_assistant.vector_stores_manager.get_openai_client')
    async def test_list_vector_stores(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await list_vector_stores()
        mock_client.beta.vector_stores.list.assert_called_once()
        self.assertIsNotNone(response)

    @patch('openai_assistant.vector_stores_manager.get_openai_client')
    async def test_retrieve_vector_store(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await retrieve_vector_store("store_id")
        mock_client.beta.vector_stores.retrieve.assert_called_once_with("store_id")
        self.assertIsNotNone(response)

    @patch('openai_assistant.vector_stores_manager.get_openai_client')
    async def test_modify_vector_store(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await modify_vector_store("store_id", "new_name")
        mock_client.beta.vector_stores.update.assert_called_once_with("store_id", name="new_name")
        self.assertIsNotNone(response)

    @patch('openai_assistant.vector_stores_manager.get_openai_client')
    async def test_delete_vector_store(self, mock_get_openai_client):
        mock_client = AsyncMock()
        mock_get_openai_client.return_value = mock_client
        response = await delete_vector_store("store_id")
        mock_client.beta.vector_stores.delete.assert_called_once_with("store_id")
        self.assertIsNotNone(response)


if __name__ == '__main__':
    unittest.main()
