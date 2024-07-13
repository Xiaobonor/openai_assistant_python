# tests/test_utils.py
import unittest
from unittest.mock import Mock, patch
from io import BytesIO
from openai_assistant.utils import init, get_openai_client, convert_base64_image


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.mock_openai_client = Mock()

    def test_init(self):
        init(self.mock_openai_client)
        self.assertIs(get_openai_client(), self.mock_openai_client)

    def test_get_openai_client_not_initialized(self):
        with self.assertRaises(ValueError):
            get_openai_client()

    def test_convert_base64_image_png(self):
        base64_image = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA'
        image_io, mimetype = convert_base64_image(base64_image)
        self.assertIsInstance(image_io, BytesIO)
        self.assertEqual(mimetype, 'image/png')

    def test_convert_base64_image_invalid(self):
        base64_image = 'data:image/xyz;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA'
        with self.assertRaises(ValueError):
            convert_base64_image(base64_image)


if __name__ == '__main__':
    unittest.main()
