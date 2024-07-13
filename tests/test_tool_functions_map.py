# tests/test_tool_functions_map.py
import unittest
from unittest.mock import Mock
from openai_assistant.tool_functions_map import (
    get_function, register_function, register_functions,
    unregister_function, replace_function, _FUNCTION_MAP
)


class TestToolFunctionsMap(unittest.TestCase):
    def setUp(self):
        _FUNCTION_MAP.clear()

    def test_get_function(self):
        func = Mock()
        _FUNCTION_MAP["test_func"] = func
        self.assertEqual(get_function("test_func"), func)
        self.assertIsNone(get_function("nonexistent_func"))

    def test_register_function(self):
        func = Mock()
        register_function("test_func", func)
        self.assertIn("test_func", _FUNCTION_MAP)
        self.assertEqual(_FUNCTION_MAP["test_func"], func)

    def test_register_functions(self):
        funcs = {
            "func1": Mock(),
            "func2": Mock()
        }
        register_functions(funcs)
        self.assertIn("func1", _FUNCTION_MAP)
        self.assertIn("func2", _FUNCTION_MAP)
        self.assertEqual(_FUNCTION_MAP["func1"], funcs["func1"])
        self.assertEqual(_FUNCTION_MAP["func2"], funcs["func2"])

    def test_unregister_function(self):
        func = Mock()
        _FUNCTION_MAP["test_func"] = func
        unregister_function("test_func")
        self.assertNotIn("test_func", _FUNCTION_MAP)
        unregister_function("nonexistent_func")  # Should not raise an error

    def test_replace_function(self):
        old_func = Mock()
        new_func = Mock()
        _FUNCTION_MAP["test_func"] = old_func
        replace_function("test_func", new_func)
        self.assertEqual(_FUNCTION_MAP["test_func"], new_func)


if __name__ == '__main__':
    unittest.main()
