import unittest

from models.llm.protocols.openai_compatible import OpenAICompatibleProtocol


class TestOpenAICompatibleProtocol(unittest.TestCase):
    def test_wraps_reasoning_content_field(self):
        content, is_reasoning = OpenAICompatibleProtocol()._wrap_thinking_by_reasoning_content(
            {"reasoning_content": "thinking", "content": ""},
            False,
        )

        self.assertEqual(content, "<think>thinking")
        self.assertTrue(is_reasoning)

    def test_wraps_reasoning_field(self):
        content, is_reasoning = OpenAICompatibleProtocol()._wrap_thinking_by_reasoning_content(
            {"reasoning": "thinking", "content": ""},
            False,
        )

        self.assertEqual(content, "<think>thinking")
        self.assertTrue(is_reasoning)


if __name__ == "__main__":
    unittest.main()
