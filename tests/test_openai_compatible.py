import unittest
from unittest.mock import MagicMock

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

    def test_not_supported_preserves_top_level_enable_thinking(self):
        protocol = OpenAICompatibleProtocol()
        captured = {}

        def fake_generate(_model, _credentials, _prompt_messages, params, *_args, **_kwargs):
            captured["model_parameters"] = params
            return MagicMock()

        protocol._generate = fake_generate
        protocol._filter_thinking_result = lambda result: result

        protocol.generate(
            model="test-model",
            credentials={"agent_thought_support": "not_supported"},
            prompt_messages=[],
            model_parameters={},
            stream=False,
        )

        params = captured["model_parameters"]
        self.assertIs(params["enable_thinking"], False)
        self.assertNotIn("chat_template_kwargs", params)


if __name__ == "__main__":
    unittest.main()
