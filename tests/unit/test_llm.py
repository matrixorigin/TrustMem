"""Tests for MinimalLLMClient and get_llm_client."""

import pytest
from unittest.mock import MagicMock, patch

from memoria.core.llm import MinimalLLMClient, get_llm_client


class TestMinimalLLMClient:
    """Tests for MinimalLLMClient."""

    @pytest.fixture
    def mock_openai(self):
        """Create a mock OpenAI client."""
        with patch("openai.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            yield mock_client

    def test_chat_basic(self, mock_openai):
        """Test basic chat completion."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello, world!"
        mock_openai.chat.completions.create.return_value = mock_response

        client = MinimalLLMClient(api_key="test-key", model="gpt-4o-mini")
        result = client.chat([{"role": "user", "content": "Hi"}])

        assert result == "Hello, world!"
        mock_openai.chat.completions.create.assert_called_once()
        call_args = mock_openai.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4o-mini"
        assert call_args.kwargs["temperature"] == 0.3

    def test_chat_empty_response(self, mock_openai):
        """Test chat with empty response returns empty string."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_openai.chat.completions.create.return_value = mock_response

        client = MinimalLLMClient(api_key="test-key")
        result = client.chat([{"role": "user", "content": "Hi"}])

        assert result == ""

    def test_chat_custom_params(self, mock_openai):
        """Test chat with custom model and temperature."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Custom"
        mock_openai.chat.completions.create.return_value = mock_response

        client = MinimalLLMClient(api_key="test-key", model="custom-model")
        client.chat(
            [{"role": "user", "content": "Hi"}], model="override-model", temperature=0.7
        )

        call_args = mock_openai.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "override-model"
        assert call_args.kwargs["temperature"] == 0.7


class TestChatWithTools:
    """Tests for chat_with_tools method."""

    @pytest.fixture
    def mock_openai(self):
        """Create a mock OpenAI client."""
        with patch("openai.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            yield mock_client

    def test_chat_with_tools_basic(self, mock_openai):
        """Test basic tool calling returns correct format."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Using tool"
        mock_response.choices[0].message.tool_calls = [
            MagicMock(id="call_1", function=MagicMock(name="test_tool", arguments="{}"))
        ]
        mock_openai.chat.completions.create.return_value = mock_response

        client = MinimalLLMClient(api_key="test-key")
        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        result = client.chat_with_tools(
            [{"role": "user", "content": "Call tool"}], tools=tools
        )

        assert result["content"] == "Using tool"
        assert len(result["tool_calls"]) == 1
        mock_openai.chat.completions.create.assert_called_once()
        call_args = mock_openai.chat.completions.create.call_args
        assert call_args.kwargs["tools"] == tools
        assert call_args.kwargs["tool_choice"] == "auto"

    def test_chat_with_tools_no_tools(self, mock_openai):
        """Test chat_with_tools with no tools provided."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "No tools"
        mock_response.choices[0].message.tool_calls = []
        mock_openai.chat.completions.create.return_value = mock_response

        client = MinimalLLMClient(api_key="test-key")
        result = client.chat_with_tools([{"role": "user", "content": "Hi"}])

        assert result["content"] == "No tools"
        assert result["tool_calls"] == []
        call_args = mock_openai.chat.completions.create.call_args
        assert call_args.kwargs["tools"] == []
        assert call_args.kwargs["tool_choice"] == "auto"

    def test_chat_with_tools_custom_tool_choice(self, mock_openai):
        """Test chat_with_tools with custom tool_choice."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_response.choices[0].message.tool_calls = []
        mock_openai.chat.completions.create.return_value = mock_response

        client = MinimalLLMClient(api_key="test-key")
        client.chat_with_tools(
            [{"role": "user", "content": "Hi"}],
            tools=[{"type": "function", "function": {"name": "test"}}],
            tool_choice="required",
        )

        call_args = mock_openai.chat.completions.create.call_args
        assert call_args.kwargs["tool_choice"] == "required"

    def test_chat_with_tools_empty_content(self, mock_openai):
        """Test chat_with_tools handles empty content gracefully."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.tool_calls = None
        mock_openai.chat.completions.create.return_value = mock_response

        client = MinimalLLMClient(api_key="test-key")
        result = client.chat_with_tools([{"role": "user", "content": "Hi"}])

        assert result["content"] == ""
        assert result["tool_calls"] == []

    def test_chat_with_tools_custom_params(self, mock_openai):
        """Test chat_with_tools passes custom model and temperature."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test"
        mock_response.choices[0].message.tool_calls = []
        mock_openai.chat.completions.create.return_value = mock_response

        client = MinimalLLMClient(api_key="test-key", model="default-model")
        client.chat_with_tools(
            [{"role": "user", "content": "Hi"}], model="custom-model", temperature=0.9
        )

        call_args = mock_openai.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "custom-model"
        assert call_args.kwargs["temperature"] == 0.9


class TestGetLLMClient:
    """Tests for get_llm_client function."""

    def test_returns_none_without_api_key(self):
        """Test get_llm_client returns None when no API key configured."""
        with patch("memoria.config.get_settings") as mock_settings:
            mock_settings.return_value.llm_api_key = None
            # Reset cache
            import memoria.core.llm as llm_module

            original_client = llm_module._client
            llm_module._client = None
            try:
                client = get_llm_client()
                assert client is None
            finally:
                llm_module._client = original_client

    def test_caches_client(self):
        """Test get_llm_client caches the client instance."""
        with patch("memoria.config.get_settings") as mock_settings:
            mock_settings.return_value.llm_api_key = "test-key"
            mock_settings.return_value.llm_base_url = None
            mock_settings.return_value.llm_model = "gpt-4o-mini"

            with patch("openai.OpenAI") as mock_openai_class:
                mock_instance = MagicMock()
                mock_openai_class.return_value = mock_instance

                # Reset the module-level cache
                import memoria.core.llm as llm_module

                original_client = llm_module._client
                llm_module._client = None
                try:
                    # First call should create client
                    client1 = get_llm_client()
                    assert client1 is not None
                    assert mock_openai_class.call_count == 1

                    # Second call should return cached client
                    client2 = get_llm_client()
                    assert client2 is client1
                    assert mock_openai_class.call_count == 1  # Not called again
                finally:
                    llm_module._client = original_client

    def test_handles_import_error(self):
        """Test get_llm_client handles openai import error gracefully."""
        with patch("memoria.config.get_settings") as mock_settings:
            mock_settings.return_value.llm_api_key = "test-key"

            with patch("openai.OpenAI") as mock_openai_class:
                mock_openai_class.side_effect = ImportError("openai not installed")

                # Reset the module-level cache
                import memoria.core.llm as llm_module

                original_client = llm_module._client
                llm_module._client = None
                try:
                    client = get_llm_client()
                    assert client is None
                finally:
                    llm_module._client = original_client
