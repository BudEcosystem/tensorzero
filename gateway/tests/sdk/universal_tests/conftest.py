"""Shared fixtures for universal tests."""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import create_universal_client, UniversalTestData


@pytest.fixture(scope="session")
def universal_client():
    """Create universal client for all tests."""
    return create_universal_client()


@pytest.fixture(scope="session") 
def test_models():
    """Get test models for all providers."""
    return UniversalTestData.get_provider_models()


@pytest.fixture(scope="session")
def embedding_models():
    """Get embedding models for all providers."""
    return UniversalTestData.get_embedding_models()


@pytest.fixture
def basic_messages():
    """Get basic chat messages."""
    return UniversalTestData.get_basic_chat_messages()


@pytest.fixture
def multi_turn_messages():
    """Get multi-turn conversation messages."""
    return UniversalTestData.get_multi_turn_messages()


@pytest.fixture  
def system_prompt_messages():
    """Get messages with system prompt."""
    return UniversalTestData.get_system_prompt_messages()


@pytest.fixture
def test_prompts():
    """Get test prompts."""
    return UniversalTestData.get_test_prompts()


@pytest.fixture
def embedding_texts():
    """Get texts for embedding tests."""
    return UniversalTestData.get_embedding_texts()


@pytest.fixture(params=["gpt-3.5-turbo", "claude-3-haiku-20240307", "meta-llama/Llama-3.2-3B-Instruct-Turbo"])
def cross_provider_model(request):
    """Parametrized fixture for testing across providers."""
    return request.param


@pytest.fixture(params=["text-embedding-3-small", "together-bge-base"])
def cross_provider_embedding_model(request):
    """Parametrized fixture for embedding tests across providers."""
    return request.param