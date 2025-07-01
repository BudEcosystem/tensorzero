"""
Comprehensive CI tests for TensorZero OpenAI SDK compatibility.
These tests verify all endpoints work with the dummy provider.
"""

import os
import pytest
from openai import OpenAI
import tempfile

# Initialize client
client = OpenAI(
    base_url=os.getenv("TENSORZERO_BASE_URL", "http://localhost:3001") + "/v1",
    api_key=os.getenv("OPENAI_API_KEY", "dummy-key")
)


class TestChatCompletions:
    """Test chat completions endpoint"""
    
    def test_basic_chat(self):
        """Test basic chat completion"""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert response.id is not None
        assert response.object == "chat.completion"
        assert response.model == "gpt-4"
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        # gpt-4 uses "test" model which returns Megumin response
        assert "Megumin" in response.choices[0].message.content
    
    def test_streaming(self):
        """Test streaming chat completion"""
        stream = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )
        
        chunks = list(stream)
        assert len(chunks) > 1
        assert chunks[0].object == "chat.completion.chunk"


class TestEmbeddings:
    """Test embeddings endpoint"""
    
    def test_single_embedding(self):
        """Test single text embedding"""
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input="Hello world"
        )
        
        assert response.object == "list"
        assert len(response.data) == 1
        assert response.data[0].object == "embedding"
        assert len(response.data[0].embedding) == 1536  # Dummy provider returns 1536 dimensions
        assert response.usage.total_tokens > 0
    
    def test_batch_embeddings(self):
        """Test batch embeddings"""
        texts = ["First text", "Second text", "Third text"]
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        
        assert len(response.data) == 3
        for i, embedding in enumerate(response.data):
            assert embedding.index == i
            assert len(embedding.embedding) == 1536


class TestModeration:
    """Test moderation endpoint"""
    
    def test_safe_content(self):
        """Test moderation of safe content"""
        response = client.moderations.create(
            model="omni-moderation-latest",
            input="Hello, how are you today?"
        )
        
        assert response.id is not None
        assert response.model == "omni-moderation-latest"
        assert len(response.results) == 1
        assert response.results[0].flagged is False
    
    def test_flagged_content(self):
        """Test moderation of content with keywords"""
        response = client.moderations.create(
            model="omni-moderation-latest",
            input="This content contains harmful keywords"
        )
        
        assert len(response.results) == 1
        # Dummy provider flags content containing "harmful"
        assert response.results[0].flagged is True
        assert response.results[0].categories.self_harm is True


class TestAudio:
    """Test audio endpoints"""
    
    @pytest.fixture
    def audio_file(self):
        """Create a temporary audio file for testing"""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            # Write some dummy audio data
            f.write(b"dummy audio content")
            return f.name
    
    def test_transcription(self, audio_file):
        """Test audio transcription"""
        with open(audio_file, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        
        assert response.text == "This is a dummy transcription"
        assert hasattr(response, "language")
        assert hasattr(response, "duration")
    
    def test_translation(self, audio_file):
        """Test audio translation"""
        with open(audio_file, "rb") as f:
            response = client.audio.translations.create(
                model="whisper-1",
                file=f
            )
        
        assert response.text == "This is a dummy translation"
    
    def test_text_to_speech(self):
        """Test text-to-speech"""
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input="Hello world"
        )
        
        # Response should contain audio data
        audio_data = b"".join(response.iter_bytes())
        assert len(audio_data) == 1024  # Dummy provider returns 1KB of data
    
    @pytest.fixture(autouse=True)
    def cleanup(self, audio_file):
        """Clean up temporary files"""
        yield
        if os.path.exists(audio_file):
            os.unlink(audio_file)


@pytest.mark.asyncio
class TestAsyncSupport:
    """Test async client support"""
    
    async def test_async_chat(self):
        """Test async chat completion"""
        from openai import AsyncOpenAI
        
        async_client = AsyncOpenAI(
            base_url=os.getenv("TENSORZERO_BASE_URL", "http://localhost:3001") + "/v1",
            api_key=os.getenv("OPENAI_API_KEY", "dummy-key")
        )
        
        response = await async_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert response.choices[0].message.content is not None
        assert "Megumin" in response.choices[0].message.content
        
        await async_client.close()
    
    async def test_async_embeddings(self):
        """Test async embeddings"""
        from openai import AsyncOpenAI
        
        async_client = AsyncOpenAI(
            base_url=os.getenv("TENSORZERO_BASE_URL", "http://localhost:3001") + "/v1",
            api_key=os.getenv("OPENAI_API_KEY", "dummy-key")
        )
        
        response = await async_client.embeddings.create(
            model="text-embedding-ada-002",
            input="Test embedding"
        )
        
        assert len(response.data[0].embedding) == 1536
        
        await async_client.close()