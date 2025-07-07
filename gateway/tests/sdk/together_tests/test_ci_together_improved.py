"""
Improved CI tests for Together AI models using universal test infrastructure.

This demonstrates how Together tests can be simplified by using the shared
universal test infrastructure instead of duplicating code.
"""

import os
import sys
import pytest

# Add parent directory to path for imports  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import create_universal_client, UniversalTestData
from common.test_suites import (
    UniversalChatTestSuite,
    UniversalStreamingTestSuite, 
    UniversalEmbeddingTestSuite,
    UniversalErrorTestSuite
)


class TestTogetherModelsCI:
    """CI tests for Together AI models using universal test infrastructure."""
    
    def setup_class(cls):
        """Setup test class."""
        cls.client = create_universal_client(provider_hint="together")
        cls.together_models = UniversalTestData.get_provider_models()["together"]
        cls.embedding_models = UniversalTestData.get_embedding_models()["together"]
    
    def test_together_chat_models_basic(self):
        """Test basic chat with Together models using universal suite."""
        # Use universal test suite instead of duplicating test logic
        suite = UniversalChatTestSuite(self.together_models, provider_hint="together")
        
        # Test all models
        results = suite.test_all_models()
        
        # Verify all models responded
        assert len(results) == len(self.together_models)
        
        # Verify model names with slashes are preserved
        for model in self.together_models:
            assert model in results
            assert results[model].model == model
            if "/" in model:
                assert "/" in results[model].model, f"Slash not preserved in {model}"
    
    def test_together_streaming(self):
        """Test streaming with Together models using universal suite."""
        suite = UniversalStreamingTestSuite(self.together_models, provider_hint="together")
        
        # Test with first model
        chunks_count, content = suite.test_basic_streaming(self.together_models[0])
        
        assert chunks_count > 0, "Should receive streaming chunks"
        assert len(content) > 0, "Should receive content"
    
    def test_together_parameters(self):
        """Test parameters with Together models using universal suite."""
        suite = UniversalChatTestSuite(self.together_models, provider_hint="together")
        
        # Test temperature parameter
        low_temp, high_temp = suite.test_temperature_parameter(self.together_models[0])
        assert low_temp.choices[0].message.content is not None
        assert high_temp.choices[0].message.content is not None
        
        # Test max_tokens parameter
        response = suite.test_max_tokens_parameter(self.together_models[0])
        assert response.choices[0].message.content is not None
    
    def test_together_embeddings(self):
        """Test Together embeddings using universal suite."""
        if not self.embedding_models:
            pytest.skip("No Together embedding models configured")
        
        suite = UniversalEmbeddingTestSuite(self.embedding_models, provider_hint="together")
        
        # Test single embedding
        response = suite.test_single_embedding(self.embedding_models[0])
        assert len(response.data) == 1
        assert response.model == self.embedding_models[0]
        
        # Test batch embeddings
        response = suite.test_batch_embeddings(self.embedding_models[0])
        assert len(response.data) > 1
    
    def test_together_error_handling(self):
        """Test error handling with Together models using universal suite."""
        suite = UniversalErrorTestSuite(self.together_models, provider_hint="together")
        
        # Test invalid model
        error = suite.test_invalid_model()
        assert error is not None
        
        # Test empty messages
        error = suite.test_empty_messages()
        assert error is not None
    
    def test_together_model_name_format(self):
        """Test that Together model names with special characters work correctly."""
        # Together models have slashes and hyphens in names
        special_models = [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "mistralai/Mixtral-8x7B-Instruct-v0.1", 
            "deepseek-ai/deepseek-v2.5"
        ]
        
        for model in special_models:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Test model name parsing"}],
                max_tokens=20
            )
            
            # Model name should be preserved exactly
            assert response.model == model
            assert "/" in response.model, f"Slash not preserved in {model}"
            assert "-" in response.model, f"Hyphen not preserved in {model}"
    
    def test_together_multi_turn_conversation(self):
        """Test multi-turn conversations with Together models."""
        suite = UniversalChatTestSuite(self.together_models, provider_hint="together")
        
        response = suite.test_multi_turn_conversation(self.together_models[0])
        assert response.choices[0].message.content is not None
        assert response.choices[0].message.role == "assistant"
    
    def test_together_system_prompts(self):
        """Test system prompts with Together models.""" 
        suite = UniversalChatTestSuite(self.together_models, provider_hint="together")
        
        response = suite.test_system_prompt(self.together_models[0])
        assert response.choices[0].message.content is not None


class TestTogetherSpecificFeatures:
    """Test Together-specific features through OpenAI SDK."""
    
    def setup_class(cls):
        """Setup test class."""
        cls.client = create_universal_client(provider_hint="together")
    
    def test_together_model_families(self):
        """Test different model families available on Together."""
        model_families = {
            "Llama": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
            "Qwen": "Qwen/Qwen2.5-72B-Instruct-Turbo", 
            "Mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "DeepSeek": "deepseek-ai/deepseek-v2.5"
        }
        
        for family, model in model_families.items():
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"Hello from {family}"}],
                max_tokens=30
            )
            
            assert response.choices[0].message.content is not None
            assert response.model == model
            print(f"✅ {family} ({model}): Working")
    
    def test_together_large_context(self):
        """Test handling of larger context with Together models."""
        # Test with a moderately long input
        long_input = "Please summarize this text: " + "This is a test sentence. " * 100
        
        response = self.client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct-Turbo",
            messages=[{"role": "user", "content": long_input}],
            max_tokens=100
        )
        
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])