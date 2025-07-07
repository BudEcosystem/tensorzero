"""
Improved Together embeddings tests using universal test infrastructure.

This demonstrates how embedding tests can be simplified by using the shared
universal test infrastructure instead of duplicating validation logic.
"""

import os
import sys
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import create_universal_client, UniversalTestData
from common.test_suites import UniversalEmbeddingTestSuite


class TestTogetherEmbeddingsImproved:
    """Improved Together embeddings tests using universal infrastructure."""
    
    def setup_class(cls):
        """Setup test class."""
        cls.client = create_universal_client(provider_hint="together")
        cls.embedding_models = UniversalTestData.get_embedding_models()["together"]
    
    def test_together_single_embedding(self):
        """Test single text embedding with Together models."""
        if not self.embedding_models:
            pytest.skip("No Together embedding models configured")
        
        suite = UniversalEmbeddingTestSuite(self.embedding_models, provider_hint="together")
        
        for model in self.embedding_models:
            response = suite.test_single_embedding(model)
            
            # Verify Together-specific aspects
            assert response.model == model
            assert len(response.data) == 1
            
            # BGE base typically has 768 dimensions
            if "bge-base" in model:
                embedding_dims = len(response.data[0].embedding)
                assert embedding_dims > 0, f"BGE model {model} should have > 0 dimensions"
    
    def test_together_batch_embeddings(self):
        """Test batch embeddings with Together models."""
        if not self.embedding_models:
            pytest.skip("No Together embedding models configured")
        
        suite = UniversalEmbeddingTestSuite(self.embedding_models, provider_hint="together")
        
        for model in self.embedding_models:
            response = suite.test_batch_embeddings(model)
            
            # Should have multiple embeddings
            assert len(response.data) > 1
            assert response.model == model
            
            # All embeddings should have same dimensions
            first_dims = len(response.data[0].embedding)
            for i, embedding_data in enumerate(response.data):
                assert len(embedding_data.embedding) == first_dims, f"Inconsistent dimensions in batch for {model}"
    
    def test_together_embedding_special_characters(self):
        """Test embeddings with special characters using universal suite."""
        if not self.embedding_models:
            pytest.skip("No Together embedding models configured")
        
        suite = UniversalEmbeddingTestSuite(self.embedding_models, provider_hint="together")
        
        # Use the universal test for special characters
        response = suite.test_special_characters(self.embedding_models[0])
        
        # Verify all embeddings were created
        assert len(response.data) == 3  # 3 special text inputs
        for embedding_data in response.data:
            assert len(embedding_data.embedding) > 0
    
    def test_together_embedding_comparison(self):
        """Compare embeddings between Together models."""
        if len(self.embedding_models) < 2:
            pytest.skip("Need at least 2 Together embedding models for comparison")
        
        test_text = "Machine learning is transforming technology"
        
        model_embeddings = {}
        
        for model in self.embedding_models:
            response = self.client.embeddings.create(
                model=model,
                input=test_text
            )
            
            model_embeddings[model] = response.data[0].embedding
        
        # Different models should produce different embeddings
        models = list(model_embeddings.keys())
        if len(models) >= 2:
            embedding1 = model_embeddings[models[0]]
            embedding2 = model_embeddings[models[1]]
            
            # Should be different (not identical)
            assert embedding1 != embedding2, f"Embeddings from {models[0]} and {models[1]} are identical"
            
            # Should have reasonable dimensions
            assert len(embedding1) > 100, f"{models[0]} embedding seems too small"
            assert len(embedding2) > 100, f"{models[1]} embedding seems too small"
    
    def test_together_embedding_consistency(self):
        """Test that same input produces consistent embeddings."""
        if not self.embedding_models:
            pytest.skip("No Together embedding models configured")
        
        model = self.embedding_models[0]
        test_text = "Consistency test for embeddings"
        
        # Generate same embedding twice
        response1 = self.client.embeddings.create(
            model=model,
            input=test_text
        )
        
        response2 = self.client.embeddings.create(
            model=model,
            input=test_text
        )
        
        embedding1 = response1.data[0].embedding
        embedding2 = response2.data[0].embedding
        
        # Should be identical (deterministic)
        assert embedding1 == embedding2, f"Embeddings not consistent for {model}"
    
    def test_together_vs_openai_embedding_format(self):
        """Compare Together embeddings format with OpenAI format expectations."""
        if not self.embedding_models:
            pytest.skip("No Together embedding models configured")
        
        model = self.embedding_models[0]
        test_text = "Format compatibility test"
        
        response = self.client.embeddings.create(
            model=model,
            input=test_text
        )
        
        # Should match OpenAI embedding response format exactly
        assert response.object == "list"
        assert len(response.data) == 1
        
        embedding_data = response.data[0]
        assert embedding_data.object == "embedding"
        assert embedding_data.index == 0
        assert len(embedding_data.embedding) > 0
        assert all(isinstance(x, float) for x in embedding_data.embedding)
        
        # Should have usage information
        assert response.usage is not None
        assert response.usage.total_tokens > 0
        
        # Model name should be preserved
        assert response.model == model


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])