// Test to check if empty messages are handled by the dummy provider

#[test]
fn test_dummy_provider_empty_messages() {
    use tensorzero_internal::inference::types::{ModelInferenceRequest, RequestMessage};
    
    // Create a request with empty messages
    let request = ModelInferenceRequest {
        messages: vec![], // Empty messages array
        system: None,
        // ... other fields
    };
    
    // The dummy provider should handle this without error
    // It generates a fixed response regardless of input
}