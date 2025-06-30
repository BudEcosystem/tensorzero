use std::collections::HashMap;
use std::time::Duration;

use reqwest::StatusCode;
use serde_json::{json, Value};

use crate::common::{get_gateway_endpoint, get_response_api_key};

#[tokio::test]
async fn test_response_create_basic() {
    let client = reqwest::Client::new();

    let request_body = json!({
        "model": "gpt-4-responses",
        "input": "Hello, world!",
        "instructions": "Be helpful and friendly"
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/responses"))
        .header("Authorization", format!("Bearer {}", get_response_api_key()))
        .json(&request_body)
        .send()
        .await
        .unwrap();

    // Since we don't have a real responses model configured, this should fail
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    
    let error_body: Value = response.json().await.unwrap();
    assert!(error_body["error"]["message"].as_str().unwrap().contains("Model"));
}

#[tokio::test]
async fn test_response_create_with_tools() {
    let client = reqwest::Client::new();

    let tools = vec![
        json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        })
    ];

    let request_body = json!({
        "model": "gpt-4-responses",
        "input": "What's the weather like in New York?",
        "tools": tools,
        "tool_choice": "auto",
        "parallel_tool_calls": true
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/responses"))
        .header("Authorization", format!("Bearer {}", get_response_api_key()))
        .json(&request_body)
        .send()
        .await
        .unwrap();

    // Since we don't have a real responses model configured, this should fail
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_response_create_streaming() {
    let client = reqwest::Client::new();

    let request_body = json!({
        "model": "gpt-4-responses",
        "input": "Tell me a story",
        "stream": true,
        "stream_options": {
            "include_usage": true
        }
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/responses"))
        .header("Authorization", format!("Bearer {}", get_response_api_key()))
        .json(&request_body)
        .send()
        .await
        .unwrap();

    // Since we don't have a real responses model configured, this should fail
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_response_create_multimodal() {
    let client = reqwest::Client::new();

    let request_body = json!({
        "model": "gpt-4o-responses",
        "input": [
            {
                "type": "text",
                "text": "What's in this image?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
                }
            }
        ],
        "modalities": ["text", "image"]
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/responses"))
        .header("Authorization", format!("Bearer {}", get_response_api_key()))
        .json(&request_body)
        .send()
        .await
        .unwrap();

    // Since we don't have a real responses model configured, this should fail
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_response_retrieve() {
    let client = reqwest::Client::new();

    let response = client
        .get(get_gateway_endpoint("/openai/v1/responses/resp_123"))
        .header("Authorization", format!("Bearer {}", get_response_api_key()))
        .send()
        .await
        .unwrap();

    // This should return an error since retrieval is not supported
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    
    let error_body: Value = response.json().await.unwrap();
    assert!(error_body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Response retrieval is not supported"));
}

#[tokio::test]
async fn test_response_delete() {
    let client = reqwest::Client::new();

    let response = client
        .delete(get_gateway_endpoint("/openai/v1/responses/resp_123"))
        .header("Authorization", format!("Bearer {}", get_response_api_key()))
        .send()
        .await
        .unwrap();

    // This should return an error since deletion is not supported
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    
    let error_body: Value = response.json().await.unwrap();
    assert!(error_body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Response deletion is not supported"));
}

#[tokio::test]
async fn test_response_cancel() {
    let client = reqwest::Client::new();

    let response = client
        .post(get_gateway_endpoint("/openai/v1/responses/resp_123/cancel"))
        .header("Authorization", format!("Bearer {}", get_response_api_key()))
        .send()
        .await
        .unwrap();

    // This should return an error since cancellation is not supported
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    
    let error_body: Value = response.json().await.unwrap();
    assert!(error_body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Response cancellation is not supported"));
}

#[tokio::test]
async fn test_response_input_items() {
    let client = reqwest::Client::new();

    let response = client
        .get(get_gateway_endpoint("/openai/v1/responses/resp_123/input_items"))
        .header("Authorization", format!("Bearer {}", get_response_api_key()))
        .send()
        .await
        .unwrap();

    // This should return an error since input items listing is not supported
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    
    let error_body: Value = response.json().await.unwrap();
    assert!(error_body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Input items listing is not supported"));
}

#[tokio::test]
async fn test_response_create_with_metadata() {
    let client = reqwest::Client::new();

    let mut metadata = HashMap::new();
    metadata.insert("user_id", json!("user_123"));
    metadata.insert("session_id", json!("session_456"));
    metadata.insert("custom_field", json!({"nested": "value"}));

    let request_body = json!({
        "model": "gpt-4-responses",
        "input": "Hello",
        "metadata": metadata,
        "user": "user_123"
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/responses"))
        .header("Authorization", format!("Bearer {}", get_response_api_key()))
        .json(&request_body)
        .send()
        .await
        .unwrap();

    // Since we don't have a real responses model configured, this should fail
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_response_create_with_previous_response() {
    let client = reqwest::Client::new();

    let request_body = json!({
        "model": "gpt-4-responses",
        "input": "Continue the conversation",
        "previous_response_id": "resp_previous_123"
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/responses"))
        .header("Authorization", format!("Bearer {}", get_response_api_key()))
        .json(&request_body)
        .send()
        .await
        .unwrap();

    // Since we don't have a real responses model configured, this should fail
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_response_create_with_reasoning() {
    let client = reqwest::Client::new();

    let request_body = json!({
        "model": "o1-responses",
        "input": "Solve this complex problem step by step",
        "reasoning": {
            "reasoning_effort": "high"
        },
        "temperature": 0.7,
        "max_output_tokens": 2000
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/responses"))
        .header("Authorization", format!("Bearer {}", get_response_api_key()))
        .json(&request_body)
        .send()
        .await
        .unwrap();

    // Since we don't have a real responses model configured, this should fail
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_response_create_with_unknown_fields() {
    let client = reqwest::Client::new();

    let request_body = json!({
        "model": "gpt-4-responses",
        "input": "Test input",
        "custom_field": "custom_value",
        "another_unknown": 123
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/responses"))
        .header("Authorization", format!("Bearer {}", get_response_api_key()))
        .json(&request_body)
        .send()
        .await
        .unwrap();

    // The gateway should accept unknown fields (with warnings) but still fail due to missing model
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}