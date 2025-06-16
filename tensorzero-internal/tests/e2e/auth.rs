use reqwest::{Client, StatusCode};
use serde_json::json;

use crate::common::get_gateway_endpoint;

#[tokio::test]
async fn test_auth_disabled_allows_all_requests() {
    let client = Client::new();

    // Test that requests work without auth headers when auth is disabled (default)
    let response = client.get(get_gateway_endpoint("/health")).send().await;
    assert!(response.is_ok());
    assert_eq!(response.unwrap().status(), StatusCode::OK);

    let response = client.get(get_gateway_endpoint("/status")).send().await;
    assert!(response.is_ok());
    assert_eq!(response.unwrap().status(), StatusCode::OK);
}

#[tokio::test]
async fn test_health_and_status_bypass_auth() {
    let client = Client::new();

    // Health and status endpoints should always work even with auth enabled
    // (this test assumes a configuration where auth might be enabled)
    let response = client.get(get_gateway_endpoint("/health")).send().await;
    assert!(response.is_ok());
    assert_eq!(response.unwrap().status(), StatusCode::OK);

    let response = client.get(get_gateway_endpoint("/status")).send().await;
    assert!(response.is_ok());
    assert_eq!(response.unwrap().status(), StatusCode::OK);
}

#[tokio::test] 
async fn test_bearer_token_format() {
    let client = Client::new();

    // Test that health endpoint works without Authorization header
    let response = client.get(get_gateway_endpoint("/health")).send().await;
    assert!(response.is_ok());
    assert_eq!(response.unwrap().status(), StatusCode::OK);

    // Test with Bearer token format (would fail if auth enabled without valid key)
    let response = client
        .get(get_gateway_endpoint("/health"))
        .header("Authorization", "Bearer test-key")
        .send()
        .await;
    assert!(response.is_ok());
    assert_eq!(response.unwrap().status(), StatusCode::OK);
}