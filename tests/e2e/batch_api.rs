use serde_json::{json, Value};
use std::path::Path;
use tempfile::TempDir;
use tensorzero::gateway::Config;
use tokio::io::AsyncWriteExt;

#[tokio::test(flavor = "multi_thread")]
async fn test_batch_api_endpoints() {
    // Set up test environment
    let temp_dir = TempDir::new().unwrap();
    let storage_dir = temp_dir.path().join("storage");
    std::fs::create_dir_all(&storage_dir).unwrap();

    // Create a simple config
    let config_content = format!(
        r#"
[gateway]
bind_address = "127.0.0.1:0"
authentication = false
storage_dir = "{}"

[models."gpt-3.5-turbo"]
routing = ["dummy"]
endpoints = ["chat"]

[models."gpt-3.5-turbo".providers.dummy]
type = "dummy"
model_name = "gpt-3.5-turbo"

[models."text-embedding-ada-002"]
routing = ["dummy"]
endpoints = ["embedding"]

[models."text-embedding-ada-002".providers.dummy]
type = "dummy"
model_name = "text-embedding-ada-002"
"#,
        storage_dir.display()
    );

    let config_path = temp_dir.path().join("tensorzero.toml");
    let mut config_file = tokio::fs::File::create(&config_path).await.unwrap();
    config_file.write_all(config_content.as_bytes()).await.unwrap();

    // Start the gateway
    let config = Config::load(Some(&config_path)).await.unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel();
    
    let gateway_handle = tokio::spawn(async move {
        // In real tests, start the gateway properly
        // For now, we'll just test the handlers directly
        drop(config);
        rx.await.ok();
    });

    // Test file upload
    // Note: In a real e2e test, we'd make actual HTTP requests
    // For now, we're just testing the core logic

    // Create test JSONL content
    let test_jsonl = r#"{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}]}}
{"custom_id": "req-2", "method": "POST", "url": "/v1/embeddings", "body": {"model": "text-embedding-ada-002", "input": "Test embedding"}}"#;

    // Write test file
    let test_file_path = temp_dir.path().join("test_batch.jsonl");
    let mut test_file = tokio::fs::File::create(&test_file_path).await.unwrap();
    test_file.write_all(test_jsonl.as_bytes()).await.unwrap();

    // Wait for gateway to be ready
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Create HTTP client
    let client = reqwest::Client::new();
    let base_url = format!("http://{}", addr);

    // Test 1: Upload a file
    let file_content = tokio::fs::read(&test_file_path).await.unwrap();
    let form = reqwest::multipart::Form::new()
        .text("purpose", "batch")
        .part(
            "file",
            reqwest::multipart::Part::bytes(file_content.clone())
                .file_name("test_batch.jsonl")
                .mime_str("application/jsonl")
                .unwrap(),
        );

    let upload_response = client
        .post(format!("{}/v1/files", base_url))
        .multipart(form)
        .send()
        .await
        .unwrap();

    assert_eq!(upload_response.status(), 200);
    let file_object: serde_json::Value = upload_response.json().await.unwrap();
    let file_id = file_object["id"].as_str().unwrap();
    assert!(file_id.starts_with("file-"));

    // Test 2: Retrieve file metadata
    let get_response = client
        .get(format!("{}/v1/files/{}", base_url, file_id))
        .send()
        .await
        .unwrap();

    assert_eq!(get_response.status(), 200);
    let retrieved_file: serde_json::Value = get_response.json().await.unwrap();
    assert_eq!(retrieved_file["id"], file_id);
    assert_eq!(retrieved_file["purpose"], "batch");

    // Test 3: Download file content
    let content_response = client
        .get(format!("{}/v1/files/{}/content", base_url, file_id))
        .send()
        .await
        .unwrap();

    assert_eq!(content_response.status(), 200);
    let downloaded_content = content_response.bytes().await.unwrap();
    assert_eq!(downloaded_content, file_content);

    // Test 4: Create a batch
    let batch_request = serde_json::json!({
        "input_file_id": file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h"
    });

    let batch_response = client
        .post(format!("{}/v1/batches", base_url))
        .json(&batch_request)
        .send()
        .await
        .unwrap();

    assert_eq!(batch_response.status(), 200);
    let batch_object: serde_json::Value = batch_response.json().await.unwrap();
    let batch_id = batch_object["id"].as_str().unwrap();
    assert!(batch_id.starts_with("batch_"));

    // Test 5: Get batch status
    let batch_status_response = client
        .get(format!("{}/v1/batches/{}", base_url, batch_id))
        .send()
        .await
        .unwrap();

    assert_eq!(batch_status_response.status(), 200);
    let batch_status: serde_json::Value = batch_status_response.json().await.unwrap();
    assert_eq!(batch_status["id"], batch_id);

    // Test 6: List batches
    let list_response = client
        .get(format!("{}/v1/batches", base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(list_response.status(), 200);
    let batch_list: serde_json::Value = list_response.json().await.unwrap();
    assert!(batch_list["data"].is_array());

    // Test 7: Cancel batch
    let cancel_response = client
        .post(format!("{}/v1/batches/{}/cancel", base_url, batch_id))
        .send()
        .await
        .unwrap();

    assert_eq!(cancel_response.status(), 200);

    // Test 8: Delete file
    let delete_response = client
        .delete(format!("{}/v1/files/{}", base_url, file_id))
        .send()
        .await
        .unwrap();

    assert_eq!(delete_response.status(), 200);

    // Clean up
    tx.send(()).ok();
    gateway_handle.await.ok();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_batch_authentication() {
    // Test that batch endpoints work with authentication enabled
    let temp_dir = TempDir::new().unwrap();
    let storage_dir = temp_dir.path().join("storage");
    std::fs::create_dir_all(&storage_dir).unwrap();

    // Create a config with authentication enabled
    let config_content = format!(
        r#"
[gateway]
bind_address = "127.0.0.1:0"
authentication = true
storage_dir = "{}"

[models."gpt-3.5-turbo"]
routing = ["dummy"]
endpoints = ["chat"]

[models."gpt-3.5-turbo".providers.dummy]
type = "dummy"
model_name = "gpt-3.5-turbo"
"#,
        storage_dir.display()
    );

    let config_path = temp_dir.path().join("tensorzero.toml");
    let mut config_file = tokio::fs::File::create(&config_path).await.unwrap();
    config_file.write_all(config_content.as_bytes()).await.unwrap();

    // Launch gateway with the config
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let gateway_config_path = config_path.clone();
    
    let gateway_handle = tokio::spawn(async move {
        let config = Config::load_from_file(&gateway_config_path).await.unwrap();
        let (addr, _) = launch_gateway(&config, Some(rx)).await.unwrap();
        addr
    });
    
    let addr = gateway_handle.await.unwrap();
    
    // Wait for gateway to be ready
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    
    // Create HTTP client
    let client = reqwest::Client::new();
    let base_url = format!("http://{}", addr);
    
    // Test that batch endpoints work without authentication when authentication is enabled
    // This verifies that batch operations are account-level, not model-level
    
    // Create a test JSONL file
    let test_jsonl = r#"{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}]}}"#;
    
    let test_file_path = temp_dir.path().join("test_auth.jsonl");
    let mut test_file = tokio::fs::File::create(&test_file_path).await.unwrap();
    test_file.write_all(test_jsonl.as_bytes()).await.unwrap();
    
    let file_content = tokio::fs::read(&test_file_path).await.unwrap();
    
    // Test file upload without authentication - should work because batch APIs bypass auth
    let form = reqwest::multipart::Form::new()
        .text("purpose", "batch")
        .part(
            "file",
            reqwest::multipart::Part::bytes(file_content)
                .file_name("test_auth.jsonl")
                .mime_str("application/jsonl")
                .unwrap(),
        );
    
    let upload_response = client
        .post(format!("{}/v1/files", base_url))
        .multipart(form)
        .send()
        .await
        .unwrap();
    
    // Should succeed even without auth header
    assert_eq!(upload_response.status(), 200);
    let file_object: serde_json::Value = upload_response.json().await.unwrap();
    let file_id = file_object["id"].as_str().unwrap();
    
    // Test batch creation without authentication
    let batch_request = serde_json::json!({
        "input_file_id": file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h"
    });
    
    let batch_response = client
        .post(format!("{}/v1/batches", base_url))
        .json(&batch_request)
        .send()
        .await
        .unwrap();
    
    // Should succeed even without auth header
    assert_eq!(batch_response.status(), 200);
    
    // Clean up
    tx.send(()).ok();
}