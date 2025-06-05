use std::{collections::HashMap, sync::Arc};
use tokio::sync::RwLock;
use redis::{aio::ConnectionManager, AsyncCommands, streams::{StreamReadOptions, StreamReadReply}};
use serde::{Deserialize, Serialize};
use tracing::{error, info};
use axum::{
    middleware::Next,
    http::{Request, StatusCode},
    response::Response,
    body::Body,
};

use crate::{
    error::{Error, ErrorDetails},
    model::ModelConfig,
    config_parser::ProviderTypesConfig,
};

/// Redis stream configuration for dynamic model updates
#[derive(Debug, Clone, Deserialize)]
pub struct DynamicModelsConfig {
    /// Redis connection URL (e.g., "redis://localhost:6379")
    pub redis_url: String,
    /// Stream name to listen for model updates
    pub stream_name: String,
    /// Consumer group name for the stream
    pub consumer_group: String,
    /// Consumer name within the group
    pub consumer_name: String,
    /// How often to poll the stream (in milliseconds)
    #[serde(default = "default_poll_interval")]
    pub poll_interval_ms: u64,
    /// Whether to enable dynamic models
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

fn default_poll_interval() -> u64 {
    1000 // 1 second default
}

fn default_enabled() -> bool {
    false
}

/// Message format for model updates from Redis stream
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "action")]
pub enum ModelUpdateMessage {
    /// Add or update a model configuration
    #[serde(rename = "upsert")]
    Upsert {
        model_name: String,
        #[serde(flatten)]
        config: serde_json::Value, // Raw JSON for the model config
    },
    /// Remove a model
    #[serde(rename = "remove")]
    Remove { 
        model_name: String 
    },
}

/// Global state for dynamic models
pub struct DynamicModelsState {
    /// Additional models loaded from Redis (model_name -> raw config JSON)
    models: Arc<RwLock<HashMap<String, serde_json::Value>>>,
    /// Configuration
    config: DynamicModelsConfig,
}

impl DynamicModelsState {
    pub fn new(config: DynamicModelsConfig) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }
    
    pub async fn new_with_global_registry(config: DynamicModelsConfig) -> Self {
        // Try to get the global models storage
        let models = if let Some(global_models) = crate::dynamic_models_registry::DynamicModelsRegistry::get_models_storage().await {
            global_models
        } else {
            Arc::new(RwLock::new(HashMap::new()))
        };
        
        Self {
            models,
            config,
        }
    }

    /// Start the Redis stream listener
    pub async fn start_listener(&self) -> Result<(), Error> {
        if !self.config.enabled {
            info!("Dynamic models disabled");
            return Ok(());
        }

        let models = Arc::clone(&self.models);
        let config = self.config.clone();

        tokio::spawn(async move {
            if let Err(e) = redis_stream_listener(models, config).await {
                error!("Redis stream listener error: {}", e);
            }
        });

        info!("Started Redis stream listener for dynamic model updates");
        Ok(())
    }

    /// Get a dynamic model configuration by name
    pub async fn get_model(&self, model_name: &str) -> Option<serde_json::Value> {
        let models = self.models.read().await;
        models.get(model_name).cloned()
    }

    /// Check if a model exists in the dynamic table
    pub async fn has_model(&self, model_name: &str) -> bool {
        let models = self.models.read().await;
        let exists = models.contains_key(model_name);
        tracing::debug!("Checking for dynamic model '{}': exists={}, total_models={}", model_name, exists, models.len());
        exists
    }
}

/// Request body structure for model inference
#[derive(Debug, Deserialize)]
struct InferenceRequestBody {
    model: String,
    messages: Vec<serde_json::Value>,
    #[serde(flatten)]
    extra: HashMap<String, serde_json::Value>,
}

/// Extension to pass validated model information through the request
#[derive(Clone)]
pub struct ValidatedModel {
    pub model_name: String,
    pub is_dynamic: bool,
}

/// Axum middleware for dynamic model validation
/// This middleware intercepts requests and validates model names against dynamic models
pub async fn dynamic_models_middleware(
    state: Arc<DynamicModelsState>,
    request: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    // Check if this is an OpenAI-compatible endpoint
    let is_openai_endpoint = request.uri().path() == "/openai/v1/chat/completions"
        || request.uri().path() == "/v1/chat/completions";
    
    if !is_openai_endpoint {
        // Not an OpenAI endpoint, pass through
        return Ok(next.run(request).await);
    }
    
    tracing::debug!("Dynamic models middleware: Processing OpenAI-compatible request");
    
    // For OpenAI-compatible endpoints, we need to check and modify the model name
    // Extract the body to check the model
    let (parts, body) = request.into_parts();
    
    // Read the body into bytes
    let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        Err(_) => {
            return Err(StatusCode::BAD_REQUEST);
        }
    };
    
    // Try to parse the request to check the model
    let mut request_json: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(json) => json,
        Err(_) => {
            // If we can't parse it, just pass it through
            let request = Request::from_parts(parts, Body::from(body_bytes));
            return Ok(next.run(request).await);
        }
    };
    
    // Check if the model field exists and if it's a dynamic model
    let should_rewrite = if let Some(model_value) = request_json.get("model") {
        if let Some(model_name) = model_value.as_str() {
            tracing::debug!("Checking model '{}' in dynamic models", model_name);
            let has_model = state.has_model(model_name).await;
            tracing::debug!("Model '{}' exists in dynamic models: {}", model_name, has_model);
            has_model
        } else {
            tracing::debug!("Model field is not a string");
            false
        }
    } else {
        tracing::debug!("No model field found in request");
        false
    };
    
    if should_rewrite {
        // Get the model name again and rewrite it
        if let Some(model_name) = request_json.get("model").and_then(|v| v.as_str()) {
            let prefixed_model = format!("tensorzero::model_name::{}", model_name);
            info!("Rewriting dynamic model '{}' to '{}'", model_name, prefixed_model);
            request_json["model"] = serde_json::Value::String(prefixed_model);
            
            // Serialize back to bytes
            let modified_body = serde_json::to_vec(&request_json)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            
            // Create new request with modified body
            let request = Request::from_parts(parts, Body::from(modified_body));
            return Ok(next.run(request).await);
        }
    }
    
    // Not a dynamic model or couldn't process, pass through unchanged
    let request = Request::from_parts(parts, Body::from(body_bytes));
    Ok(next.run(request).await)
}

/// Redis stream listener implementation
async fn redis_stream_listener(
    mut models: Arc<RwLock<HashMap<String, serde_json::Value>>>,
    config: DynamicModelsConfig,
) -> Result<(), Error> {
    // Get the global models storage if available
    if let Some(global_models) = crate::dynamic_models_registry::DynamicModelsRegistry::get_models_storage().await {
        models = global_models;
    }
    
    info!("Connecting to Redis at {}", config.redis_url);
    
    let client = redis::Client::open(config.redis_url.as_str())
        .map_err(|e| Error::new(ErrorDetails::Config {
            message: format!("Failed to create Redis client: {}", e),
        }))?;
    
    let mut con = ConnectionManager::new(client).await
        .map_err(|e| Error::new(ErrorDetails::Config {
            message: format!("Failed to connect to Redis: {}", e),
        }))?;

    // Create consumer group if it doesn't exist
    let _: Result<(), _> = con
        .xgroup_create_mkstream(&config.stream_name, &config.consumer_group, "$")
        .await;

    info!(
        "Connected to Redis stream '{}' as consumer '{}' in group '{}'",
        config.stream_name, config.consumer_name, config.consumer_group
    );

    loop {
        let opts = StreamReadOptions::default()
            .group(&config.consumer_group, &config.consumer_name)
            .count(10)
            .block(config.poll_interval_ms as usize);

        let result: Result<StreamReadReply, _> = con
            .xread_options(&[&config.stream_name], &[">"], &opts)
            .await;

        match result {
            Ok(reply) => {
                let message_count = reply.keys.iter().map(|k| k.ids.len()).sum::<usize>();
                if message_count > 0 {
                    tracing::debug!("Received {} messages from Redis stream", message_count);
                }
                for stream_key in reply.keys {
                    for stream_id in stream_key.ids {
                        if let Err(e) = process_message(
                            &models,
                            &stream_id.map,
                            &mut con,
                            &config.stream_name,
                            &stream_id.id,
                            &config.consumer_group,
                        ).await {
                            error!("Error processing message: {}", e);
                        }
                    }
                }
            }
            Err(e) => {
                error!("Error reading from Redis stream: {}", e);
                tokio::time::sleep(tokio::time::Duration::from_millis(config.poll_interval_ms)).await;
            }
        }
    }
}

/// Process a single message from the Redis stream
async fn process_message(
    models: &Arc<RwLock<HashMap<String, serde_json::Value>>>,
    data: &HashMap<String, redis::Value>,
    con: &mut ConnectionManager,
    stream_name: &str,
    message_id: &str,
    consumer_group: &str,
) -> Result<(), Error> {
    let message_data = data.get("data")
        .and_then(|v| match v {
            redis::Value::BulkString(bytes) => Some(bytes),
            _ => None,
        })
        .ok_or_else(|| Error::new(ErrorDetails::Config {
            message: "Invalid message format: missing 'data' field".to_string(),
        }))?;

    let message: ModelUpdateMessage = serde_json::from_slice(message_data)
        .map_err(|e| Error::new(ErrorDetails::Config {
            message: format!("Failed to parse message: {}", e),
        }))?;

    match message {
        ModelUpdateMessage::Upsert { model_name, config } => {
            info!("Upserting model: {}", model_name);
            let mut models_write = models.write().await;
            models_write.insert(model_name, config);
        }
        ModelUpdateMessage::Remove { model_name } => {
            info!("Removing model: {}", model_name);
            let mut models_write = models.write().await;
            models_write.remove(&model_name);
        }
    }

    // Acknowledge the message
    let _: Result<(), _> = con.xack(stream_name, consumer_group, &[message_id]).await;
    Ok(())
}

/// Extension trait to add dynamic model lookup to existing model tables
#[async_trait::async_trait]
pub trait DynamicModelLookup {
    /// Try to get a model from dynamic sources first, then fall back to static
    async fn get_with_dynamic(
        &self,
        model_name: &str,
        dynamic_state: &DynamicModelsState,
        provider_types: &ProviderTypesConfig,
    ) -> Result<Option<ModelConfig>, Error>;
}

// Example of how to publish model updates to Redis
pub async fn publish_model_update(
    redis_url: &str,
    stream_name: &str,
    message: ModelUpdateMessage,
) -> Result<(), Error> {
    let client = redis::Client::open(redis_url)
        .map_err(|e| Error::new(ErrorDetails::Config {
            message: format!("Failed to create Redis client: {}", e),
        }))?;
    
    let mut con = client.get_multiplexed_async_connection().await
        .map_err(|e| Error::new(ErrorDetails::Config {
            message: format!("Failed to connect to Redis: {}", e),
        }))?;

    let data = serde_json::to_string(&message)
        .map_err(|e| Error::new(ErrorDetails::Config {
            message: format!("Failed to serialize message: {}", e),
        }))?;

    let _: String = con.xadd(stream_name, "*", &[("data", data)])
        .await
        .map_err(|e| Error::new(ErrorDetails::Config {
            message: format!("Failed to publish to stream: {}", e),
        }))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_serialization() {
        let message = ModelUpdateMessage::Upsert {
            model_name: "test-model".to_string(),
            config: serde_json::json!({
                "routing": ["provider1"],
                "providers": {
                    "provider1": {
                        "kind": "openai",
                        "api_key": "test-key"
                    }
                }
            }),
        };

        let serialized = serde_json::to_string(&message).unwrap();
        let deserialized: ModelUpdateMessage = serde_json::from_str(&serialized).unwrap();
        
        match deserialized {
            ModelUpdateMessage::Upsert { model_name, .. } => {
                assert_eq!(model_name, "test-model");
            }
            _ => panic!("Wrong message type"),
        }
    }
}