use std::{collections::HashMap, sync::Arc};
use tokio::sync::RwLock;
use redis::{aio::ConnectionManager, AsyncCommands, streams::{StreamReadOptions, StreamReadReply}};
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

use crate::{
    error::{Error, ErrorDetails},
    model::{ModelConfig, UninitializedModelConfig},
    model_table::{BaseModelTable, ShorthandModelConfig},
    config_parser::ProviderTypesConfig,
};

/// Redis stream configuration for dynamic model updates
#[derive(Debug, Clone, Deserialize)]
pub struct RedisStreamConfig {
    /// Redis connection URL (e.g., "redis://localhost:6379")
    pub connection_url: String,
    /// Stream name to listen for model updates
    pub stream_name: String,
    /// Consumer group name for the stream
    pub consumer_group: String,
    /// Consumer name within the group
    pub consumer_name: String,
    /// How often to poll the stream (in milliseconds)
    #[serde(default = "default_poll_interval")]
    pub poll_interval_ms: u64,
}

fn default_poll_interval() -> u64 {
    1000 // 1 second default
}

/// Message format for model updates from Redis stream
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "action")]
pub enum ModelUpdateMessage {
    /// Add or update a model
    #[serde(rename = "upsert")]
    Upsert {
        model_name: String,
        /// Raw JSON for the model config to avoid serialization issues
        model_config: serde_json::Value,
    },
    /// Remove a model
    #[serde(rename = "remove")]
    Remove { model_name: String },
}

/// A wrapper that provides the BaseModelTable interface but with dynamic updates
/// This maintains API compatibility while adding Redis stream support
pub struct DynamicModelTable {
    /// The models, stored as Arc for efficient cloning
    models: Arc<RwLock<HashMap<Arc<str>, Arc<ModelConfig>>>>,
    /// Redis configuration (optional - if None, behaves like static table)
    redis_config: Option<RedisStreamConfig>,
    /// Provider types configuration for loading models
    provider_types: Arc<ProviderTypesConfig>,
}

impl DynamicModelTable {
    /// Create a new dynamic model table from an existing BaseModelTable
    pub fn from_base_table(
        base_table: BaseModelTable<ModelConfig>,
        redis_config: Option<RedisStreamConfig>,
        provider_types: Arc<ProviderTypesConfig>,
    ) -> Self {
        // Convert the BaseModelTable to our Arc-based storage
        let models = HashMap::new();
        // Note: We can't clone ModelConfig, so we start with an empty dynamic table
        // Static models will still be available through the base_table lookup
        
        Self {
            models: Arc::new(RwLock::new(models)),
            redis_config,
            provider_types,
        }
    }

    /// Start the Redis stream listener (spawns a background task)
    pub fn start_redis_listener(&self) -> Result<(), Error> {
        let Some(redis_config) = &self.redis_config else {
            info!("No Redis configuration provided, running with static model table");
            return Ok(());
        };

        let models = Arc::clone(&self.models);
        let config = redis_config.clone();
        let provider_types = Arc::clone(&self.provider_types);

        // Spawn the Redis listener task
        tokio::spawn(async move {
            if let Err(e) = redis_stream_listener(models, config, provider_types).await {
                error!("Redis stream listener error: {}", e);
            }
        });

        info!("Started Redis stream listener for dynamic model updates");
        Ok(())
    }

    /// Get the inner models table for use with existing code
    /// This provides a BaseModelTable-compatible interface
    pub async fn as_base_table(&self) -> BaseModelTableView {
        BaseModelTableView {
            models: Arc::clone(&self.models),
        }
    }
}

/// A view into the dynamic model table that implements the BaseModelTable interface
pub struct BaseModelTableView {
    models: Arc<RwLock<HashMap<Arc<str>, Arc<ModelConfig>>>>,
}

impl BaseModelTableView {
    /// Get a model by name, checking both static and shorthand models
    pub async fn get(&self, key: &str) -> Result<Option<Arc<ModelConfig>>, Error> {
        let models = self.models.read().await;
        
        // First check if it's a static model
        if let Some(model) = models.get(key) {
            return Ok(Some(Arc::clone(model)));
        }
        
        // Then check if it's a shorthand model
        // We delegate to the BaseModelTable's shorthand logic
        if let Some(shorthand) = Self::check_shorthand(key) {
            // Create the shorthand model config
            let config = ModelConfig::from_shorthand(shorthand.provider_type, shorthand.model_name).await?;
            return Ok(Some(Arc::new(config)));
        }
        
        Ok(None)
    }

    /// Check if a key matches a shorthand pattern
    fn check_shorthand(key: &str) -> Option<ShorthandMatch> {
        // This replicates the logic from BaseModelTable
        const SHORTHAND_PREFIXES: &[&str] = &[
            "anthropic::", "openai::", "aws_bedrock::", "azure::", 
            "deepseek::", "fireworks::", "gcp_vertex_anthropic::",
            "gcp_vertex_gemini::", "google_ai_studio_gemini::", 
            "hyperbolic::", "mistral::", "openrouter::", "together::",
            "xai::", "vllm::", "sglang::", "tgi::", "aws_sagemaker::",
        ];
        
        for prefix in SHORTHAND_PREFIXES {
            if let Some(model_name) = key.strip_prefix(prefix) {
                let provider_type = &prefix[..prefix.len() - 2];
                return Some(ShorthandMatch {
                    provider_type,
                    model_name,
                });
            }
        }
        None
    }

    /// Validate a model name
    pub async fn validate(&self, key: &str) -> Result<(), Error> {
        let models = self.models.read().await;
        
        // Check if it's in the table
        if models.contains_key(key) {
            return Ok(());
        }
        
        // Check if it's a valid shorthand
        if Self::check_shorthand(key).is_some() {
            return Ok(());
        }
        
        Err(ErrorDetails::Config {
            message: format!("Model name '{}' not found in model table", key),
        }.into())
    }
}

struct ShorthandMatch<'a> {
    provider_type: &'a str,
    model_name: &'a str,
}

impl ModelConfig {
    /// Create a ModelConfig from shorthand notation
    /// This is a simplified version - you'll need to implement the actual logic
    async fn from_shorthand(provider_type: &str, model_name: &str) -> Result<Self, Error> {
        // This would need to be implemented based on your actual ModelConfig structure
        // For now, returning an error
        Err(ErrorDetails::Config {
            message: format!("Shorthand model creation not yet implemented for {}::{}", provider_type, model_name),
        }.into())
    }
}

/// The main Redis stream listener loop
async fn redis_stream_listener(
    models: Arc<RwLock<HashMap<Arc<str>, Arc<ModelConfig>>>>,
    config: RedisStreamConfig,
    provider_types: Arc<ProviderTypesConfig>,
) -> Result<(), Error> {
    info!("Starting Redis stream listener for model updates");
    
    // Create Redis connection
    let client = redis::Client::open(config.connection_url.as_str())
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
        "Redis stream listener connected to stream '{}' as consumer '{}' in group '{}'",
        config.stream_name, config.consumer_name, config.consumer_group
    );

    // Main listening loop
    loop {
        // Read from the stream
        let opts = StreamReadOptions::default()
            .group(&config.consumer_group, &config.consumer_name)
            .count(10) // Process up to 10 messages at a time
            .block(config.poll_interval_ms as usize);

        let result: Result<StreamReadReply, _> = con
            .xread_options(&[&config.stream_name], &[">"], &opts)
            .await;

        match result {
            Ok(reply) => {
                for stream_key in reply.keys {
                    for stream_id in stream_key.ids {
                        // Process the message
                        if let Err(e) = process_model_update_message(
                            &models,
                            &provider_types,
                            &stream_id.map,
                            &mut con,
                            &config.stream_name,
                            &stream_id.id,
                            &config.consumer_group,
                        ).await {
                            error!("Error processing model update message: {}", e);
                        }
                    }
                }
            }
            Err(e) => {
                error!("Error reading from Redis stream: {}", e);
                // Wait before retrying
                tokio::time::sleep(tokio::time::Duration::from_millis(config.poll_interval_ms)).await;
            }
        }
    }
}

/// Process a single model update message from the Redis stream
async fn process_model_update_message(
    models: &Arc<RwLock<HashMap<Arc<str>, Arc<ModelConfig>>>>,
    provider_types: &Arc<ProviderTypesConfig>,
    data: &HashMap<String, redis::Value>,
    con: &mut ConnectionManager,
    stream_name: &str,
    message_id: &str,
    consumer_group: &str,
) -> Result<(), Error> {
    // Extract the message payload
    let message_data = data.get("data")
        .and_then(|v| match v {
            redis::Value::BulkString(bytes) => Some(bytes),
            _ => None,
        })
        .ok_or_else(|| Error::new(ErrorDetails::Config {
            message: "Invalid message format: missing 'data' field".to_string(),
        }))?;

    // Parse the message
    let message: ModelUpdateMessage = serde_json::from_slice(message_data)
        .map_err(|e| Error::new(ErrorDetails::Config {
            message: format!("Failed to parse model update message: {}", e),
        }))?;

    // Process the message based on action type
    match message {
        ModelUpdateMessage::Upsert { model_name, model_config } => {
            info!("Processing model upsert for: {}", model_name);
            
            // Parse the uninitialized model config from JSON
            let uninitialized: UninitializedModelConfig = serde_json::from_value(model_config)
                .map_err(|e| Error::new(ErrorDetails::Config {
                    message: format!("Failed to parse model config JSON: {}", e),
                }))?;
            
            // Load the model config
            let loaded_config = uninitialized.load(&model_name, provider_types)
                .map_err(|e| Error::new(ErrorDetails::Config {
                    message: format!("Failed to load model config for {}: {}", model_name, e),
                }))?;
            
            // Update the models table
            let mut models_write = models.write().await;
            models_write.insert(Arc::from(model_name.as_str()), Arc::new(loaded_config));
            
            info!("Successfully updated model: {}", model_name);
        }
        ModelUpdateMessage::Remove { model_name } => {
            info!("Processing model removal for: {}", model_name);
            
            // Remove from the models table
            let mut models_write = models.write().await;
            if models_write.remove(model_name.as_str()).is_some() {
                info!("Successfully removed model: {}", model_name);
            } else {
                warn!("Model {} not found for removal", model_name);
            }
        }
    }

    // Acknowledge the message
    let _: Result<(), _> = con
        .xack(stream_name, consumer_group, &[message_id])
        .await;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dynamic_table_without_redis() {
        // Test that the dynamic table works without Redis config
        let base_table = BaseModelTable::default();
        let provider_types = Arc::new(ProviderTypesConfig::default());
        let dynamic_table = DynamicModelTable::from_base_table(
            base_table,
            None,
            provider_types,
        );

        // Should behave like a normal table
        let view = dynamic_table.as_base_table().await;
        assert!(view.get("nonexistent").await.unwrap().is_none());
    }
}