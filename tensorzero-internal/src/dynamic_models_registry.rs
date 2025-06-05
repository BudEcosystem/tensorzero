use std::sync::Arc;
use tokio::sync::RwLock;
use lazy_static::lazy_static;
use serde_json::Value;

use crate::{
    error::{Error, ErrorDetails},
    model::{ModelConfig, UninitializedModelConfig},
    config_parser::ProviderTypesConfig,
};

lazy_static! {
    /// Global registry for dynamic models
    static ref DYNAMIC_MODELS_REGISTRY: Arc<RwLock<Option<DynamicModelsRegistry>>> = Arc::new(RwLock::new(None));
}

pub struct DynamicModelsRegistry {
    /// Raw model configurations from Redis
    models: Arc<RwLock<std::collections::HashMap<String, Value>>>,
    /// Provider types for loading models
    provider_types: Arc<ProviderTypesConfig>,
}

impl DynamicModelsRegistry {
    /// Initialize the global registry
    pub async fn init(provider_types: Arc<ProviderTypesConfig>) -> Result<(), Error> {
        let registry = DynamicModelsRegistry {
            models: Arc::new(RwLock::new(std::collections::HashMap::new())),
            provider_types,
        };
        
        let mut global = DYNAMIC_MODELS_REGISTRY.write().await;
        *global = Some(registry);
        Ok(())
    }
    
    /// Get the models storage for the middleware to update
    pub async fn get_models_storage() -> Option<Arc<RwLock<std::collections::HashMap<String, Value>>>> {
        let registry = DYNAMIC_MODELS_REGISTRY.read().await;
        registry.as_ref().map(|r| r.models.clone())
    }
    
    /// Look up a dynamic model by name
    pub async fn get_model(model_name: &str) -> Result<Option<ModelConfig>, Error> {
        let registry = DYNAMIC_MODELS_REGISTRY.read().await;
        let Some(registry) = registry.as_ref() else {
            return Ok(None);
        };
        
        let models = registry.models.read().await;
        let Some(config_json) = models.get(model_name) else {
            return Ok(None);
        };
        
        // Parse the dynamic model configuration
        let uninitialized: UninitializedModelConfig = serde_json::from_value(config_json.clone())
            .map_err(|e| Error::new(ErrorDetails::Config {
                message: format!("Failed to parse dynamic model config: {}", e),
            }))?;
        
        // Load the model configuration
        let model_config = uninitialized.load(model_name, &registry.provider_types)?;
        Ok(Some(model_config))
    }
    
    /// Check if a model exists
    pub async fn has_model(model_name: &str) -> bool {
        let registry = DYNAMIC_MODELS_REGISTRY.read().await;
        let Some(registry) = registry.as_ref() else {
            return false;
        };
        
        let models = registry.models.read().await;
        models.contains_key(model_name)
    }
}

/// Check if a model name is a dynamic model reference
pub fn is_dynamic_model_reference(model_name: &str) -> Option<&str> {
    model_name.strip_prefix("tensorzero::model_name::")
}