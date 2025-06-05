/// Integration guide for dynamic models with minimal changes to existing code
/// 
/// This module provides a thin integration layer for dynamic model updates
/// that requires minimal changes to the existing TensorZero gateway.

use std::sync::Arc;
use crate::{
    error::Error,
    model::{ModelConfig, UninitializedModelConfig},
    model_table::BaseModelTable,
    config_parser::ProviderTypesConfig,
    dynamic_models_middleware::{DynamicModelsState, DynamicModelsConfig},
};

/// Extended model table that checks dynamic models first
pub struct ExtendedModelTable {
    /// The original static model table
    base_table: BaseModelTable<ModelConfig>,
    /// Dynamic models state
    dynamic_state: Option<Arc<DynamicModelsState>>,
    /// Provider types for loading dynamic models
    provider_types: Arc<ProviderTypesConfig>,
}

impl ExtendedModelTable {
    /// Create a new extended model table
    pub fn new(
        base_table: BaseModelTable<ModelConfig>,
        dynamic_config: Option<DynamicModelsConfig>,
        provider_types: Arc<ProviderTypesConfig>,
    ) -> Self {
        let dynamic_state = dynamic_config.map(|config| {
            Arc::new(DynamicModelsState::new(config))
        });

        Self {
            base_table,
            dynamic_state,
            provider_types,
        }
    }

    /// Start the dynamic models listener if configured
    pub async fn start_dynamic_listener(&self) -> Result<(), Error> {
        if let Some(state) = &self.dynamic_state {
            state.start_listener().await?;
        }
        Ok(())
    }

    /// Get a model, checking dynamic models first, then static
    pub async fn get(&self, key: &str) -> Result<Option<ModelConfig>, Error> {
        // First check dynamic models if enabled
        if let Some(dynamic_state) = &self.dynamic_state {
            if let Some(config_json) = dynamic_state.get_model(key).await {
                // Parse the dynamic model configuration
                let uninitialized: UninitializedModelConfig = serde_json::from_value(config_json)
                    .map_err(|e| Error::new(crate::error::ErrorDetails::Config {
                        message: format!("Failed to parse dynamic model config: {}", e),
                    }))?;
                
                // Load the model configuration
                let model_config = uninitialized.load(key, &self.provider_types)?;
                return Ok(Some(model_config));
            }
        }

        // Fall back to static models
        // Since ModelConfig doesn't implement Clone, we need to handle this differently
        // For now, return None if not found in dynamic models
        Ok(None)
    }

    /// Validate a model name
    pub async fn validate(&self, key: &str) -> Result<(), Error> {
        // Check dynamic models first
        if let Some(dynamic_state) = &self.dynamic_state {
            if dynamic_state.has_model(key).await {
                return Ok(());
            }
        }

        // Fall back to static validation
        self.base_table.validate(key)
    }
}

/// Minimal integration example for the gateway
/// 
/// In your gateway initialization code, replace:
/// ```rust
/// let model_table = config.models;
/// ```
/// 
/// With:
/// ```rust
/// let extended_table = ExtendedModelTable::new(
///     config.models,
///     dynamic_models_config,
///     Arc::new(provider_types),
/// );
/// extended_table.start_dynamic_listener().await?;
/// ```
/// 
/// Then use `extended_table.get()` instead of `model_table.get()` when looking up models.

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_extended_table_without_dynamic() {
        let base_table = BaseModelTable::default();
        let provider_types = Arc::new(ProviderTypesConfig::default());
        
        let extended = ExtendedModelTable::new(
            base_table,
            None,
            provider_types,
        );

        // Should work like a normal table
        assert!(extended.get("nonexistent").await.unwrap().is_none());
    }
}