/// Simplified dynamic model validation approach
/// This module provides model validation without complex middleware body handling

use std::sync::Arc;
use crate::{
    error::{Error, ErrorDetails},
    model::ModelConfig,
    model_table::BaseModelTable,
    config_parser::ProviderTypesConfig,
    dynamic_models_middleware::DynamicModelsState,
};

/// A simple wrapper that validates models from both dynamic and static sources
pub struct ModelValidator {
    base_table: BaseModelTable<ModelConfig>,
    dynamic_state: Option<Arc<DynamicModelsState>>,
}

impl ModelValidator {
    pub fn new(
        base_table: BaseModelTable<ModelConfig>,
        dynamic_state: Option<Arc<DynamicModelsState>>,
    ) -> Self {
        Self {
            base_table,
            dynamic_state,
        }
    }

    /// Check if a model exists (in either dynamic or static tables)
    pub async fn validate_model(&self, model_name: &str) -> Result<bool, Error> {
        // First check dynamic models if enabled
        if let Some(state) = &self.dynamic_state {
            if state.has_model(model_name).await {
                return Ok(true);
            }
        }

        // Then check static models
        match self.base_table.validate(model_name) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Get model configuration with dynamic lookup
    pub async fn get_model(
        &self,
        model_name: &str,
        provider_types: &ProviderTypesConfig,
    ) -> Result<Option<ModelConfig>, Error> {
        // First check dynamic models
        if let Some(state) = &self.dynamic_state {
            if let Some(config_json) = state.get_model(model_name).await {
                // Parse and load the dynamic model
                let uninitialized: crate::model::UninitializedModelConfig = 
                    serde_json::from_value(config_json)
                        .map_err(|e| Error::new(ErrorDetails::Config {
                            message: format!("Failed to parse dynamic model config: {}", e),
                        }))?;
                
                let model_config = uninitialized.load(model_name, provider_types)?;
                return Ok(Some(model_config));
            }
        }

        // Fall back to static models
        // Since we can't clone ModelConfig and BaseModelTable returns CowNoClone,
        // we'll need to work within the existing constraints
        // For now, return None - in practice, you'd need to refactor the model lookup
        Ok(None)
    }
}

/// Helper function to validate model in request handlers
/// Call this from your inference handlers instead of using middleware
pub async fn validate_request_model(
    model_name: &str,
    validator: &ModelValidator,
) -> Result<(), Error> {
    let exists = validator.validate_model(model_name).await?;
    
    if !exists {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!("Model '{}' not found", model_name),
        }));
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_validator_without_dynamic() {
        let base_table = BaseModelTable::default();
        let validator = ModelValidator::new(base_table, None);
        
        // Non-existent model should return false
        assert!(!validator.validate_model("nonexistent").await.unwrap());
    }
}