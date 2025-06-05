/// Example of how to add dynamic models support to the gateway
/// This shows the minimal changes needed in main.rs

use std::sync::Arc;
use tensorzero_internal::{
    config_parser::Config,
    gateway_util::AppStateData,
    dynamic_models_middleware::{DynamicModelsState, DynamicModelsConfig},
    dynamic_models_validation::ModelValidator,
};

/// Extended AppState that includes dynamic models support
pub struct ExtendedAppState {
    /// Original app state
    pub base: AppStateData,
    /// Dynamic models state (optional)
    pub dynamic_models: Option<Arc<DynamicModelsState>>,
    /// Model validator that checks both dynamic and static models
    pub model_validator: Arc<ModelValidator>,
}

/// Initialize app state with dynamic models support if enabled
pub async fn initialize_app_state_with_dynamic_models(
    config: Arc<Config<'static>>,
) -> Result<ExtendedAppState, Box<dyn std::error::Error>> {
    // Initialize base app state
    let base = AppStateData::new(config.clone()).await?;
    
    // Check if dynamic models are enabled via environment
    let dynamic_models_enabled = std::env::var("TENSORZERO_DYNAMIC_MODELS_ENABLED")
        .unwrap_or_else(|_| "false".to_string())
        .parse::<bool>()
        .unwrap_or(false);
    
    let (dynamic_models, model_validator) = if dynamic_models_enabled {
        // Get Redis configuration from environment
        let redis_url = std::env::var("TENSORZERO_REDIS_URL")?;
        let stream_name = std::env::var("TENSORZERO_REDIS_STREAM")
            .unwrap_or_else(|_| "tensorzero:model_updates".to_string());
        let consumer_group = std::env::var("TENSORZERO_REDIS_CONSUMER_GROUP")
            .unwrap_or_else(|_| "gateway_group".to_string());
        let poll_interval = std::env::var("TENSORZERO_REDIS_POLL_INTERVAL")
            .unwrap_or_else(|_| "1000".to_string())
            .parse::<u64>()
            .unwrap_or(1000);
        
        // Create dynamic models configuration
        let dynamic_config = DynamicModelsConfig {
            enabled: true,
            redis_url: redis_url.clone(),
            stream_name: stream_name.clone(),
            consumer_group: consumer_group.clone(),
            consumer_name: format!("gateway_{}", std::process::id()),
            poll_interval_ms: poll_interval,
        };
        
        // Initialize dynamic models state
        let dynamic_state = Arc::new(DynamicModelsState::new(dynamic_config));
        dynamic_state.start_listener().await?;
        
        // Create model validator with dynamic support
        let validator = Arc::new(ModelValidator::new(
            config.models.clone(),
            Some(dynamic_state.clone()),
        ));
        
        tracing::info!(
            "Dynamic models enabled: Redis={}, Stream={}, Group={}",
            redis_url, stream_name, consumer_group
        );
        
        (Some(dynamic_state), validator)
    } else {
        // Create model validator without dynamic support
        let validator = Arc::new(ModelValidator::new(config.models.clone(), None));
        tracing::info!("Dynamic models disabled");
        (None, validator)
    };
    
    Ok(ExtendedAppState {
        base,
        dynamic_models,
        model_validator,
    })
}

/// Example of how to use in the gateway main.rs:
/// 
/// ```rust
/// // Replace the standard app state initialization:
/// let app_state = gateway_util::AppStateData::new(config.clone()).await?;
/// 
/// // With:
/// let extended_state = initialize_app_state_with_dynamic_models(config.clone()).await?;
/// 
/// // Then use extended_state.base for the router state
/// let router = Router::new()
///     .route("/inference", post(inference_handler))
///     .with_state(extended_state.base);
/// ```

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_initialization_without_env() {
        // Without environment variables, dynamic models should be disabled
        let config = Arc::new(Config::default());
        let state = initialize_app_state_with_dynamic_models(config).await.unwrap();
        assert!(state.dynamic_models.is_none());
    }
}