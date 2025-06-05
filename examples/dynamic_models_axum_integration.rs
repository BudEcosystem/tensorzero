use std::sync::Arc;
use axum::{
    Router,
    routing::post,
    extract::{State, Extension},
    response::Json,
    middleware,
};
use serde_json::{json, Value};
use tensorzero_internal::{
    dynamic_models_middleware::{
        DynamicModelsState, DynamicModelsConfig, dynamic_models_middleware, ValidatedModel
    },
    dynamic_models_integration::ExtendedModelTable,
    gateway_util::AppState,
    config_parser::Config,
};

/// Example handler that uses the validated model information
async fn inference_handler(
    State(app_state): State<AppState>,
    Extension(validated_model): Extension<ValidatedModel>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    // The middleware has already validated the model exists
    let model_name = &validated_model.model_name;
    let is_dynamic = validated_model.is_dynamic;
    
    println!("Processing inference for model: {} (dynamic: {})", model_name, is_dynamic);
    
    // Here you would use the ExtendedModelTable to get the actual model config
    // and proceed with inference
    
    Json(json!({
        "status": "success",
        "model": model_name,
        "is_dynamic": is_dynamic,
        "message": "Model validated successfully"
    }))
}

/// Example of how to set up the gateway with dynamic models
pub async fn setup_gateway_with_dynamic_models(
    config: Config,
) -> Result<Router, Box<dyn std::error::Error>> {
    // Configure dynamic models
    let dynamic_config = DynamicModelsConfig {
        enabled: true,
        redis_url: std::env::var("REDIS_URL")
            .unwrap_or_else(|_| "redis://localhost:6379".to_string()),
        stream_name: "tensorzero:model_updates".to_string(),
        consumer_group: "gateway_group".to_string(),
        consumer_name: format!("gateway_{}", std::process::id()),
        poll_interval_ms: 1000,
    };
    
    // Create the dynamic models state
    let dynamic_state = Arc::new(DynamicModelsState::new(dynamic_config.clone()));
    
    // Start the Redis listener
    dynamic_state.start_listener().await?;
    
    // Create the extended model table
    let extended_model_table = ExtendedModelTable::new(
        config.models,
        Some(dynamic_config),
        Arc::new(config.providers),
    );
    
    // Start the dynamic listener for the extended table
    extended_model_table.start_dynamic_listener().await?;
    
    // Create your app state (simplified version)
    // In real implementation, you'd use the extended_model_table here
    let app_state = AppState {
        // ... your app state fields
    };
    
    // Build the router with middleware
    let app = Router::new()
        // Main inference endpoint
        .route("/v1/inference", post(inference_handler))
        // Add the dynamic models middleware
        .layer(middleware::from_fn_with_state(
            dynamic_state.clone(),
            dynamic_models_middleware,
        ))
        // Add the app state
        .with_state(app_state);
    
    Ok(app)
}

/// Alternative: Middleware that integrates with ExtendedModelTable
pub async fn extended_model_validation_middleware<B>(
    State(extended_table): State<Arc<ExtendedModelTable>>,
    mut request: axum::http::Request<B>,
    next: middleware::Next<B>,
) -> Result<axum::response::Response, axum::http::StatusCode>
where
    B: axum::body::HttpBody + Send + 'static,
    B::Data: Send,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    // Extract the model name from the request
    // (You'd need to implement body parsing similar to above)
    
    // For now, just pass through
    Ok(next.run(request).await)
}

/// Example usage in main
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load your configuration
    let config = Config::default(); // Load from your config file
    
    // Set up the gateway with dynamic models
    let app = setup_gateway_with_dynamic_models(config).await?;
    
    // Run the server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    println!("Server running on http://0.0.0.0:3000");
    axum::serve(listener, app).await?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;
    
    #[tokio::test]
    async fn test_model_validation() {
        // Create a test app
        let dynamic_config = DynamicModelsConfig {
            enabled: false, // Disabled for testing
            redis_url: "redis://localhost:6379".to_string(),
            stream_name: "test".to_string(),
            consumer_group: "test".to_string(),
            consumer_name: "test".to_string(),
            poll_interval_ms: 1000,
        };
        
        let dynamic_state = Arc::new(DynamicModelsState::new(dynamic_config));
        
        let app = Router::new()
            .route("/v1/inference", post(inference_handler))
            .layer(middleware::from_fn_with_state(
                dynamic_state,
                dynamic_models_middleware,
            ));
        
        // Test request
        let request = Request::builder()
            .method("POST")
            .uri("/v1/inference")
            .header("content-type", "application/json")
            .body(r#"{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}"#)
            .unwrap();
        
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
}