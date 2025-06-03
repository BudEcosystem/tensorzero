use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::interval;
use reqwest::Client;
use url::Url;

use crate::config_parser::{ProviderTypesConfig, DynamicModelServiceConfig};
use crate::error::{Error, ErrorDetails};
use crate::model::{ModelConfig, ModelProvider, ModelTable};
use crate::inference::providers::provider_trait::ProviderName;

/// Response types from the external model service
#[derive(Debug, serde::Deserialize)]
pub struct ModelServiceResponse {
    pub success: bool,
    pub result: ModelServiceResult,
    pub message: String,
}

#[derive(Debug, serde::Deserialize)]
pub struct ModelServiceResult {
    pub project_id: String,
    pub project_name: String,
    pub endpoint_name: String,
    pub model_configuration: Vec<ModelConfiguration>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelConfiguration {
    pub model_name: String,
    pub litellm_params: LiteLLMParams,
    pub model_info: ModelInfo,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LiteLLMParams {
    pub model: String,
    pub api_base: String,
    pub api_key: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelInfo {
    pub metadata: ModelMetadata,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelMetadata {
    pub provider: String,
}

/// Background task that periodically fetches and updates models directly in ModelTable
pub async fn start_dynamic_model_updater(
    model_table: Arc<RwLock<ModelTable>>,
    provider_types: Arc<ProviderTypesConfig>,
    service_config: DynamicModelServiceConfig,
) {
    let client = Client::builder()
        .timeout(Duration::from_secs(service_config.request_timeout_secs))
        .build()
        .expect("Failed to create HTTP client");

    let mut interval = interval(Duration::from_secs(service_config.refresh_interval_secs));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    tracing::info!(
        "Starting dynamic model updater with refresh interval: {} seconds",
        service_config.refresh_interval_secs
    );

    loop {
        interval.tick().await;
        
        if let Err(e) = update_models_from_service(
            &model_table,
            &provider_types,
            &service_config,
            &client,
        ).await {
            tracing::error!("Failed to update dynamic models: {}", e);
        }
    }
}

async fn update_models_from_service(
    model_table: &Arc<RwLock<ModelTable>>,
    provider_types: &Arc<ProviderTypesConfig>,
    service_config: &DynamicModelServiceConfig,
    client: &Client,
) -> Result<(), Error> {
    // Fetch all configured endpoints
    let endpoints = if let Some(endpoint) = &service_config.endpoint_name {
        vec![endpoint.clone()]
    } else {
        // In a real implementation, you might fetch a list of endpoints
        // For now, we'll just use the configured one
        return Ok(());
    };

    for endpoint_name in endpoints {
        match fetch_and_update_endpoint(
            model_table,
            provider_types,
            service_config,
            client,
            &endpoint_name,
        ).await {
            Ok(count) => {
                if count > 0 {
                    tracing::info!("Updated {} models for endpoint '{}'", count, endpoint_name);
                }
            }
            Err(e) => {
                tracing::error!("Failed to update models for endpoint '{}': {}", endpoint_name, e);
            }
        }
    }

    Ok(())
}

async fn fetch_and_update_endpoint(
    model_table: &Arc<RwLock<ModelTable>>,
    provider_types: &Arc<ProviderTypesConfig>,
    service_config: &DynamicModelServiceConfig,
    client: &Client,
    endpoint_name: &str,
) -> Result<usize, Error> {
    // Build request
    let mut request = client.get(service_config.service_url.clone());
    
    let mut query_params = vec![("endpoint_name", endpoint_name)];
    let api_key_string;
    if let Some(api_key) = &service_config.api_key {
        api_key_string = api_key.clone();
        query_params.push(("api_key", &api_key_string));
    }
    request = request.query(&query_params);

    // Send request
    let response = request.send().await.map_err(|e| {
        Error::new(ErrorDetails::InferenceClient {
            message: format!("Failed to fetch models from service: {}", e),
            status_code: None,
            provider_type: "dynamic_model_service".to_string(),
            raw_request: None,
            raw_response: None,
        })
    })?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(Error::new(ErrorDetails::InferenceClient {
            message: format!("Model service returned error: {} - {}", status, text),
            status_code: Some(status),
            provider_type: "dynamic_model_service".to_string(),
            raw_request: None,
            raw_response: Some(text),
        }));
    }

    let response_text = response.text().await.map_err(|e| {
        Error::new(ErrorDetails::InferenceClient {
            message: format!("Failed to read response: {}", e),
            status_code: None,
            provider_type: "dynamic_model_service".to_string(),
            raw_request: None,
            raw_response: None,
        })
    })?;

    let model_response: ModelServiceResponse = serde_json::from_str(&response_text).map_err(|e| {
        Error::new(ErrorDetails::InferenceClient {
            message: format!("Failed to parse response: {}", e),
            status_code: None,
            provider_type: "dynamic_model_service".to_string(),
            raw_request: None,
            raw_response: Some(response_text),
        })
    })?;

    if !model_response.success {
        return Err(Error::new(ErrorDetails::InferenceClient {
            message: format!("Service returned error: {}", model_response.message),
            status_code: None,
            provider_type: "dynamic_model_service".to_string(),
            raw_request: None,
            raw_response: None,
        }));
    }

    // Update ModelTable with fetched models
    let mut table = model_table.write().await;
    let mut updated_count = 0;

    for model_config in model_response.result.model_configuration {
        let model_key = format!("{}::{}", endpoint_name, model_config.model_name);
        
        match convert_to_model_config(&model_config, provider_types) {
            Ok(config) => {
                table.insert(Arc::from(model_key.clone()), config);
                updated_count += 1;
                tracing::debug!("Updated model: {}", model_key);
            }
            Err(e) => {
                tracing::error!("Failed to convert model '{}': {}", model_key, e);
            }
        }
    }

    Ok(updated_count)
}

fn convert_to_model_config(
    model: &ModelConfiguration,
    provider_types: &ProviderTypesConfig,
) -> Result<ModelConfig, Error> {
    // Map provider from metadata
    let provider_name = map_provider_name(&model.litellm_params.model)?;
    
    // Create provider config based on the provider type
    let provider_config = create_provider_config(provider_name, &model.litellm_params)?;
    
    // Create a single provider with a unique name
    let provider_instance_name = Arc::from("default");
    let mut providers = HashMap::new();
    
    providers.insert(
        provider_instance_name.clone(),
        ModelProvider {
            name: provider_instance_name.clone(),
            config: provider_config.load(provider_types)?,
            extra_body: None,
            extra_headers: None,
        },
    );

    Ok(ModelConfig {
        routing: vec![provider_instance_name],
        providers,
    })
}

fn map_provider_name(model_string: &str) -> Result<ProviderName, Error> {
    // Extract provider from model string (e.g., "openai/gpt-4" -> "openai")
    if let Some((provider, _)) = model_string.split_once('/') {
        match provider {
            "openai" => Ok(ProviderName::OpenAI),
            "anthropic" => Ok(ProviderName::Anthropic),
            "together" => Ok(ProviderName::Together),
            "mistral" => Ok(ProviderName::Mistral),
            "deepseek" => Ok(ProviderName::DeepSeek),
            "fireworks" => Ok(ProviderName::Fireworks),
            "hyperbolic" => Ok(ProviderName::Hyperbolic),
            "xai" => Ok(ProviderName::XAI),
            _ => Err(Error::new(ErrorDetails::Config {
                message: format!("Unknown provider: {}", provider),
            })),
        }
    } else {
        Err(Error::new(ErrorDetails::Config {
            message: format!("Invalid model format: {}", model_string),
        }))
    }
}

use crate::model::UninitializedProviderConfig;
use crate::inference::providers::openai::OpenAIProvider;
use crate::inference::providers::anthropic::AnthropicProvider;

fn create_provider_config(
    provider_name: ProviderName,
    litellm_params: &LiteLLMParams,
) -> Result<UninitializedProviderConfig, Error> {
    // Create provider-specific config from LiteLLM params
    let config_json = match provider_name {
        ProviderName::OpenAI => {
            serde_json::json!({
                "model_name": litellm_params.model.split('/').nth(1).unwrap_or(&litellm_params.model),
                "api_key_location": serde_json::json!({
                    "value": litellm_params.api_key
                })
            })
        }
        ProviderName::Anthropic => {
            serde_json::json!({
                "model_name": litellm_params.model.split('/').nth(1).unwrap_or(&litellm_params.model),
                "api_key_location": serde_json::json!({
                    "value": litellm_params.api_key
                })
            })
        }
        _ => {
            // Generic config for other providers
            serde_json::json!({
                "model_name": litellm_params.model,
                "api_key_location": serde_json::json!({
                    "value": litellm_params.api_key
                })
            })
        }
    };

    // Convert JSON to provider config
    let config_value = serde_json::Value::Object(config_json.as_object().unwrap().clone());
    serde_json::from_value(config_value).map_err(|e| {
        Error::new(ErrorDetails::Config {
            message: format!("Failed to create provider config: {}", e),
        })
    })
}