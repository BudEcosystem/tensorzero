use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::Deserialize;
use reqwest::Client;
use url::Url;
use secrecy::SecretString;

use crate::error::{Error, ErrorDetails};
use crate::model::{ModelConfig, UninitializedModelConfig, UninitializedModelProvider};
use crate::config_parser::ProviderTypesConfig;

/// Configuration for the dynamic model service integration
#[derive(Debug, Clone, Deserialize)]
pub struct DynamicModelServiceConfig {
    /// URL of the model service API
    pub service_url: Url,
    /// Optional API key for authentication
    pub api_key: Option<String>,
    /// Optional default endpoint name (can be overridden at runtime)
    pub endpoint_name: Option<String>,
    /// How often to refresh the model list (in seconds)
    #[serde(default = "default_refresh_interval")]
    pub refresh_interval_secs: u64,
    /// Timeout for API requests (in seconds)
    #[serde(default = "default_request_timeout")]
    pub request_timeout_secs: u64,
}

fn default_refresh_interval() -> u64 {
    60 // 1 minute
}

fn default_request_timeout() -> u64 {
    10 // 10 seconds
}

/// Response from the model service API
#[derive(Debug, Deserialize)]
pub struct ModelServiceResponse {
    pub success: bool,
    pub result: ModelServiceResult,
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelServiceResult {
    pub project_id: String,
    pub project_name: String,
    pub endpoint_name: String,
    pub routing_policy: Option<serde_json::Value>,
    pub cache_configuration: Option<serde_json::Value>,
    pub model_configuration: Vec<ModelConfiguration>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfiguration {
    pub model_name: String,
    pub litellm_params: LiteLLMParams,
    pub model_info: ModelInfo,
    pub input_cost_per_token: Option<f64>,
    pub output_cost_per_token: Option<f64>,
    pub tpm: Option<u64>,
    pub rpm: Option<u64>,
    pub complexity_threshold: Option<f64>,
    pub weight: Option<f64>,
    pub cool_down_period: Option<f64>,
    pub fallback_endpoint_ids: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LiteLLMParams {
    pub model: String,
    pub api_base: String,
    pub api_key: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub metadata: ModelMetadata,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub provider: String,
    pub modality: Vec<String>,
    pub endpoint_id: String,
    pub cloud: bool,
    pub deploy_model_uri: String,
}

/// Information about a dynamically loaded model (internal representation)
#[derive(Debug, Clone)]
pub struct DynamicModelInfo {
    /// Model name used for inference
    pub model_name: String,
    /// LiteLLM parameters
    pub litellm_params: LiteLLMParams,
    /// Model metadata
    pub metadata: ModelMetadata,
}

/// Cache for dynamic models with automatic refresh
#[derive(Debug)]
pub struct DynamicModelCache {
    config: DynamicModelServiceConfig,
    cache: Arc<RwLock<CacheState>>,
    client: Client,
}

#[derive(Debug)]
struct CacheState {
    // Nested map: endpoint_name -> model_name -> DynamicModelInfo
    models_by_endpoint: HashMap<String, HashMap<String, DynamicModelInfo>>,
    // Track last refresh time per endpoint
    last_refresh_by_endpoint: HashMap<String, Instant>,
}

impl DynamicModelCache {
    pub fn new(config: DynamicModelServiceConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            config,
            cache: Arc::new(RwLock::new(CacheState {
                models_by_endpoint: HashMap::new(),
                last_refresh_by_endpoint: HashMap::new(),
            })),
            client,
        }
    }

    /// Get a model by endpoint name, refreshing cache if needed
    /// The input is treated as an endpoint name, and we return the first available model
    /// Format: "endpoint_name" or "endpoint_name::specific_model"
    pub async fn get_model(&self, model_name_or_endpoint: &str) -> Result<Option<DynamicModelInfo>, Error> {
        // Check if this is endpoint::model format or just endpoint
        let (endpoint_name, specific_model) = if let Some((endpoint, model)) = model_name_or_endpoint.split_once("::") {
            (endpoint.to_string(), Some(model.to_string()))
        } else {
            // Treat the entire string as an endpoint name
            (model_name_or_endpoint.to_string(), None)
        };
        
        // Check if we need to refresh for this endpoint
        {
            let cache = self.cache.read().await;
            if let Some(last_refresh) = cache.last_refresh_by_endpoint.get(&endpoint_name) {
                let elapsed = last_refresh.elapsed();
                if elapsed < Duration::from_secs(self.config.refresh_interval_secs) {
                    // Cache is still fresh, return from cache
                    if let Some(endpoint_models) = cache.models_by_endpoint.get(&endpoint_name) {
                        if let Some(specific_model) = &specific_model {
                            // Looking for a specific model
                            return Ok(endpoint_models.get(specific_model).cloned());
                        } else {
                            // Return the first available model for this endpoint
                            return Ok(endpoint_models.values().next().cloned());
                        }
                    }
                    return Ok(None);
                }
            }
        }

        // Cache is stale or doesn't exist for this endpoint, refresh it
        self.refresh_cache_for_endpoint(&endpoint_name).await?;

        // Return from refreshed cache
        let cache = self.cache.read().await;
        Ok(cache.models_by_endpoint
            .get(&endpoint_name)
            .and_then(|models| {
                if let Some(specific_model) = &specific_model {
                    models.get(specific_model).cloned()
                } else {
                    // Return the first available model
                    models.values().next().cloned()
                }
            }))
    }
    

    /// Refresh the model cache for a specific endpoint
    pub async fn refresh_cache_for_endpoint(&self, endpoint_name: &str) -> Result<(), Error> {
        let models = self.fetch_models_for_endpoint(endpoint_name).await?;
        
        let mut cache = self.cache.write().await;
        
        // Clear existing models for this endpoint and count models
        let model_count = {
            let endpoint_models = cache.models_by_endpoint.entry(endpoint_name.to_string()).or_insert_with(HashMap::new);
            endpoint_models.clear();
            
            // Insert new models
            for model in models {
                endpoint_models.insert(model.model_name.clone(), model);
            }
            
            endpoint_models.len()
        };
        
        // Update refresh time for this endpoint
        cache.last_refresh_by_endpoint.insert(endpoint_name.to_string(), Instant::now());
        
        tracing::info!("Refreshed dynamic model cache for endpoint '{}' with {} models", endpoint_name, model_count);
        Ok(())
    }

    /// Fetch models from the external service for a specific endpoint
    async fn fetch_models_for_endpoint(&self, endpoint_name: &str) -> Result<Vec<DynamicModelInfo>, Error> {
        let mut request = self.client.get(self.config.service_url.clone());
        
        // Add required query parameters
        let mut query_params = vec![("endpoint_name", endpoint_name)];
        let api_key_string;
        if let Some(api_key) = &self.config.api_key {
            api_key_string = api_key.clone();
            query_params.push(("api_key", &api_key_string));
        }
        request = request.query(&query_params);
        
        tracing::debug!("Fetching models from dynamic service: {:?}", request);

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
                message: format!(
                    "Model service returned error status: {} - {}",
                    status,
                    text
                ),
                status_code: Some(status),
                provider_type: "dynamic_model_service".to_string(),
                raw_request: None,
                raw_response: Some(text),
            }));
        }

        let response_text = response.text().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                message: format!("Failed to read response body: {}", e),
                status_code: None,
                provider_type: "dynamic_model_service".to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?;

        let model_response: ModelServiceResponse = serde_json::from_str(&response_text).map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                message: format!("Failed to parse model service response: {}", e),
                status_code: None,
                provider_type: "dynamic_model_service".to_string(),
                raw_request: None,
                raw_response: Some(response_text),
            })
        })?;

        if !model_response.success {
            return Err(Error::new(ErrorDetails::InferenceClient {
                message: format!("Model service returned error: {}", model_response.message),
                status_code: None,
                provider_type: "dynamic_model_service".to_string(),
                raw_request: None,
                raw_response: None,
            }));
        }

        // Convert model configurations to internal representation
        let models: Vec<DynamicModelInfo> = model_response.result.model_configuration
            .into_iter()
            .map(|config| DynamicModelInfo {
                model_name: config.model_name,
                litellm_params: config.litellm_params,
                metadata: config.model_info.metadata,
            })
            .collect();

        tracing::info!("Successfully fetched {} model(s) from dynamic service", models.len());
        for model in &models {
            let api_key_info = if model.litellm_params.api_key.is_empty() {
                "<empty>".to_string()
            } else if model.litellm_params.api_key.len() > 10 {
                format!("{}...", &model.litellm_params.api_key[..10])
            } else {
                "<short>".to_string()
            };
            tracing::debug!(
                "Dynamic model available: {} -> {} (api_key: {})",
                model.model_name,
                model.litellm_params.model,
                api_key_info
            );
        }

        Ok(models)
    }

    /// Convert a DynamicModelInfo to a ModelConfig
    pub fn to_model_config(
        &self,
        model_info: &DynamicModelInfo,
        provider_types: &ProviderTypesConfig,
    ) -> Result<ModelConfig, Error> {
        // Create the provider configuration based on the provider type
        let provider_config = self.create_provider_config(model_info)?;
        
        // Create an uninitialized model config
        let uninitialized = UninitializedModelConfig {
            routing: vec![Arc::from("dynamic")],
            providers: {
                let mut providers = HashMap::new();
                providers.insert(
                    Arc::from("dynamic"),
                    UninitializedModelProvider {
                        config: provider_config,
                        extra_body: None,
                        extra_headers: None,
                    },
                );
                providers
            },
        };

        // Load the model config
        uninitialized.load(&model_info.model_name, provider_types)
    }

    fn create_provider_config(
        &self,
        model_info: &DynamicModelInfo,
    ) -> Result<crate::model::UninitializedProviderConfig, Error> {
        use crate::model::UninitializedProviderConfig;
        use crate::model::CredentialLocation;
        
        // Since the API returns LiteLLM params with OpenAI format, we'll create an OpenAI provider config
        // The model name from litellm_params includes the provider prefix (e.g., "openai/bud-test-50dc44ed")
        let parts: Vec<&str> = model_info.litellm_params.model.split('/').collect();
        let (provider_type, model_name) = if parts.len() >= 2 {
            (parts[0], parts[1..].join("/"))
        } else {
            ("openai", model_info.litellm_params.model.clone())
        };

        // Create API key credential location
        // If the API key is empty or a placeholder, use environment variable
        let api_key_location = if model_info.litellm_params.api_key.is_empty() 
            || model_info.litellm_params.api_key.starts_with("${") 
            || model_info.litellm_params.api_key == "PLACEHOLDER" {
            tracing::debug!(
                "Dynamic model API key is empty/placeholder for provider '{}', using environment variable",
                provider_type
            );
            // Use environment variable based on provider type
            match provider_type {
                "openai" => Some(CredentialLocation::Env("OPENAI_API_KEY".to_string())),
                "anthropic" => Some(CredentialLocation::Env("ANTHROPIC_API_KEY".to_string())),
                _ => None,
            }
        } else {
            tracing::debug!(
                "Using API key from dynamic model service for provider '{}'",
                provider_type
            );
            // Set the API key as an environment variable with a unique name
            let env_var_name = format!("TENSORZERO_DYNAMIC_KEY_{}", model_info.model_name.to_uppercase().replace("-", "_"));
            std::env::set_var(&env_var_name, &model_info.litellm_params.api_key);
            tracing::debug!("Set dynamic API key in environment variable: {}", env_var_name);
            Some(CredentialLocation::Env(env_var_name))
        };

        match provider_type {
            "openai" => Ok(UninitializedProviderConfig::OpenAI {
                model_name,
                api_base: Some(Url::parse(&model_info.litellm_params.api_base).map_err(|e| {
                    Error::new(ErrorDetails::Config {
                        message: format!("Invalid API base URL: {}", e),
                    })
                })?),
                api_key_location,
            }),
            "anthropic" => Ok(UninitializedProviderConfig::Anthropic {
                model_name,
                api_key_location,
            }),
            "azure" => Ok(UninitializedProviderConfig::Azure {
                deployment_id: model_name,
                endpoint: Url::parse(&model_info.litellm_params.api_base).map_err(|e| {
                    Error::new(ErrorDetails::Config {
                        message: format!("Invalid Azure endpoint URL: {}", e),
                    })
                })?,
                api_key_location,
            }),
            _ => {
                // Default to OpenAI provider for unknown providers
                Ok(UninitializedProviderConfig::OpenAI {
                    model_name: model_info.litellm_params.model.clone(),
                    api_base: Some(Url::parse(&model_info.litellm_params.api_base).map_err(|e| {
                        Error::new(ErrorDetails::Config {
                            message: format!("Invalid API base URL: {}", e),
                        })
                    })?),
                    api_key_location,
                })
            }
        }
    }
}

