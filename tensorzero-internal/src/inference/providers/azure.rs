use std::sync::OnceLock;

use futures::{StreamExt, TryStreamExt};
use reqwest::StatusCode;
use reqwest_eventsource::RequestBuilderExt;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::time::Instant;
use url::Url;

use crate::audio::{
    AudioOutputFormat, AudioTranscriptionProvider, AudioTranscriptionProviderResponse,
    AudioTranscriptionRequest, AudioTranslationProvider, AudioTranslationProviderResponse,
    AudioTranslationRequest, TextToSpeechProvider, TextToSpeechProviderResponse,
    TextToSpeechRequest,
};
use crate::cache::ModelProviderRequest;
use crate::embeddings::{EmbeddingProvider, EmbeddingProviderResponse, EmbeddingRequest};
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{DisplayOrDebugGateway, Error, ErrorDetails};
use crate::images::{
    ImageGenerationProvider, ImageGenerationProviderResponse, ImageGenerationRequest,
};
use crate::inference::types::batch::BatchRequestRow;
use crate::inference::types::batch::PollBatchInferenceResponse;
use crate::inference::types::{
    batch::StartBatchProviderInferenceResponse, current_timestamp, Latency, ModelInferenceRequest,
    ModelInferenceRequestJsonMode, PeekableProviderInferenceResponseStream,
    ProviderInferenceResponse, Usage,
};
use crate::inference::types::{ContentBlockOutput, ProviderInferenceResponseArgs};
use crate::model::{build_creds_caching_default, Credential, CredentialLocation, ModelProvider};

use super::helpers::inject_extra_request_data;
use super::openai::{
    handle_openai_error, prepare_openai_messages, prepare_openai_tools, stream_openai,
    OpenAIRequestMessage, OpenAIResponse, OpenAIResponseChoice, OpenAITool, OpenAIToolChoice,
    OpenAIToolChoiceString, SpecificToolChoice,
};
use super::provider_trait::{InferenceProvider, TensorZeroEventError};

const PROVIDER_NAME: &str = "Azure";
const PROVIDER_TYPE: &str = "azure";

#[derive(Debug)]
pub struct AzureProvider {
    deployment_id: String,
    endpoint: Url,
    credentials: AzureCredentials,
}

static DEFAULT_CREDENTIALS: OnceLock<AzureCredentials> = OnceLock::new();

impl AzureProvider {
    pub fn new(
        deployment_id: String,
        endpoint: Url,
        api_key_location: Option<CredentialLocation>,
    ) -> Result<Self, Error> {
        let credentials = build_creds_caching_default(
            api_key_location,
            default_api_key_location(),
            PROVIDER_TYPE,
            &DEFAULT_CREDENTIALS,
        )?;
        Ok(AzureProvider {
            deployment_id,
            endpoint,
            credentials,
        })
    }

    pub fn deployment_id(&self) -> &str {
        &self.deployment_id
    }
}

#[derive(Clone, Debug, Deserialize)]
pub enum AzureCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
}

impl TryFrom<Credential> for AzureCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(AzureCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(AzureCredentials::Dynamic(key_name)),
            Credential::Missing => Ok(AzureCredentials::None),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for Azure provider".to_string(),
            })),
        }
    }
}

impl AzureCredentials {
    fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<&'a SecretString, Error> {
        match self {
            AzureCredentials::Static(api_key) => Ok(api_key),
            AzureCredentials::Dynamic(key_name) => {
                dynamic_api_keys.get(key_name).ok_or_else(|| {
                    ErrorDetails::ApiKeyMissing {
                        provider_name: PROVIDER_NAME.to_string(),
                    }
                    .into()
                })
            }
            AzureCredentials::None => Err(ErrorDetails::ApiKeyMissing {
                provider_name: PROVIDER_NAME.to_string(),
            }
            .into()),
        }
    }
}

fn default_api_key_location() -> CredentialLocation {
    CredentialLocation::Env("AZURE_OPENAI_API_KEY".to_string())
}

impl InferenceProvider for AzureProvider {
    async fn infer<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
        }: ModelProviderRequest<'a>,
        http_client: &'a reqwest::Client,
        api_key: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<ProviderInferenceResponse, Error> {
        let mut request_body = serde_json::to_value(AzureRequest::new(request)?).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing Azure request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        let headers = inject_extra_request_data(
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            &mut request_body,
        )?;
        let request_url = get_azure_chat_url(&self.endpoint, &self.deployment_id)?;
        let start_time = Instant::now();
        let api_key = self.credentials.get_api_key(api_key)?;
        let res = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .header("api-key", api_key.expose_secret())
            .headers(headers)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: e.to_string(),
                    status_code: Some(e.status().unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;
        if res.status().is_success() {
            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };

            let raw_response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing text response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;

            let response = serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}: {raw_response}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: Some(raw_response.clone()),
                })
            })?;

            Ok(AzureResponseWithMetadata {
                response,
                latency,
                request: request_body,
                generic_request: request,
                raw_response,
            }
            .try_into()?)
        } else {
            let status = res.status();
            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing error response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;
            Err(handle_openai_error(
                &serde_json::to_string(&request_body).unwrap_or_default(),
                status,
                &response,
                PROVIDER_TYPE,
            ))
        }
    }

    async fn infer_stream<'a>(
        &'a self,
        ModelProviderRequest {
            request,
            provider_name: _,
            model_name,
        }: ModelProviderRequest<'a>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
        model_provider: &'a ModelProvider,
    ) -> Result<(PeekableProviderInferenceResponseStream, String), Error> {
        let mut request_body = serde_json::to_value(AzureRequest::new(request)?).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing Azure request: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        let headers = inject_extra_request_data(
            &request.extra_body,
            &request.extra_headers,
            model_provider,
            model_name,
            &mut request_body,
        )?;
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing request body as JSON: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        let request_url = get_azure_chat_url(&self.endpoint, &self.deployment_id)?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
        let event_source = http_client
            .post(request_url)
            .header("Content-Type", "application/json")
            .header("api-key", api_key.expose_secret())
            .headers(headers)
            .json(&request_body)
            .eventsource()
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    message: format!(
                        "Error sending request to Azure: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    status_code: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(raw_request.clone()),
                    raw_response: None,
                })
            })?;
        let stream = stream_openai(
            PROVIDER_TYPE.to_string(),
            event_source.map_err(TensorZeroEventError::EventSource),
            start_time,
        )
        .peekable();
        Ok((stream, raw_request))
    }

    async fn start_batch_inference<'a>(
        &'a self,
        _requests: &'a [ModelInferenceRequest<'_>],
        _client: &'a reqwest::Client,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<StartBatchProviderInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: "Azure".to_string(),
        }
        .into())
    }

    async fn poll_batch_inference<'a>(
        &'a self,
        _batch_request: &'a BatchRequestRow<'a>,
        _http_client: &'a reqwest::Client,
        _dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<PollBatchInferenceResponse, Error> {
        Err(ErrorDetails::UnsupportedModelProviderForBatchInference {
            provider_type: PROVIDER_TYPE.to_string(),
        }
        .into())
    }
}

fn get_azure_chat_url(endpoint: &Url, deployment_id: &str) -> Result<Url, Error> {
    let mut url = endpoint.clone();
    url.path_segments_mut()
        .map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error parsing URL: {e:?}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?
        .push("openai")
        .push("deployments")
        .push(deployment_id)
        .push("chat")
        .push("completions");
    url.query_pairs_mut()
        .append_pair("api-version", "2024-10-21");
    Ok(url)
}

fn get_azure_embeddings_url(endpoint: &Url, deployment_id: &str) -> Result<Url, Error> {
    let mut url = endpoint.clone();
    url.path_segments_mut()
        .map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error parsing URL: {e:?}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?
        .push("openai")
        .push("deployments")
        .push(deployment_id)
        .push("embeddings");
    url.query_pairs_mut()
        .append_pair("api-version", "2024-10-21");
    Ok(url)
}

fn get_azure_transcriptions_url(endpoint: &Url, deployment_id: &str) -> Result<Url, Error> {
    let mut url = endpoint.clone();
    url.path_segments_mut()
        .map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error parsing URL: {e:?}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?
        .push("openai")
        .push("deployments")
        .push(deployment_id)
        .push("audio")
        .push("transcriptions");
    url.query_pairs_mut()
        .append_pair("api-version", "2024-10-21");
    Ok(url)
}

fn get_azure_translations_url(endpoint: &Url, deployment_id: &str) -> Result<Url, Error> {
    let mut url = endpoint.clone();
    url.path_segments_mut()
        .map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error parsing URL: {e:?}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?
        .push("openai")
        .push("deployments")
        .push(deployment_id)
        .push("audio")
        .push("translations");
    url.query_pairs_mut()
        .append_pair("api-version", "2024-10-21");
    Ok(url)
}

fn get_azure_speech_url(endpoint: &Url, deployment_id: &str) -> Result<Url, Error> {
    let mut url = endpoint.clone();
    url.path_segments_mut()
        .map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error parsing URL: {e:?}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?
        .push("openai")
        .push("deployments")
        .push(deployment_id)
        .push("audio")
        .push("speech");
    url.query_pairs_mut()
        .append_pair("api-version", "2024-10-21");
    Ok(url)
}

fn get_azure_images_generations_url(endpoint: &Url, deployment_id: &str) -> Result<Url, Error> {
    let mut url = endpoint.clone();
    url.path_segments_mut()
        .map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("Error parsing URL: {e:?}"),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?
        .push("openai")
        .push("deployments")
        .push(deployment_id)
        .push("images")
        .push("generations");
    url.query_pairs_mut()
        .append_pair("api-version", "2024-10-21");
    Ok(url)
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(untagged)]
enum AzureToolChoice<'a> {
    String(AzureToolChoiceString),
    Specific(SpecificToolChoice<'a>),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum AzureToolChoiceString {
    None,
    Auto,
    // Note: Azure doesn't support required tool choice.
}

impl<'a> From<OpenAIToolChoice<'a>> for AzureToolChoice<'a> {
    fn from(tool_choice: OpenAIToolChoice<'a>) -> Self {
        match tool_choice {
            OpenAIToolChoice::String(tool_choice) => {
                match tool_choice {
                    OpenAIToolChoiceString::None => {
                        AzureToolChoice::String(AzureToolChoiceString::None)
                    }
                    OpenAIToolChoiceString::Auto => {
                        AzureToolChoice::String(AzureToolChoiceString::Auto)
                    }
                    OpenAIToolChoiceString::Required => {
                        AzureToolChoice::String(AzureToolChoiceString::Auto)
                    } // Azure doesn't support required
                }
            }
            OpenAIToolChoice::Specific(tool_choice) => AzureToolChoice::Specific(tool_choice),
        }
    }
}

/// This struct defines the supported parameters for the Azure OpenAI inference API
/// See the [API documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart)
/// for more details.
/// We are not handling logprobs, top_logprobs, n, prompt_truncate_len
/// presence_penalty, frequency_penalty, seed, service_tier, stop, user,
/// or context_length_exceeded_behavior
#[derive(Debug, Serialize)]
struct AzureRequest<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<AzureResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AzureToolChoice<'a>>,
}

impl<'a> AzureRequest<'a> {
    pub fn new(request: &'a ModelInferenceRequest<'_>) -> Result<AzureRequest<'a>, Error> {
        let response_format = AzureResponseFormat::new(&request.json_mode, request.output_schema);
        let messages = prepare_openai_messages(request)?;
        let (tools, tool_choice, _) = prepare_openai_tools(request);
        Ok(AzureRequest {
            messages,
            temperature: request.temperature,
            top_p: request.top_p,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            max_tokens: request.max_tokens,
            stream: request.stream,
            response_format,
            seed: request.seed,
            tools,
            tool_choice: tool_choice.map(AzureToolChoice::from),
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
pub enum AzureResponseFormat {
    #[default]
    Text,
    JsonObject,
    JsonSchema {
        json_schema: Value,
    },
}

impl AzureResponseFormat {
    fn new(
        json_mode: &ModelInferenceRequestJsonMode,
        output_schema: Option<&Value>,
    ) -> Option<Self> {
        // Note: Some models on Azure won't support strict JSON mode.
        // Azure will 400 if you try to use it for those.
        // See these docs: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs
        match json_mode {
            ModelInferenceRequestJsonMode::On => Some(AzureResponseFormat::JsonObject),
            // For now, we never explicitly send `AzureResponseFormat::Text`
            ModelInferenceRequestJsonMode::Off => None,
            ModelInferenceRequestJsonMode::Strict => match output_schema {
                Some(schema) => {
                    let json_schema = json!({"name": "response", "strict": true, "schema": schema});
                    Some(AzureResponseFormat::JsonSchema { json_schema })
                }
                None => Some(AzureResponseFormat::JsonObject),
            },
        }
    }
}

struct AzureResponseWithMetadata<'a> {
    response: OpenAIResponse,
    raw_response: String,
    latency: Latency,
    request: serde_json::Value,
    generic_request: &'a ModelInferenceRequest<'a>,
}

impl<'a> TryFrom<AzureResponseWithMetadata<'a>> for ProviderInferenceResponse {
    type Error = Error;
    fn try_from(value: AzureResponseWithMetadata<'a>) -> Result<Self, Self::Error> {
        let AzureResponseWithMetadata {
            mut response,
            latency,
            request: request_body,
            generic_request,
            raw_response,
        } = value;

        if response.choices.len() != 1 {
            return Err(ErrorDetails::InferenceServer {
                message: format!(
                    "Response has invalid number of choices: {}. Expected 1.",
                    response.choices.len()
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: Some(raw_response.clone()),
            }
            .into());
        }
        let system = generic_request.system.clone();
        let input_messages = generic_request.messages.clone();
        let usage = response.usage.into();
        let OpenAIResponseChoice {
            message,
            finish_reason,
            ..
        } = response
            .choices
            .pop()
            .ok_or_else(|| Error::new(ErrorDetails::InferenceServer {
                message: "Response has no choices (this should never happen). Please file a bug report: https://github.com/tensorzero/tensorzero/issues/new".to_string(),
                raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                raw_response: Some(raw_response.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            }))?;
        let mut content: Vec<ContentBlockOutput> = Vec::new();
        if let Some(text) = message.content {
            content.push(text.into());
        }
        if let Some(tool_calls) = message.tool_calls {
            for tool_call in tool_calls {
                content.push(ContentBlockOutput::ToolCall(tool_call.into()));
            }
        }
        let raw_request = serde_json::to_string(&request_body).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error serializing request body as JSON: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;

        Ok(ProviderInferenceResponse::new(
            ProviderInferenceResponseArgs {
                output: content,
                system,
                input_messages,
                raw_request,
                raw_response,
                usage,
                latency,
                finish_reason: Some(finish_reason.into()),
            },
        ))
    }
}

// Embedding support
impl EmbeddingProvider for AzureProvider {
    async fn embed(
        &self,
        request: &EmbeddingRequest,
        client: &reqwest::Client,
        dynamic_api_keys: &InferenceCredentials,
    ) -> Result<EmbeddingProviderResponse, Error> {
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let request_body = serde_json::json!({
            "model": &self.deployment_id,
            "input": &request.input,
            "encoding_format": request.encoding_format.as_deref(),
        });
        let request_url = get_azure_embeddings_url(&self.endpoint, &self.deployment_id)?;
        let start_time = Instant::now();
        let res = client
            .post(request_url)
            .header("Content-Type", "application/json")
            .header("api-key", api_key.expose_secret())
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    status_code: e.status(),
                    message: format!(
                        "Error sending request to Azure: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;
        if res.status().is_success() {
            let raw_response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing text response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;
            let response: serde_json::Value = serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}"),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: Some(raw_response.clone()),
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;
            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };

            let data = response["data"].as_array().ok_or_else(|| {
                Error::new(ErrorDetails::InferenceServer {
                    message: "Missing 'data' field in embedding response".to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: Some(raw_response.clone()),
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

            let embeddings: Vec<Vec<f32>> = data
                .iter()
                .map(|item| {
                    item["embedding"]
                        .as_array()
                        .ok_or_else(|| {
                            Error::new(ErrorDetails::InferenceServer {
                                message: "Missing 'embedding' field in response data".to_string(),
                                raw_request: Some(
                                    serde_json::to_string(&request_body).unwrap_or_default(),
                                ),
                                raw_response: Some(raw_response.clone()),
                                provider_type: PROVIDER_TYPE.to_string(),
                            })
                        })
                        .and_then(|arr| {
                            arr.iter()
                                .map(|v| {
                                    v.as_f64().map(|f| f as f32).ok_or_else(|| {
                                        Error::new(ErrorDetails::InferenceServer {
                                            message: "Invalid float value in embedding".to_string(),
                                            raw_request: Some(
                                                serde_json::to_string(&request_body)
                                                    .unwrap_or_default(),
                                            ),
                                            raw_response: Some(raw_response.clone()),
                                            provider_type: PROVIDER_TYPE.to_string(),
                                        })
                                    })
                                })
                                .collect::<Result<Vec<f32>, Error>>()
                        })
                })
                .collect::<Result<Vec<Vec<f32>>, Error>>()?;

            let usage = if let Some(usage_obj) = response.get("usage") {
                Usage {
                    input_tokens: usage_obj["prompt_tokens"].as_u64().unwrap_or(0) as u32,
                    output_tokens: usage_obj["completion_tokens"].as_u64().unwrap_or(0) as u32,
                }
            } else {
                Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                }
            };
            Ok(EmbeddingProviderResponse::new(
                embeddings,
                request.input.clone(),
                serde_json::to_string(&request_body).unwrap_or_default(),
                raw_response,
                usage,
                latency,
            ))
        } else {
            let status = res.status();
            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing error response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;
            Err(handle_openai_error(
                &serde_json::to_string(&request_body).unwrap_or_default(),
                status,
                &response,
                PROVIDER_TYPE,
            ))
        }
    }
}

// Audio transcription support
impl AudioTranscriptionProvider for AzureProvider {
    async fn transcribe(
        &self,
        request: &AudioTranscriptionRequest,
        client: &reqwest::Client,
        dynamic_api_keys: &InferenceCredentials,
    ) -> Result<AudioTranscriptionProviderResponse, Error> {
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let request_url = get_azure_transcriptions_url(&self.endpoint, &self.deployment_id)?;

        // Create multipart form
        let mut form = reqwest::multipart::Form::new()
            .part(
                "file",
                reqwest::multipart::Part::bytes(request.file.clone())
                    .file_name(request.filename.clone()),
            )
            .text("model", self.deployment_id.clone());

        if let Some(language) = &request.language {
            form = form.text("language", language.clone());
        }
        if let Some(prompt) = &request.prompt {
            form = form.text("prompt", prompt.clone());
        }
        if let Some(response_format) = &request.response_format {
            form = form.text("response_format", response_format.as_str());
        }
        if let Some(temperature) = request.temperature {
            form = form.text("temperature", temperature.to_string());
        }
        if let Some(timestamp_granularities) = &request.timestamp_granularities {
            for granularity in timestamp_granularities {
                form = form.text(
                    "timestamp_granularities[]",
                    match granularity {
                        crate::audio::TimestampGranularity::Word => "word",
                        crate::audio::TimestampGranularity::Segment => "segment",
                    },
                );
            }
        }

        let start_time = Instant::now();
        let res = client
            .post(request_url)
            .header("api-key", api_key.expose_secret())
            .multipart(form)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    status_code: e.status(),
                    message: format!(
                        "Error sending request to Azure: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(format!(
                        "Audio transcription request for file: {}",
                        request.filename
                    )),
                    raw_response: None,
                })
            })?;

        if res.status().is_success() {
            let raw_response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing text response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(format!(
                        "Audio transcription request for file: {}",
                        request.filename
                    )),
                    raw_response: None,
                })
            })?;

            let response: serde_json::Value = serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(format!(
                        "Audio transcription request for file: {}",
                        request.filename
                    )),
                    raw_response: Some(raw_response.clone()),
                })
            })?;

            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };

            let usage = Usage {
                input_tokens: 0, // Azure doesn't provide token usage for audio
                output_tokens: 0,
            };

            Ok(AudioTranscriptionProviderResponse {
                id: request.id,
                text: response["text"].as_str().unwrap_or("").to_string(),
                language: response
                    .get("language")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                duration: response
                    .get("duration")
                    .and_then(|v| v.as_f64())
                    .map(|f| f as f32),
                words: response
                    .get("words")
                    .and_then(|v| v.as_array())
                    .map(|words| {
                        words
                            .iter()
                            .filter_map(|w| {
                                Some(crate::audio::WordTimestamp {
                                    word: w["word"].as_str()?.to_string(),
                                    start: w["start"].as_f64()? as f32,
                                    end: w["end"].as_f64()? as f32,
                                })
                            })
                            .collect()
                    }),
                segments: response
                    .get("segments")
                    .and_then(|v| v.as_array())
                    .map(|segments| {
                        segments
                            .iter()
                            .filter_map(|s| {
                                Some(crate::audio::SegmentTimestamp {
                                    id: s["id"].as_u64()?,
                                    seek: s["seek"].as_u64()?,
                                    start: s["start"].as_f64()? as f32,
                                    end: s["end"].as_f64()? as f32,
                                    text: s["text"].as_str()?.to_string(),
                                    tokens: s["tokens"]
                                        .as_array()?
                                        .iter()
                                        .filter_map(|t| t.as_u64())
                                        .collect(),
                                    temperature: s["temperature"].as_f64()? as f32,
                                    avg_logprob: s["avg_logprob"].as_f64()? as f32,
                                    compression_ratio: s["compression_ratio"].as_f64()? as f32,
                                    no_speech_prob: s["no_speech_prob"].as_f64()? as f32,
                                })
                            })
                            .collect()
                    }),
                created: current_timestamp(),
                raw_request: format!("Audio transcription request for file: {}", request.filename),
                raw_response,
                usage,
                latency,
            })
        } else {
            let status = res.status();
            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing error response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(format!(
                        "Audio transcription request for file: {}",
                        request.filename
                    )),
                    raw_response: None,
                })
            })?;
            Err(handle_openai_error(
                &format!("Audio transcription request for file: {}", request.filename),
                status,
                &response,
                PROVIDER_TYPE,
            ))
        }
    }
}

// Audio translation support
impl AudioTranslationProvider for AzureProvider {
    async fn translate(
        &self,
        request: &AudioTranslationRequest,
        client: &reqwest::Client,
        dynamic_api_keys: &InferenceCredentials,
    ) -> Result<AudioTranslationProviderResponse, Error> {
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let request_url = get_azure_translations_url(&self.endpoint, &self.deployment_id)?;

        // Create multipart form
        let mut form = reqwest::multipart::Form::new()
            .part(
                "file",
                reqwest::multipart::Part::bytes(request.file.clone())
                    .file_name(request.filename.clone()),
            )
            .text("model", self.deployment_id.clone());

        if let Some(prompt) = &request.prompt {
            form = form.text("prompt", prompt.clone());
        }
        if let Some(response_format) = &request.response_format {
            form = form.text("response_format", response_format.as_str());
        }
        if let Some(temperature) = request.temperature {
            form = form.text("temperature", temperature.to_string());
        }

        let start_time = Instant::now();
        let res = client
            .post(request_url)
            .header("api-key", api_key.expose_secret())
            .multipart(form)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    status_code: e.status(),
                    message: format!(
                        "Error sending request to Azure: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(format!(
                        "Audio translation request for file: {}",
                        request.filename
                    )),
                    raw_response: None,
                })
            })?;

        if res.status().is_success() {
            let raw_response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing text response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(format!(
                        "Audio translation request for file: {}",
                        request.filename
                    )),
                    raw_response: None,
                })
            })?;

            let response: serde_json::Value = serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(format!(
                        "Audio translation request for file: {}",
                        request.filename
                    )),
                    raw_response: Some(raw_response.clone()),
                })
            })?;

            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };

            let usage = Usage {
                input_tokens: 0, // Azure doesn't provide token usage for audio
                output_tokens: 0,
            };

            Ok(AudioTranslationProviderResponse {
                id: request.id,
                text: response["text"].as_str().unwrap_or("").to_string(),
                created: current_timestamp(),
                raw_request: format!("Audio translation request for file: {}", request.filename),
                raw_response,
                usage,
                latency,
            })
        } else {
            let status = res.status();
            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing error response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(format!(
                        "Audio translation request for file: {}",
                        request.filename
                    )),
                    raw_response: None,
                })
            })?;
            Err(handle_openai_error(
                &format!("Audio translation request for file: {}", request.filename),
                status,
                &response,
                PROVIDER_TYPE,
            ))
        }
    }
}

// Text-to-speech support
impl TextToSpeechProvider for AzureProvider {
    async fn generate_speech(
        &self,
        request: &TextToSpeechRequest,
        client: &reqwest::Client,
        dynamic_api_keys: &InferenceCredentials,
    ) -> Result<TextToSpeechProviderResponse, Error> {
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let request_url = get_azure_speech_url(&self.endpoint, &self.deployment_id)?;

        let request_body = serde_json::json!({
            "model": &self.deployment_id,
            "input": &request.input,
            "voice": &request.voice,
            "response_format": request.response_format.as_ref().map(|f| f.as_str()),
            "speed": request.speed,
        });

        let start_time = Instant::now();
        let res = client
            .post(request_url)
            .header("Content-Type", "application/json")
            .header("api-key", api_key.expose_secret())
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    status_code: e.status(),
                    message: format!(
                        "Error sending request to Azure: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;

        if res.status().is_success() {
            let audio_data = res.bytes().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing audio response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;

            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };

            let usage = Usage {
                input_tokens: 0, // Azure doesn't provide token usage for TTS
                output_tokens: 0,
            };

            Ok(TextToSpeechProviderResponse {
                id: request.id,
                audio_data: audio_data.to_vec(),
                format: request
                    .response_format
                    .clone()
                    .unwrap_or(AudioOutputFormat::Mp3),
                created: current_timestamp(),
                raw_request: serde_json::to_string(&request_body).unwrap_or_default(),
                usage,
                latency,
            })
        } else {
            let status = res.status();
            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing error response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;
            Err(handle_openai_error(
                &serde_json::to_string(&request_body).unwrap_or_default(),
                status,
                &response,
                PROVIDER_TYPE,
            ))
        }
    }
}

// Image generation support
impl ImageGenerationProvider for AzureProvider {
    async fn generate_image(
        &self,
        request: &ImageGenerationRequest,
        client: &reqwest::Client,
        dynamic_api_keys: &InferenceCredentials,
    ) -> Result<ImageGenerationProviderResponse, Error> {
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let request_url = get_azure_images_generations_url(&self.endpoint, &self.deployment_id)?;

        let request_body = serde_json::json!({
            "prompt": &request.prompt,
            "n": request.n,
            "size": request.size.as_ref().map(|s| s.as_str()),
            "quality": request.quality.as_ref().map(|q| q.as_str()),
            "response_format": request.response_format.as_ref().map(|f| f.as_str()),
            "style": request.style.as_ref().map(|s| s.as_str()),
            "user": request.user.as_ref(),
        });

        let start_time = Instant::now();
        let res = client
            .post(request_url)
            .header("Content-Type", "application/json")
            .header("api-key", api_key.expose_secret())
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    status_code: e.status(),
                    message: format!(
                        "Error sending request to Azure: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;

        if res.status().is_success() {
            let raw_response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing text response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;

            let response: serde_json::Value = serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing JSON response: {e}"),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: Some(raw_response.clone()),
                })
            })?;

            let latency = Latency::NonStreaming {
                response_time: start_time.elapsed(),
            };

            let usage = Usage {
                input_tokens: 0, // Azure doesn't provide token usage for images
                output_tokens: 0,
            };

            let data_array = response["data"].as_array().ok_or_else(|| {
                Error::new(ErrorDetails::InferenceServer {
                    message: "Missing 'data' field in image generation response".to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: Some(raw_response.clone()),
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

            let data = data_array
                .iter()
                .map(|item| {
                    Ok(crate::images::ImageData {
                        url: item
                            .get("url")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string()),
                        b64_json: item
                            .get("b64_json")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string()),
                        revised_prompt: item
                            .get("revised_prompt")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string()),
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;

            Ok(ImageGenerationProviderResponse {
                id: request.id,
                created: response["created"]
                    .as_u64()
                    .unwrap_or_else(current_timestamp),
                data,
                raw_request: serde_json::to_string(&request_body).unwrap_or_default(),
                raw_response,
                usage,
                latency,
            })
        } else {
            let status = res.status();
            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing error response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                })
            })?;
            Err(handle_openai_error(
                &serde_json::to_string(&request_body).unwrap_or_default(),
                status,
                &response,
                PROVIDER_TYPE,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::time::Duration;

    use uuid::Uuid;

    use super::*;

    use crate::inference::providers::openai::{
        OpenAIFinishReason, OpenAIResponseChoice, OpenAIResponseMessage, OpenAIToolType,
        OpenAIUsage, SpecificToolFunction,
    };
    use crate::inference::providers::test_helpers::{WEATHER_TOOL, WEATHER_TOOL_CONFIG};
    use crate::inference::types::{
        FinishReason, FunctionType, ModelInferenceRequestJsonMode, RequestMessage, Role,
    };

    #[test]
    fn test_azure_request_new() {
        let request_with_tools = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: Some(0.5),
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: Some(100),
            stream: false,
            seed: Some(69),
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let azure_request = AzureRequest::new(&request_with_tools).unwrap();

        assert_eq!(azure_request.messages.len(), 1);
        assert_eq!(azure_request.temperature, Some(0.5));
        assert_eq!(azure_request.max_tokens, Some(100));
        assert!(!azure_request.stream);
        assert_eq!(azure_request.seed, Some(69));
        assert_eq!(azure_request.response_format, None);
        assert!(azure_request.tools.is_some());
        let tools = azure_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);

        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            azure_request.tool_choice,
            Some(AzureToolChoice::Specific(SpecificToolChoice {
                r#type: OpenAIToolType::Function,
                function: SpecificToolFunction {
                    name: WEATHER_TOOL.name(),
                }
            }))
        );

        let request_with_tools = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather?".to_string().into()],
            }],
            system: None,
            temperature: Some(0.5),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            max_tokens: Some(100),
            stream: false,
            seed: Some(69),
            json_mode: ModelInferenceRequestJsonMode::On,
            tool_config: Some(Cow::Borrowed(&WEATHER_TOOL_CONFIG)),
            function_type: FunctionType::Json,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };

        let azure_request = AzureRequest::new(&request_with_tools).unwrap();

        assert_eq!(azure_request.messages.len(), 2);
        assert_eq!(azure_request.temperature, Some(0.5));
        assert_eq!(azure_request.max_tokens, Some(100));
        assert_eq!(azure_request.top_p, Some(0.9));
        assert_eq!(azure_request.presence_penalty, Some(0.1));
        assert_eq!(azure_request.frequency_penalty, Some(0.2));
        assert!(!azure_request.stream);
        assert_eq!(azure_request.seed, Some(69));
        assert_eq!(
            azure_request.response_format,
            Some(AzureResponseFormat::JsonObject)
        );
        assert!(azure_request.tools.is_some());
        let tools = azure_request.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);

        assert_eq!(tools[0].function.name, WEATHER_TOOL.name());
        assert_eq!(tools[0].function.parameters, WEATHER_TOOL.parameters());
        assert_eq!(
            azure_request.tool_choice,
            Some(AzureToolChoice::Specific(SpecificToolChoice {
                r#type: OpenAIToolType::Function,
                function: SpecificToolFunction {
                    name: WEATHER_TOOL.name(),
                }
            }))
        );
    }

    #[test]
    fn test_azure_tool_choice_from() {
        // Required is converted to Auto
        let tool_choice = OpenAIToolChoice::String(OpenAIToolChoiceString::Required);
        let azure_tool_choice = AzureToolChoice::from(tool_choice);
        assert_eq!(
            azure_tool_choice,
            AzureToolChoice::String(AzureToolChoiceString::Auto)
        );

        // Specific tool choice is converted to Specific
        let specific_tool_choice = OpenAIToolChoice::Specific(SpecificToolChoice {
            r#type: OpenAIToolType::Function,
            function: SpecificToolFunction {
                name: "test_function",
            },
        });
        let azure_specific_tool_choice = AzureToolChoice::from(specific_tool_choice);
        assert_eq!(
            azure_specific_tool_choice,
            AzureToolChoice::Specific(SpecificToolChoice {
                r#type: OpenAIToolType::Function,
                function: SpecificToolFunction {
                    name: "test_function",
                }
            })
        );

        // None is converted to None
        let none_tool_choice = OpenAIToolChoice::String(OpenAIToolChoiceString::None);
        let azure_none_tool_choice = AzureToolChoice::from(none_tool_choice);
        assert_eq!(
            azure_none_tool_choice,
            AzureToolChoice::String(AzureToolChoiceString::None)
        );

        // Auto is converted to Auto
        let auto_tool_choice = OpenAIToolChoice::String(OpenAIToolChoiceString::Auto);
        let azure_auto_tool_choice = AzureToolChoice::from(auto_tool_choice);
        assert_eq!(
            azure_auto_tool_choice,
            AzureToolChoice::String(AzureToolChoiceString::Auto)
        );
    }

    #[test]
    fn test_credential_to_azure_credentials() {
        // Test Static credential
        let generic = Credential::Static(SecretString::from("test_key"));
        let creds = AzureCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, AzureCredentials::Static(_)));

        // Test Dynamic credential
        let generic = Credential::Dynamic("key_name".to_string());
        let creds = AzureCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, AzureCredentials::Dynamic(_)));

        // Test Missing credential
        let generic = Credential::Missing;
        let creds = AzureCredentials::try_from(generic).unwrap();
        assert!(matches!(creds, AzureCredentials::None));

        // Test invalid type
        let generic = Credential::FileContents(SecretString::from("test"));
        let result = AzureCredentials::try_from(generic);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().get_owned_details(),
            ErrorDetails::Config { message } if message.contains("Invalid api_key_location")
        ));
    }

    #[test]
    fn test_azure_response_with_metadata_try_into() {
        let valid_response = OpenAIResponse {
            choices: vec![OpenAIResponseChoice {
                index: 0,
                message: OpenAIResponseMessage {
                    content: Some("Hello, world!".to_string()),
                    tool_calls: None,
                    reasoning_content: None,
                },
                finish_reason: OpenAIFinishReason::Stop,
            }],
            usage: OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            },
        };
        let generic_request = ModelInferenceRequest {
            inference_id: Uuid::now_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["test_user".to_string().into()],
            }],
            system: None,
            temperature: Some(0.5),
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            max_tokens: Some(100),
            stream: false,
            seed: Some(69),
            json_mode: ModelInferenceRequestJsonMode::Off,
            tool_config: None,
            function_type: FunctionType::Chat,
            output_schema: None,
            extra_body: Default::default(),
            ..Default::default()
        };
        let azure_response_with_metadata = AzureResponseWithMetadata {
            response: valid_response,
            raw_response: "test_response".to_string(),
            latency: Latency::NonStreaming {
                response_time: Duration::from_secs(0),
            },
            request: serde_json::to_value(AzureRequest::new(&generic_request).unwrap()).unwrap(),
            generic_request: &generic_request,
        };
        let inference_response: ProviderInferenceResponse =
            azure_response_with_metadata.try_into().unwrap();

        assert_eq!(inference_response.output.len(), 1);
        assert_eq!(
            inference_response.output[0],
            "Hello, world!".to_string().into()
        );
        assert_eq!(inference_response.raw_response, "test_response");
        assert_eq!(inference_response.usage.input_tokens, 10);
        assert_eq!(inference_response.usage.output_tokens, 20);
        assert_eq!(inference_response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(
            inference_response.latency,
            Latency::NonStreaming {
                response_time: Duration::from_secs(0)
            }
        );
    }
}
