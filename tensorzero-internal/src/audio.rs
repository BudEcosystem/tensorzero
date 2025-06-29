use crate::endpoints::inference::InferenceCredentials;
use crate::error::Error;
use crate::inference::types::{Latency, Usage};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::sync::Arc;
use uuid::Uuid;

// Audio transcription types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioTranscriptionRequest {
    pub id: Uuid,
    pub file: Vec<u8>,
    pub filename: String,
    pub model: Arc<str>,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub response_format: Option<AudioTranscriptionResponseFormat>,
    pub temperature: Option<f32>,
    pub timestamp_granularities: Option<Vec<TimestampGranularity>>,
    pub chunking_strategy: Option<ChunkingStrategy>,
    pub include: Option<Vec<String>>,
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ChunkingStrategy {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "static")]
    Static { 
        #[serde(skip_serializing_if = "Option::is_none")]
        chunk_size: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        overlap_size: Option<u32>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioTranscriptionResponseFormat {
    Json,
    Text,
    Srt,
    VerboseJson,
    Vtt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimestampGranularity {
    Word,
    Segment,
}

#[derive(Debug, Clone)]
pub struct AudioTranscriptionResponse {
    pub id: Uuid,
    pub text: String,
    pub language: Option<String>,
    pub duration: Option<f32>,
    pub words: Option<Vec<WordTimestamp>>,
    pub segments: Option<Vec<SegmentTimestamp>>,
    pub created: u64,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordTimestamp {
    pub word: String,
    pub start: f32,
    pub end: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentTimestamp {
    pub id: u64,
    pub seek: u64,
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub tokens: Vec<u64>,
    pub temperature: f32,
    pub avg_logprob: f32,
    pub compression_ratio: f32,
    pub no_speech_prob: f32,
}

// Audio translation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioTranslationRequest {
    pub id: Uuid,
    pub file: Vec<u8>,
    pub filename: String,
    pub model: Arc<str>,
    pub prompt: Option<String>,
    pub response_format: Option<AudioTranscriptionResponseFormat>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct AudioTranslationResponse {
    pub id: Uuid,
    pub text: String,
    pub created: u64,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
}

// Text-to-speech types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextToSpeechRequest {
    pub id: Uuid,
    pub input: String,
    pub model: Arc<str>,
    pub voice: AudioVoice,
    pub response_format: Option<AudioOutputFormat>,
    pub speed: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioVoice {
    Alloy,
    Ash,
    Ballad,
    Coral,
    Echo,
    Fable,
    Onyx,
    Nova,
    Sage,
    Shimmer,
    Verse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioOutputFormat {
    Mp3,
    Opus,
    Aac,
    Flac,
    Wav,
    Pcm,
}

#[derive(Debug, Clone)]
pub struct TextToSpeechResponse {
    pub id: Uuid,
    pub audio_data: Vec<u8>,
    pub format: AudioOutputFormat,
    pub created: u64,
    pub raw_request: String,
    pub usage: Usage,
    pub latency: Latency,
}

// Provider response types
#[derive(Debug, Clone)]
pub struct AudioTranscriptionProviderResponse {
    pub id: Uuid,
    pub text: String,
    pub language: Option<String>,
    pub duration: Option<f32>,
    pub words: Option<Vec<WordTimestamp>>,
    pub segments: Option<Vec<SegmentTimestamp>>,
    pub created: u64,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
}

#[derive(Debug, Clone)]
pub struct AudioTranslationProviderResponse {
    pub id: Uuid,
    pub text: String,
    pub created: u64,
    pub raw_request: String,
    pub raw_response: String,
    pub usage: Usage,
    pub latency: Latency,
}

#[derive(Debug, Clone)]
pub struct TextToSpeechProviderResponse {
    pub id: Uuid,
    pub audio_data: Vec<u8>,
    pub format: AudioOutputFormat,
    pub created: u64,
    pub raw_request: String,
    pub usage: Usage,
    pub latency: Latency,
}

// Provider traits
pub trait AudioTranscriptionProvider {
    fn transcribe(
        &self,
        request: &AudioTranscriptionRequest,
        client: &Client,
        dynamic_api_keys: &InferenceCredentials,
    ) -> impl Future<Output = Result<AudioTranscriptionProviderResponse, Error>> + Send;
}

pub trait AudioTranslationProvider {
    fn translate(
        &self,
        request: &AudioTranslationRequest,
        client: &Client,
        dynamic_api_keys: &InferenceCredentials,
    ) -> impl Future<Output = Result<AudioTranslationProviderResponse, Error>> + Send;
}

pub trait TextToSpeechProvider {
    fn generate_speech(
        &self,
        request: &TextToSpeechRequest,
        client: &Client,
        dynamic_api_keys: &InferenceCredentials,
    ) -> impl Future<Output = Result<TextToSpeechProviderResponse, Error>> + Send;
}

// Streaming support for TTS
pub trait TextToSpeechStreamProvider {
    fn generate_speech_stream(
        &self,
        request: &TextToSpeechRequest,
        client: &Client,
        dynamic_api_keys: &InferenceCredentials,
    ) -> impl Future<Output = Result<(AudioStream, AudioOutputFormat), Error>> + Send;
}

pub type AudioStream = Pin<Box<dyn Stream<Item = Result<Vec<u8>, Error>> + Send>>;

use futures::Stream;
use std::pin::Pin;

impl AudioTranscriptionResponseFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            AudioTranscriptionResponseFormat::Json => "json",
            AudioTranscriptionResponseFormat::Text => "text",
            AudioTranscriptionResponseFormat::Srt => "srt",
            AudioTranscriptionResponseFormat::VerboseJson => "verbose_json",
            AudioTranscriptionResponseFormat::Vtt => "vtt",
        }
    }
}
