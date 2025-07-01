import os
import pytest
import json
import time
import httpx
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configuration
TENSORZERO_BASE_URL = os.getenv("TENSORZERO_BASE_URL", "http://localhost:3001")
TENSORZERO_API_KEY = os.getenv("TENSORZERO_API_KEY", "test-api-key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Clients
tensorzero_client = OpenAI(
    base_url=f"{TENSORZERO_BASE_URL}/v1",
    api_key=TENSORZERO_API_KEY
)

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
) if OPENAI_API_KEY else None

# HTTP client for direct endpoint testing
http_client = httpx.Client(base_url=TENSORZERO_BASE_URL)


class TestRealtimeSessions:
    """Test Realtime API session management endpoints"""

    def test_create_basic_realtime_session(self):
        """Test creating a basic realtime session"""
        response = http_client.post(
            "/v1/realtime/sessions",
            json={
                "model": "gpt-4o-realtime-preview"
            }
        )
        
        assert response.status_code == 200
        session = response.json()
        
        # Verify response structure matches OpenAI format
        assert session["object"] == "realtime.session"
        assert session["id"].startswith("sess_")
        assert session["model"] == "gpt-4o-realtime-preview"
        assert session["expires_at"] == 0
        
        # Verify client secret
        client_secret = session["client_secret"]
        assert client_secret["value"].startswith("eph_")
        assert client_secret["expires_at"] > int(time.time())
        
        # Verify default values
        assert session["voice"] == "alloy"
        assert session["input_audio_format"] == "pcm16"
        assert session["output_audio_format"] == "pcm16"
        assert session["temperature"] == 0.8
        assert session["max_response_output_tokens"] == "inf"
        assert session["tool_choice"] == "auto"
        assert session["speed"] == 1.0
        
        # Verify modalities
        modalities = session["modalities"]
        assert len(modalities) == 2
        assert "text" in modalities
        assert "audio" in modalities
        
        # Verify turn detection
        turn_detection = session["turn_detection"]
        assert turn_detection["type"] == "server_vad"
        assert turn_detection["threshold"] == 0.5
        assert turn_detection["prefix_padding_ms"] == 300
        assert turn_detection["silence_duration_ms"] == 200
        assert turn_detection["create_response"] is True
        assert turn_detection["interrupt_response"] is True
        
        # Verify tools is empty array
        assert session["tools"] == []

    def test_create_realtime_session_with_custom_parameters(self):
        """Test creating a realtime session with custom parameters"""
        custom_params = {
            "model": "gpt-4o-realtime-preview-2024-12-17",
            "voice": "nova",
            "temperature": 0.5,
            "instructions": "You are a helpful coding assistant.",
            "modalities": ["text"],
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.7,
                "prefix_padding_ms": 500,
                "silence_duration_ms": 300,
                "create_response": False,
                "interrupt_response": True
            },
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_alaw",
            "max_response_output_tokens": 1000,
            "tool_choice": "none",
            "speed": 1.25
        }
        
        response = http_client.post(
            "/v1/realtime/sessions",
            json=custom_params
        )
        
        assert response.status_code == 200
        session = response.json()
        
        # Verify custom parameters are preserved
        assert session["model"] == "gpt-4o-realtime-preview-2024-12-17"
        assert session["voice"] == "nova"
        assert session["temperature"] == 0.5
        assert session["instructions"] == "You are a helpful coding assistant."
        assert session["input_audio_format"] == "g711_ulaw"
        assert session["output_audio_format"] == "g711_alaw"
        assert session["max_response_output_tokens"] == 1000
        assert session["tool_choice"] == "none"
        assert session["speed"] == 1.25
        
        # Verify custom modalities
        assert session["modalities"] == ["text"]

    def test_realtime_session_invalid_model(self):
        """Test error handling for invalid model"""
        response = http_client.post(
            "/v1/realtime/sessions",
            json={
                "model": "gpt-4"  # Regular model, not realtime
            }
        )
        
        assert response.status_code == 400
        error = response.json()
        assert "error" in error
        assert "not found or does not support realtime sessions" in error["error"]["message"]


class TestRealtimeTranscriptionSessions:
    """Test Realtime API transcription session endpoints"""

    def test_create_basic_transcription_session(self):
        """Test creating a basic transcription session"""
        response = http_client.post(
            "/v1/realtime/transcription_sessions",
            json={
                "model": "gpt-4o-realtime-preview"
            }
        )
        
        assert response.status_code == 200
        session = response.json()
        
        # Verify response structure
        assert session["object"] == "realtime.transcription_session"
        assert session["id"].startswith("sess_")
        assert session["model"] == "gpt-4o-realtime-preview"
        assert session["expires_at"] == 0
        
        # Verify client secret for transcription
        client_secret = session["client_secret"]
        assert client_secret["value"].startswith("eph_transcribe_")
        assert client_secret["expires_at"] > int(time.time())
        
        # Verify modalities is always ["text"] for transcription
        assert session["modalities"] == ["text"]

    def test_transcription_session_invalid_model(self):
        """Test error handling for invalid model in transcription sessions"""
        response = http_client.post(
            "/v1/realtime/transcription_sessions",
            json={
                "model": "gpt-4"  # Regular model, not realtime
            }
        )
        
        assert response.status_code == 400
        error = response.json()
        assert "error" in error
        assert "not found or does not support realtime transcription" in error["error"]["message"]


class TestRealtimeSDKPatterns:
    """Test patterns that match the OpenAI Python SDK usage"""

    def test_sdk_style_session_creation(self):
        """Test session creation in SDK style"""
        session_params = {
            "model": "gpt-4o-realtime-preview",
            "voice": "alloy",
            "instructions": "You are a friendly assistant that helps with coding questions.",
            "modalities": ["audio", "text"],
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 200,
                "create_response": True,
                "interrupt_response": True
            },
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "temperature": 0.8,
            "max_response_output_tokens": "inf",
            "tool_choice": "auto",
            "tools": []
        }
        
        response = http_client.post(
            "/v1/realtime/sessions",
            json=session_params
        )
        
        assert response.status_code == 200
        session = response.json()
        
        # Verify all fields match the request
        assert session["model"] == session_params["model"]
        assert session["voice"] == session_params["voice"]
        assert session["instructions"] == session_params["instructions"]
        assert set(session["modalities"]) == set(session_params["modalities"])
        assert session["input_audio_format"] == session_params["input_audio_format"]
        assert session["output_audio_format"] == session_params["output_audio_format"]
        assert session["temperature"] == session_params["temperature"]
        assert session["max_response_output_tokens"] == session_params["max_response_output_tokens"]
        assert session["tool_choice"] == session_params["tool_choice"]
        assert session["tools"] == session_params["tools"]


class TestRealtimeErrorHandling:
    """Test error handling and edge cases"""

    def test_missing_model_parameter(self):
        """Test error when model parameter is missing"""
        response = http_client.post(
            "/v1/realtime/sessions",
            json={}
        )
        
        assert response.status_code == 400

    def test_session_expiration_time(self):
        """Test that client secret has reasonable expiration time"""
        response = http_client.post(
            "/v1/realtime/sessions",
            json={
                "model": "gpt-4o-realtime-preview"
            }
        )
        
        assert response.status_code == 200
        session = response.json()
        
        client_secret = session["client_secret"]
        expires_at = client_secret["expires_at"]
        current_time = int(time.time())
        
        # Secret should expire within reasonable time (e.g., 1-3600 seconds from now)
        assert current_time < expires_at <= current_time + 3600
        
        # For our implementation, it should be exactly 60 seconds
        time_until_expiry = expires_at - current_time
        assert 55 <= time_until_expiry <= 65  # Allow some tolerance for test execution time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])