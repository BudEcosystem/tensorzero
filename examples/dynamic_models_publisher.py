#!/usr/bin/env python3
"""
Example script for publishing dynamic model updates to TensorZero via Redis streams.

Requirements:
    pip install redis

Usage:
    python dynamic_models_publisher.py add my-gpt4 openai gpt-4
    python dynamic_models_publisher.py remove my-gpt4
"""

import json
import sys
import redis
from typing import Dict, Any


def create_openai_model_config(model_name: str, openai_model: str) -> Dict[str, Any]:
    """Create a model configuration for OpenAI models."""
    return {
        "action": "upsert",
        "model_name": model_name,
        "routing": ["openai_provider"],
        "providers": {
            "openai_provider": {
                "kind": "openai",
                "model_name": openai_model,
                "api_key_location": "env::OPENAI_API_KEY"
            }
        }
    }


def create_anthropic_model_config(model_name: str, anthropic_model: str) -> Dict[str, Any]:
    """Create a model configuration for Anthropic models."""
    return {
        "action": "upsert",
        "model_name": model_name,
        "routing": ["anthropic_provider"],
        "providers": {
            "anthropic_provider": {
                "kind": "anthropic",
                "model_name": anthropic_model,
                "api_key_location": "env::ANTHROPIC_API_KEY"
            }
        }
    }


def create_multi_provider_model_config(model_name: str) -> Dict[str, Any]:
    """Create a model configuration with multiple providers for fallback."""
    return {
        "action": "upsert",
        "model_name": model_name,
        "routing": ["primary", "fallback"],
        "providers": {
            "primary": {
                "kind": "openai",
                "model_name": "gpt-4",
                "api_key_location": "env::OPENAI_API_KEY"
            },
            "fallback": {
                "kind": "anthropic",
                "model_name": "claude-3-opus-20240229",
                "api_key_location": "env::ANTHROPIC_API_KEY"
            }
        }
    }


def publish_model_update(
    redis_client: redis.Redis,
    stream_name: str,
    message: Dict[str, Any]
) -> str:
    """Publish a model update message to the Redis stream."""
    message_data = json.dumps(message)
    message_id = redis_client.xadd(
        stream_name,
        {"data": message_data}
    )
    return message_id.decode() if isinstance(message_id, bytes) else message_id


def main():
    # Configuration
    REDIS_URL = "redis://localhost:6379"
    STREAM_NAME = "tensorzero:model_updates"
    
    # Connect to Redis
    r = redis.from_url(REDIS_URL)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Add model:    python dynamic_models_publisher.py add <model_name> <provider> <provider_model>")
        print("  Remove model: python dynamic_models_publisher.py remove <model_name>")
        print("  Add multi:    python dynamic_models_publisher.py add-multi <model_name>")
        sys.exit(1)
    
    action = sys.argv[1]
    
    if action == "add" and len(sys.argv) >= 5:
        model_name = sys.argv[2]
        provider = sys.argv[3]
        provider_model = sys.argv[4]
        
        if provider == "openai":
            config = create_openai_model_config(model_name, provider_model)
        elif provider == "anthropic":
            config = create_anthropic_model_config(model_name, provider_model)
        else:
            print(f"Unknown provider: {provider}")
            sys.exit(1)
        
        message_id = publish_model_update(r, STREAM_NAME, config)
        print(f"Published model update for '{model_name}' (message ID: {message_id})")
    
    elif action == "add-multi" and len(sys.argv) >= 3:
        model_name = sys.argv[2]
        config = create_multi_provider_model_config(model_name)
        message_id = publish_model_update(r, STREAM_NAME, config)
        print(f"Published multi-provider model '{model_name}' (message ID: {message_id})")
    
    elif action == "remove" and len(sys.argv) >= 3:
        model_name = sys.argv[2]
        config = {
            "action": "remove",
            "model_name": model_name
        }
        message_id = publish_model_update(r, STREAM_NAME, config)
        print(f"Published model removal for '{model_name}' (message ID: {message_id})")
    
    else:
        print("Invalid command or missing arguments")
        sys.exit(1)
    
    # Verify the message was added to the stream
    messages = r.xread({STREAM_NAME: "$"}, count=1, block=100)
    if messages:
        print("Message successfully added to stream")
    
    # Show stream info
    info = r.xinfo_stream(STREAM_NAME)
    print(f"Stream '{STREAM_NAME}' now has {info['length']} messages")


if __name__ == "__main__":
    main()