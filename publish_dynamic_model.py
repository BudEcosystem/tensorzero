#!/usr/bin/env python3
"""
Helper script to publish dynamic model updates to Redis.

Environment Variables for Redis Connection:
  REDIS_URL:          Full Redis URL (e.g., redis://user:pass@host:port/db).
                      Defaults to "redis://localhost:6379".
  REDIS_USERNAME:     Redis username. If set, this overrides username in REDIS_URL
                      and uses host/port/db from REDIS_URL.
  REDIS_PASSWORD:     Redis password. Used with REDIS_USERNAME.
  REDIS_STREAM:       Redis stream name. Defaults to "tensorzero:model_updates".

Usage:
    # Add a model
    python publish_dynamic_model.py add my-gpt4 openai gpt-4o-mini
    
    # Remove a model
    python publish_dynamic_model.py remove my-gpt4
    
    # Add a model with custom Redis URL
    REDIS_URL=redis://myredis:6379 python publish_dynamic_model.py add my-claude anthropic claude-3-haiku-20240307

    # Add a model with custom Redis URL and credentials via separate environment variables
    REDIS_URL=redis://myredishost:6380/2 REDIS_USERNAME=myuser REDIS_PASSWORD=mypass \
    python publish_dynamic_model.py add my-model openai gpt-4
"""

import json
import os
import sys
import redis
from typing import Dict, Any
from urllib.parse import urlparse

DEFAULT_REDIS_URL = "redis://default:budpassword@localhost:6379"
DEFAULT_STREAM_NAME = "tensorzero:model_updates"


def create_model_config(model_name: str, provider_kind: str, provider_model_name: str) -> Dict[str, Any]:
    """Create a model configuration for the given parameters."""
    provider_name = f"{model_name}_provider"
    
    config = {
        "routing": [provider_name],
        "providers": {
            provider_name: {
                "type": provider_kind,
                "model_name": provider_model_name,
            }
        }
    }
    
    # Add API key location based on provider
    if provider_kind == "openai":
        config["providers"][provider_name]["api_key_location"] = "env::OPENAI_API_KEY"
    elif provider_kind == "anthropic":
        config["providers"][provider_name]["api_key_location"] = "env::ANTHROPIC_API_KEY"
    elif provider_kind == "mistral":
        config["providers"][provider_name]["api_key_location"] = "env::MISTRAL_API_KEY"
    elif provider_kind == "deepseek":
        config["providers"][provider_name]["api_key_location"] = "env::DEEPSEEK_API_KEY"
    elif provider_kind == "together":
        config["providers"][provider_name]["api_key_location"] = "env::TOGETHER_API_KEY"
    elif provider_kind == "fireworks":
        config["providers"][provider_name]["api_key_location"] = "env::FIREWORKS_API_KEY"
    elif provider_kind == "hyperbolic":
        config["providers"][provider_name]["api_key_location"] = "env::HYPERBOLIC_API_KEY"
    elif provider_kind == "x":
        config["providers"][provider_name]["api_key_location"] = "env::XAI_API_KEY"
    # Add more providers as needed
    
    return config


def publish_model_update(action: str, model_name: str, provider_kind: str = None, provider_model_name: str = None):
    """Publish a model update to Redis."""
    effective_redis_url = os.environ.get("REDIS_URL", DEFAULT_REDIS_URL)
    redis_username = os.environ.get("REDIS_USERNAME")
    redis_password = os.environ.get("REDIS_PASSWORD")
    stream_name = os.environ.get("REDIS_STREAM", DEFAULT_STREAM_NAME)
    
    # Connect to Redis
    r: redis.Redis
    
    connection_options = {"decode_responses": True}

    if redis_username:
        # If REDIS_USERNAME is set, use it along with REDIS_PASSWORD (if set).
        # Host, port, and db are derived from effective_redis_url.
        parsed_url = urlparse(effective_redis_url)
        
        host = parsed_url.hostname or "localhost"
        port = parsed_url.port or 6379
        db = 0
        # Extract db number from path if present (e.g., /0, /1, etc.)
        if parsed_url.path and len(parsed_url.path) > 1 and parsed_url.path[1:].isdigit():
            db = int(parsed_url.path[1:])
        
        connection_options.update({
            "host": host,
            "port": port,
            "db": db,
            "username": redis_username,
        })
        if redis_password:  # Only add password to options if it's actually provided
            connection_options["password"] = redis_password
        
        r = redis.Redis(**connection_options)
        print(f"Connecting to Redis at {host}:{port}, db: {db}, with username: '{redis_username}'")
    else:
        # If REDIS_USERNAME is not set, use from_url.
        # This will correctly use credentials if they are embedded in effective_redis_url.
        r = redis.from_url(effective_redis_url, decode_responses=True)
        print(f"Connecting to Redis using URL: {effective_redis_url}")
    
    if action == "add":
        if not provider_kind or not provider_model_name:
            print("Error: provider_kind and provider_model_name are required for 'add' action", file=sys.stderr)
            sys.exit(1)
            
        config = create_model_config(model_name, provider_kind, provider_model_name)
        message = {
            "action": "upsert",
            "model_name": model_name,
            **config
        }
    elif action == "remove":
        message = {
            "action": "remove",
            "model_name": model_name
        }
    else:
        print(f"Error: Unknown action '{action}'. Use 'add' or 'remove'.", file=sys.stderr)
        sys.exit(1)
    
    # Publish to Redis stream
    message_json = json.dumps(message)
    try:
        message_id = r.xadd(stream_name, {"data": message_json})
        
        print(f"Published {action} message for model '{model_name}' to Redis stream '{stream_name}'")
        print(f"Message ID: {message_id}")
        print(f"Message: {json.dumps(message, indent=2)}")
    except redis.exceptions.RedisError as e:
        print(f"Error publishing to Redis: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Add a model:    python publish_dynamic_model.py add <model_name> <provider_kind> <provider_model_name>")
        print("  Remove a model: python publish_dynamic_model.py remove <model_name>")
        print()
        print("Supported providers: openai, anthropic, mistral, deepseek, together, fireworks, hyperbolic, x")
        print()
        print("Environment Variables for Redis Connection:")
        print("  REDIS_URL:      (default: redis://localhost:6379)")
        print("  REDIS_USERNAME: (optional)")
        print("  REDIS_PASSWORD: (optional, used with REDIS_USERNAME)")
        print("  REDIS_STREAM:   (default: tensorzero:model_updates)")
        print()
        print("Examples:")
        print("  python publish_dynamic_model.py add my-gpt4 openai gpt-4o-mini")
        print("  python publish_dynamic_model.py add my-claude anthropic claude-3-haiku-20240307")
        print("  REDIS_URL=redis://user:pass@otherhost:1234/1 python publish_dynamic_model.py remove my-gpt4")
        sys.exit(1)
    
    action = sys.argv[1]
    model_name = sys.argv[2]
    
    if action == "add":
        if len(sys.argv) < 5:
            print("Error: 'add' action requires provider_kind and provider_model_name", file=sys.stderr)
            print("Usage: python publish_dynamic_model.py add <model_name> <provider_kind> <provider_model_name>", file=sys.stderr)
            sys.exit(1)
        provider_kind = sys.argv[3]
        provider_model_name = sys.argv[4]
        publish_model_update(action, model_name, provider_kind, provider_model_name)
    elif action == "remove":
        if len(sys.argv) > 3:
            print("Error: 'remove' action takes only <model_name>", file=sys.stderr)
            print("Usage: python publish_dynamic_model.py remove <model_name>", file=sys.stderr)
            sys.exit(1)
        publish_model_update(action, model_name)
    else:
        print(f"Error: Unknown action '{action}'. Use 'add' or 'remove'.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()