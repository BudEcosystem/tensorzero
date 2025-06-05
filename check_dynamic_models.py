#!/usr/bin/env python3
"""
Diagnostic script to check dynamic models configuration and Redis connection.
"""

import json
import os
import sys
import redis
from urllib.parse import urlparse

DEFAULT_REDIS_URL = "redis://default:budpassword@localhost:6379"
DEFAULT_STREAM_NAME = "tensorzero:model_updates"


def check_redis_connection():
    """Check if we can connect to Redis."""
    effective_redis_url = os.environ.get("REDIS_URL", DEFAULT_REDIS_URL)
    redis_username = os.environ.get("REDIS_USERNAME")
    redis_password = os.environ.get("REDIS_PASSWORD")
    
    print("=== Redis Connection Check ===")
    print(f"REDIS_URL: {effective_redis_url}")
    if redis_username:
        print(f"REDIS_USERNAME: {redis_username}")
        print(f"REDIS_PASSWORD: {'***' if redis_password else 'Not set'}")
    
    try:
        if redis_username:
            parsed_url = urlparse(effective_redis_url)
            host = parsed_url.hostname or "localhost"
            port = parsed_url.port or 6379
            db = 0
            if parsed_url.path and len(parsed_url.path) > 1 and parsed_url.path[1:].isdigit():
                db = int(parsed_url.path[1:])
            
            r = redis.Redis(
                host=host,
                port=port,
                db=db,
                username=redis_username,
                password=redis_password,
                decode_responses=True
            )
        else:
            r = redis.from_url(effective_redis_url, decode_responses=True)
        
        # Test connection
        r.ping()
        print("✓ Redis connection successful")
        return r
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
        return None


def check_stream_messages(r):
    """Check messages in the Redis stream."""
    stream_name = os.environ.get("REDIS_STREAM", DEFAULT_STREAM_NAME)
    
    print(f"\n=== Redis Stream Check ===")
    print(f"Stream name: {stream_name}")
    
    try:
        # Get stream info
        info = r.xinfo_stream(stream_name)
        print(f"✓ Stream exists")
        print(f"  Length: {info['length']}")
        print(f"  First entry: {info.get('first-entry', 'None')}")
        print(f"  Last entry: {info.get('last-entry', 'None')}")
        
        # Get consumer groups
        try:
            groups = r.xinfo_groups(stream_name)
            print(f"\nConsumer groups ({len(groups)}):")
            for group in groups:
                print(f"  - {group['name']}: {group['pending']} pending, last delivered: {group.get('last-delivered-id', 'None')}")
        except:
            print("  No consumer groups found")
        
        # Read last 10 messages
        messages = r.xrevrange(stream_name, count=10)
        print(f"\nLast {len(messages)} messages:")
        for msg_id, data in messages:
            if 'data' in data:
                try:
                    msg_data = json.loads(data['data'])
                    action = msg_data.get('action', 'unknown')
                    model_name = msg_data.get('model_name', 'unknown')
                    print(f"  {msg_id}: {action} '{model_name}'")
                except:
                    print(f"  {msg_id}: Invalid JSON")
            else:
                print(f"  {msg_id}: No 'data' field")
                
    except redis.ResponseError as e:
        if "no such key" in str(e):
            print(f"✗ Stream '{stream_name}' does not exist")
            print("  No messages have been published yet")
        else:
            print(f"✗ Error reading stream: {e}")


def check_gateway_environment():
    """Check gateway-related environment variables."""
    print("\n=== Gateway Environment Check ===")
    
    vars_to_check = [
        ("TENSORZERO_DYNAMIC_MODELS_ENABLED", "false"),
        ("TENSORZERO_REDIS_URL", "Not set"),
        ("TENSORZERO_REDIS_STREAM", DEFAULT_STREAM_NAME),
        ("TENSORZERO_REDIS_CONSUMER_GROUP", "gateway_group"),
        ("TENSORZERO_REDIS_CONSUMER_NAME", "Not set"),
        ("TENSORZERO_REDIS_POLL_INTERVAL", "1000"),
    ]
    
    for var, default in vars_to_check:
        value = os.environ.get(var, default)
        status = "✓" if var in os.environ else "✗"
        print(f"{status} {var}: {value}")
    
    # Check if dynamic models are enabled
    enabled = os.environ.get("TENSORZERO_DYNAMIC_MODELS_ENABLED", "false").lower() == "true"
    if not enabled:
        print("\n⚠️  WARNING: Dynamic models are NOT enabled!")
        print("   Set TENSORZERO_DYNAMIC_MODELS_ENABLED=true to enable")
    else:
        print("\n✓ Dynamic models are enabled")


def list_current_models(r):
    """List all models currently in the stream."""
    stream_name = os.environ.get("REDIS_STREAM", DEFAULT_STREAM_NAME)
    
    print("\n=== Current Dynamic Models ===")
    
    try:
        # Read all messages to build current model state
        messages = r.xrange(stream_name)
        models = {}
        
        for msg_id, data in messages:
            if 'data' in data:
                try:
                    msg_data = json.loads(data['data'])
                    action = msg_data.get('action')
                    model_name = msg_data.get('model_name')
                    
                    if action == 'upsert' and model_name:
                        provider_info = "unknown"
                        if 'providers' in msg_data:
                            for pname, pconfig in msg_data['providers'].items():
                                provider_info = f"{pconfig.get('kind', 'unknown')}::{pconfig.get('model_name', 'unknown')}"
                                break
                        models[model_name] = provider_info
                    elif action == 'remove' and model_name:
                        models.pop(model_name, None)
                except:
                    pass
        
        if models:
            print(f"Found {len(models)} active model(s):")
            for name, info in models.items():
                print(f"  - {name} -> {info}")
        else:
            print("No active models found")
            
    except Exception as e:
        print(f"Error listing models: {e}")


def main():
    print("TensorZero Dynamic Models Diagnostic Tool")
    print("=" * 50)
    
    # Check Redis connection
    r = check_redis_connection()
    if not r:
        print("\n❌ Cannot proceed without Redis connection")
        sys.exit(1)
    
    # Check gateway environment
    check_gateway_environment()
    
    # Check stream messages
    check_stream_messages(r)
    
    # List current models
    list_current_models(r)
    
    # Final recommendations
    print("\n=== Recommendations ===")
    
    enabled = os.environ.get("TENSORZERO_DYNAMIC_MODELS_ENABLED", "false").lower() == "true"
    if not enabled:
        print("1. Enable dynamic models: export TENSORZERO_DYNAMIC_MODELS_ENABLED=true")
    
    if "TENSORZERO_REDIS_URL" not in os.environ:
        print("2. Set Redis URL: export TENSORZERO_REDIS_URL=redis://localhost:6379")
    
    print("\nMake sure to restart the gateway after setting environment variables!")


if __name__ == "__main__":
    main()