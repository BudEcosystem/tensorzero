#!/usr/bin/env python3
"""
Reset the Redis consumer group to reprocess all messages.
"""

import os
import redis
from urllib.parse import urlparse

DEFAULT_REDIS_URL = "redis://default:budpassword@localhost:6379"
DEFAULT_STREAM_NAME = "tensorzero:model_updates"
DEFAULT_CONSUMER_GROUP = "gateway_group"


def reset_consumer_group():
    """Reset the consumer group to reprocess all messages."""
    effective_redis_url = os.environ.get("REDIS_URL", DEFAULT_REDIS_URL)
    redis_username = os.environ.get("REDIS_USERNAME")
    redis_password = os.environ.get("REDIS_PASSWORD")
    stream_name = os.environ.get("REDIS_STREAM", DEFAULT_STREAM_NAME)
    consumer_group = os.environ.get("REDIS_CONSUMER_GROUP", DEFAULT_CONSUMER_GROUP)
    
    # Connect to Redis
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
    
    try:
        # Delete the consumer group
        r.xgroup_destroy(stream_name, consumer_group)
        print(f"✓ Deleted consumer group '{consumer_group}'")
    except redis.ResponseError as e:
        if "no such consumer group" in str(e):
            print(f"Consumer group '{consumer_group}' does not exist")
        else:
            print(f"Error deleting consumer group: {e}")
    
    try:
        # Recreate the consumer group to read from the beginning
        r.xgroup_create(stream_name, consumer_group, id="0")
        print(f"✓ Created consumer group '{consumer_group}' to read from beginning")
    except redis.ResponseError as e:
        if "BUSYGROUP" in str(e):
            print(f"Consumer group '{consumer_group}' already exists")
        else:
            print(f"Error creating consumer group: {e}")
    
    # Show stream info
    try:
        info = r.xinfo_stream(stream_name)
        print(f"\nStream '{stream_name}':")
        print(f"  Length: {info['length']}")
        print(f"  First entry: {info.get('first-entry', 'None')}")
        print(f"  Last entry: {info.get('last-entry', 'None')}")
    except Exception as e:
        print(f"Error getting stream info: {e}")


if __name__ == "__main__":
    print("Resetting Redis consumer group...")
    reset_consumer_group()
    print("\nNow restart the gateway to reprocess all messages from the beginning.")