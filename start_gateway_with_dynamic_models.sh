#!/bin/bash
# Start TensorZero gateway with dynamic models support

echo "Starting TensorZero Gateway with Dynamic Models Support"
echo "======================================================="

# Set dynamic models environment variables
export TENSORZERO_DYNAMIC_MODELS_ENABLED=true
export TENSORZERO_REDIS_URL="${TENSORZERO_REDIS_URL:-redis://default:budpassword@localhost:6379}"
export TENSORZERO_REDIS_STREAM="${TENSORZERO_REDIS_STREAM:-tensorzero:model_updates}"
export TENSORZERO_REDIS_CONSUMER_GROUP="${TENSORZERO_REDIS_CONSUMER_GROUP:-gateway_group}"
export TENSORZERO_REDIS_CONSUMER_NAME="${TENSORZERO_REDIS_CONSUMER_NAME:-gateway_$$}"
export TENSORZERO_REDIS_POLL_INTERVAL="${TENSORZERO_REDIS_POLL_INTERVAL:-1000}"

# Set other required environment variables if not already set
export TENSORZERO_CLICKHOUSE_URL="${TENSORZERO_CLICKHOUSE_URL:-http://budadmin:6pDQ2TEw@localhost:8246/tensorzero}"

# Display configuration
echo "Configuration:"
echo "  TENSORZERO_DYNAMIC_MODELS_ENABLED: $TENSORZERO_DYNAMIC_MODELS_ENABLED"
echo "  TENSORZERO_REDIS_URL: $TENSORZERO_REDIS_URL"
echo "  TENSORZERO_REDIS_STREAM: $TENSORZERO_REDIS_STREAM"
echo "  TENSORZERO_REDIS_CONSUMER_GROUP: $TENSORZERO_REDIS_CONSUMER_GROUP"
echo "  TENSORZERO_REDIS_CONSUMER_NAME: $TENSORZERO_REDIS_CONSUMER_NAME"
echo "  TENSORZERO_REDIS_POLL_INTERVAL: $TENSORZERO_REDIS_POLL_INTERVAL"
echo ""

# Check if cargo is available
if ! command -v cargo &> /dev/null; then
    echo "Error: cargo not found. Please install Rust."
    exit 1
fi

# Start the gateway
echo "Starting gateway..."
cd "$(dirname "$0")"
cargo run --bin gateway -- --config-file tensorzero.toml "$@"