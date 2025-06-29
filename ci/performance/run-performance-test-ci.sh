#!/bin/bash
set -euo pipefail

# CI-specific performance test script
# Assumes binaries are already built

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Configuration
DURATION="${PERF_TEST_DURATION:-10s}"
RATE="${PERF_TEST_RATE:-100}"
TIMEOUT="${PERF_TEST_TIMEOUT:-2s}"
OUTPUT_FILE="${PERF_TEST_OUTPUT:-performance-results.json}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo "🚀 Starting TensorZero Performance Test (CI)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Duration: $DURATION"
echo "Rate: $RATE requests/sec"
echo ""

# Function to check if a process is running
check_process() {
    if kill -0 "$1" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to cleanup processes
cleanup() {
    local exit_code=$?
    echo ""
    echo "🧹 Cleaning up..."
    
    if [ -n "${GATEWAY_PID:-}" ] && check_process "$GATEWAY_PID"; then
        kill "$GATEWAY_PID" 2>/dev/null || true
    fi
    
    if [ -n "${MOCK_PID:-}" ] && check_process "$MOCK_PID"; then
        kill "$MOCK_PID" 2>/dev/null || true
    fi
    
    # Show logs on failure
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "❌ Test failed. Showing logs:"
        if [ -f gateway.log ]; then
            echo ""
            echo "Gateway logs (last 30 lines):"
            tail -30 gateway.log
        fi
        if [ -f mock-provider.log ]; then
            echo ""
            echo "Mock provider logs (last 30 lines):"
            tail -30 mock-provider.log
        fi
    fi
}

# Set up cleanup on exit
trap cleanup EXIT

# Check if binaries exist
if [ ! -f "$PROJECT_ROOT/target/release/gateway" ] || [ ! -f "$PROJECT_ROOT/target/release/mock-inference-provider" ]; then
    echo -e "${RED}❌ Error: Release binaries not found!${NC}"
    echo "Expected:"
    echo "  - $PROJECT_ROOT/target/release/gateway"
    echo "  - $PROJECT_ROOT/target/release/mock-inference-provider"
    exit 1
fi

# Start mock inference provider
echo "🎭 Starting mock inference provider..."
cd "$PROJECT_ROOT"
./target/release/mock-inference-provider > mock-provider.log 2>&1 &
MOCK_PID=$!

# Wait for mock provider using nc (netcat)
echo "⏳ Waiting for mock provider..."
for i in {1..20}; do
    if nc -z localhost 3030 2>/dev/null; then
        echo -e "${GREEN}✓ Mock provider is ready${NC}"
        break
    fi
    if [ $i -eq 20 ]; then
        echo -e "${RED}❌ Mock provider failed to start${NC}"
        exit 1
    fi
    sleep 1
done

# Start gateway
echo "🌉 Starting TensorZero gateway..."
./target/release/gateway --config-file "$PROJECT_ROOT/tensorzero-internal/tests/load/tensorzero-without-observability.toml" > gateway.log 2>&1 &
GATEWAY_PID=$!

# Wait for gateway
echo "⏳ Waiting for gateway..."
for i in {1..20}; do
    if curl -s -f http://localhost:3000/health >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Gateway is ready${NC}"
        break
    fi
    if [ $i -eq 20 ]; then
        echo -e "${RED}❌ Gateway failed to become healthy${NC}"
        exit 1
    fi
    sleep 1
done

# Test single request
echo ""
echo "🧪 Testing single request..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST http://localhost:3000/inference \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer budserve_ApTuiKIpZEjMytHt3nmCJqBQGaIlqc2TfoKYFusk" \
    -d @"$SCRIPT_DIR/test-request.json" \
    --max-time 5)

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
if [ "$HTTP_CODE" != "200" ]; then
    echo -e "${RED}❌ Test request failed with HTTP code: $HTTP_CODE${NC}"
    echo "Response:"
    echo "$RESPONSE" | head -n-1
    exit 1
else
    echo -e "${GREEN}✓ Test request successful${NC}"
fi

# Run performance test
echo ""
echo "📊 Running performance test..."
echo 'POST http://localhost:3000/inference' | \
vegeta attack \
    -header="Content-Type: application/json" \
    -header="Authorization: Bearer budserve_ApTuiKIpZEjMytHt3nmCJqBQGaIlqc2TfoKYFusk" \
    -body="$SCRIPT_DIR/test-request.json" \
    -duration="$DURATION" \
    -rate="$RATE" \
    -timeout="$TIMEOUT" \
    | vegeta encode | \
    vegeta report -type=json > "$OUTPUT_FILE"

# Parse and display results
echo ""
echo "📈 Performance Test Results"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Extract key metrics using jq
if command -v jq >/dev/null 2>&1; then
    LATENCY_MEAN=$(jq -r '.latencies.mean' "$OUTPUT_FILE" | awk '{print $1/1000000}')
    LATENCY_P50=$(jq -r '.latencies."50th"' "$OUTPUT_FILE" | awk '{print $1/1000000}')
    LATENCY_P95=$(jq -r '.latencies."95th"' "$OUTPUT_FILE" | awk '{print $1/1000000}')
    LATENCY_P99=$(jq -r '.latencies."99th"' "$OUTPUT_FILE" | awk '{print $1/1000000}')
    SUCCESS_RATE=$(jq -r '.success' "$OUTPUT_FILE" | awk '{print $1*100}')
    THROUGHPUT=$(jq -r '.throughput' "$OUTPUT_FILE")
    
    printf "Latency:\n"
    printf "  P50:  %7.2f ms\n" "$LATENCY_P50"
    printf "  P95:  %7.2f ms\n" "$LATENCY_P95"
    printf "  P99:  %7.2f ms\n" "$LATENCY_P99"
    printf "  Mean: %7.2f ms\n" "$LATENCY_MEAN"
    printf "\n"
    printf "Success Rate: %.2f%%\n" "$SUCCESS_RATE"
    printf "Throughput:   %.2f req/s\n" "$THROUGHPUT"
    
    # Basic threshold check using awk instead of bc for better compatibility
    echo ""
    if [ $(awk -v p99="$LATENCY_P99" 'BEGIN { print (p99 > 50.0) }') -eq 1 ]; then
        echo -e "${YELLOW}⚠️  Warning: P99 latency is high (${LATENCY_P99}ms)${NC}"
    fi
    
    if [ $(awk -v sr="$SUCCESS_RATE" 'BEGIN { print (sr < 99.0) }') -eq 1 ]; then
        echo -e "${RED}❌ Error: Success rate is too low (${SUCCESS_RATE}%)${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}✅ Performance test completed successfully!${NC}"
echo "📄 Full results saved to: $OUTPUT_FILE"