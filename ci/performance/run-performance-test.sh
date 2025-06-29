#!/bin/bash
set -euo pipefail

# Performance test script for CI environments
# This is a lighter version of the full performance benchmarks

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Configuration
DURATION="${PERF_TEST_DURATION:-10s}"
RATE="${PERF_TEST_RATE:-100}"
TIMEOUT="${PERF_TEST_TIMEOUT:-2s}"
WARMUP_REQUESTS="${PERF_TEST_WARMUP:-50}"
OUTPUT_FILE="${PERF_TEST_OUTPUT:-performance-results.json}"

# Use release profile in CI to avoid long build times
PROFILE="${PERF_TEST_PROFILE:-performance}"
if [ "${CI:-false}" = "true" ]; then
    PROFILE="release"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo "🚀 Starting TensorZero Performance Test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Duration: $DURATION"
echo "Rate: $RATE requests/sec"
echo "Timeout: $TIMEOUT"
echo "Warmup: $WARMUP_REQUESTS requests"
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
    echo ""
    echo "🧹 Cleaning up..."
    
    # If we're exiting with an error, show logs
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Test failed. Showing logs:"
        if [ -f gateway.log ]; then
            echo ""
            echo "Gateway logs:"
            tail -50 gateway.log
        fi
        if [ -f mock-provider.log ]; then
            echo ""
            echo "Mock provider logs:"
            tail -50 mock-provider.log
        fi
    fi
    
    if [ -n "${GATEWAY_PID:-}" ] && check_process "$GATEWAY_PID"; then
        echo "  Stopping gateway (PID: $GATEWAY_PID)"
        kill "$GATEWAY_PID" 2>/dev/null || true
        wait "$GATEWAY_PID" 2>/dev/null || true
    fi
    
    if [ -n "${MOCK_PID:-}" ] && check_process "$MOCK_PID"; then
        echo "  Stopping mock provider (PID: $MOCK_PID)"
        kill "$MOCK_PID" 2>/dev/null || true
        wait "$MOCK_PID" 2>/dev/null || true
    fi
}

# Set up cleanup on exit
trap cleanup EXIT

# Step 1: Build everything with selected profile
echo "📦 Building with $PROFILE profile..."
cd "$PROJECT_ROOT"
cargo build --profile "$PROFILE" --bin gateway --bin mock-inference-provider

# Step 2: Start mock inference provider
echo ""
echo "🎭 Starting mock inference provider..."
cargo run --profile "$PROFILE" --bin mock-inference-provider > mock-provider.log 2>&1 &
MOCK_PID=$!

# Wait for mock provider to be ready
echo "⏳ Waiting for mock provider to be ready..."
for i in {1..30}; do
    # Check if the mock provider is listening on port 3030
    if nc -z localhost 3030 2>/dev/null; then
        echo -e "${GREEN}✓ Mock provider is ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ Mock provider failed to start${NC}"
        echo "Mock provider logs:"
        cat mock-provider.log
        exit 1
    fi
    sleep 1
done

# Step 3: Start gateway
echo "🌉 Starting TensorZero gateway..."
TENSORZERO_CLICKHOUSE_URL="${TENSORZERO_CLICKHOUSE_URL:-}" \
cargo run --profile "$PROFILE" --bin gateway -- \
    --config-file "$PROJECT_ROOT/tensorzero-internal/tests/load/tensorzero-without-observability.toml" \
    > gateway.log 2>&1 &
GATEWAY_PID=$!

# Wait for gateway to be ready
echo "⏳ Waiting for gateway to be ready..."
for i in {1..30}; do
    if curl -s -f http://localhost:3000/health >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Gateway is ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ Gateway failed to become healthy${NC}"
        echo "Gateway logs:"
        cat gateway.log
        exit 1
    fi
    sleep 1
done

# Step 4: Run warmup requests
echo ""
echo "🔥 Running warmup requests..."
echo 'POST http://localhost:3000/inference' | \
vegeta attack \
    -header="Content-Type: application/json" \
    -header="Authorization: Bearer budserve_ApTuiKIpZEjMytHt3nmCJqBQGaIlqc2TfoKYFusk" \
    -body="$SCRIPT_DIR/test-request.json" \
    -duration="${WARMUP_REQUESTS}" \
    -rate=10 \
    -timeout="$TIMEOUT" \
    > /dev/null 2>&1

# Step 5: Run performance test
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

# Step 6: Parse and display results
echo ""
echo "📈 Performance Test Results"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Extract key metrics using jq
if command -v jq >/dev/null 2>&1; then
    LATENCY_MEAN=$(jq -r '.latencies.mean' "$OUTPUT_FILE" | awk '{print $1/1000000}')
    LATENCY_P50=$(jq -r '.latencies."50th"' "$OUTPUT_FILE" | awk '{print $1/1000000}')
    LATENCY_P95=$(jq -r '.latencies."95th"' "$OUTPUT_FILE" | awk '{print $1/1000000}')
    LATENCY_P99=$(jq -r '.latencies."99th"' "$OUTPUT_FILE" | awk '{print $1/1000000}')
    LATENCY_MAX=$(jq -r '.latencies.max' "$OUTPUT_FILE" | awk '{print $1/1000000}')
    SUCCESS_RATE=$(jq -r '.success' "$OUTPUT_FILE" | awk '{print $1*100}')
    THROUGHPUT=$(jq -r '.throughput' "$OUTPUT_FILE")
    
    printf "Latency (ms):\n"
    printf "  Mean:   %8.2f ms\n" "$LATENCY_MEAN"
    printf "  P50:    %8.2f ms\n" "$LATENCY_P50"
    printf "  P95:    %8.2f ms\n" "$LATENCY_P95"
    printf "  P99:    %8.2f ms\n" "$LATENCY_P99"
    printf "  Max:    %8.2f ms\n" "$LATENCY_MAX"
    printf "\n"
    printf "Success Rate: %.2f%%\n" "$SUCCESS_RATE"
    printf "Throughput:   %.2f req/s\n" "$THROUGHPUT"
    
    # Check for performance thresholds
    echo ""
    echo "🎯 Performance Checks"
    echo "━━━━━━━━━━━━━━━━━━━━"
    
    # Define thresholds
    P99_THRESHOLD=50.0  # 50ms
    SUCCESS_THRESHOLD=99.0  # 99%
    
    CHECKS_PASSED=true
    
    # Check P99 latency
    if (( $(echo "$LATENCY_P99 > $P99_THRESHOLD" | bc -l) )); then
        echo -e "${RED}❌ P99 latency (${LATENCY_P99}ms) exceeds threshold (${P99_THRESHOLD}ms)${NC}"
        CHECKS_PASSED=false
    else
        echo -e "${GREEN}✓ P99 latency (${LATENCY_P99}ms) within threshold (${P99_THRESHOLD}ms)${NC}"
    fi
    
    # Check success rate
    if (( $(echo "$SUCCESS_RATE < $SUCCESS_THRESHOLD" | bc -l) )); then
        echo -e "${RED}❌ Success rate (${SUCCESS_RATE}%) below threshold (${SUCCESS_THRESHOLD}%)${NC}"
        CHECKS_PASSED=false
    else
        echo -e "${GREEN}✓ Success rate (${SUCCESS_RATE}%) meets threshold (${SUCCESS_THRESHOLD}%)${NC}"
    fi
    
    if [ "$CHECKS_PASSED" = false ]; then
        echo ""
        echo -e "${RED}⚠️  Performance checks failed!${NC}"
        exit 1
    else
        echo ""
        echo -e "${GREEN}✅ All performance checks passed!${NC}"
    fi
else
    echo "⚠️  jq not found, showing raw results:"
    cat "$OUTPUT_FILE"
fi

echo ""
echo "📄 Full results saved to: $OUTPUT_FILE"