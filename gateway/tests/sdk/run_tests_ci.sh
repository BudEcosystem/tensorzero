#!/bin/bash

# CI-specific integration tests for TensorZero OpenAI provider
# Uses dummy provider and doesn't require real API keys

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}TensorZero CI Integration Tests${NC}"
echo "===================================="

# Check if virtual environment exists (support both .venv and venv)
if [ -d ".venv" ]; then
    VENV_DIR=".venv"
elif [ -d "venv" ]; then
    VENV_DIR="venv"
else
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv .venv
    VENV_DIR=".venv"
fi

# Activate virtual environment
source ${VENV_DIR}/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -q -r requirements.txt

# Set CI environment variables
export TENSORZERO_BASE_URL="http://localhost:3001"
export TENSORZERO_API_KEY="test-api-key"
export OPENAI_API_KEY="dummy-key"

# Check if TensorZero is running
echo -e "${YELLOW}Checking TensorZero gateway...${NC}"
if ! curl -s -f "${TENSORZERO_BASE_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: TensorZero gateway is not running!${NC}"
    echo "Please start TensorZero with: cargo run --bin gateway -- --config-file gateway/tests/sdk/test_config_ci.toml"
    exit 1
fi
echo -e "${GREEN}✓ TensorZero gateway is running${NC}"

# Run CI-specific tests
echo -e "\n${YELLOW}Running CI integration tests...${NC}\n"

# Function to run test module
run_test() {
    local module=$1
    local description=$2
    
    echo -e "${YELLOW}Testing ${description}...${NC}"
    if pytest -v "${module}" -x; then
        echo -e "${GREEN}✓ ${description} tests passed${NC}\n"
    else
        echo -e "${RED}✗ ${description} tests failed${NC}\n"
        exit 1
    fi
}

# Create test images if needed
echo -e "${YELLOW}Setting up test images...${NC}"
python create_test_images.py

# Run CI-specific test modules
run_test "test_ci_basic.py" "All Endpoints (CI)"
run_test "test_ci_chat.py" "Chat Completions (CI)"
run_test "test_ci_images.py" "Images (CI)"

echo -e "\n${GREEN}All CI integration tests passed!${NC}"

# Deactivate virtual environment
deactivate