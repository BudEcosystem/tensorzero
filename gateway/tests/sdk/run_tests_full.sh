#!/bin/bash

# Full integration tests for TensorZero OpenAI provider
# Requires real OpenAI API keys and runs against actual providers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}TensorZero Full Integration Tests${NC}"
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

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    echo "Please copy .env.example to .env and set your API keys"
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check if TensorZero is running
echo -e "${YELLOW}Checking TensorZero gateway...${NC}"
if ! curl -s -f "${TENSORZERO_BASE_URL:-http://localhost:3000}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: TensorZero gateway is not running!${NC}"
    echo "Please start TensorZero with: cargo run --bin gateway -- --config-file gateway/tests/sdk/test_config.toml"
    exit 1
fi
echo -e "${GREEN}✓ TensorZero gateway is running${NC}"

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY is not set!${NC}"
    exit 1
fi

# Run tests
echo -e "\n${YELLOW}Running full integration tests...${NC}\n"

# Function to run test module
run_test() {
    local module=$1
    local description=$2
    
    echo -e "${YELLOW}Testing ${description}...${NC}"
    if pytest -v "test_${module}.py" -k "not compare_with_openai"; then
        echo -e "${GREEN}✓ ${description} tests passed${NC}\n"
    else
        echo -e "${RED}✗ ${description} tests failed${NC}\n"
        exit 1
    fi
}

# Create test images if needed
echo -e "${YELLOW}Setting up test images...${NC}"
python create_test_images.py

# Run each test module
run_test "chat" "Chat Completions"
run_test "embeddings" "Embeddings"
run_test "moderation" "Moderation"
run_test "audio" "Audio (Transcription, Translation, TTS)"
run_test "images" "Images (Generation, Editing, Variations)"
run_test "realtime" "Realtime API (Session Management)"

# Run comparison tests (optional)
if [ "$1" == "--compare" ]; then
    echo -e "\n${YELLOW}Running comparison tests with direct OpenAI API...${NC}"
    pytest -v -k "compare_with_openai"
fi

echo -e "\n${GREEN}All full integration tests passed!${NC}"

# Deactivate virtual environment
deactivate