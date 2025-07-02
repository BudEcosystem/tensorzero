#!/bin/bash

# Legacy script - redirects to new unified test runner
# For OpenAI full integration tests

echo "Note: This script is deprecated. Using new unified test runner..."
echo ""

# Pass through to new test runner with OpenAI provider in full mode
./run_tests.sh --provider openai --mode full "$@"