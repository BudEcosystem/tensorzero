# Together AI SDK Tests

This directory contains tests demonstrating Together AI models working through TensorZero's universal OpenAI SDK compatibility layer.

## Overview

These tests showcase how Together AI's diverse model portfolio (Llama, Qwen, Mistral, DeepSeek, etc.) works seamlessly with the OpenAI SDK through TensorZero, without requiring the Together AI Python SDK.

## Test Files

- `test_universal_openai_sdk.py` - Full integration tests with real Together AI API
- `test_ci_together.py` - CI-friendly tests using dummy providers (no API key required)
- `demonstrate_together_openai_sdk.py` - Interactive demonstration script

## Running Tests

### CI Tests (No API Key Required)

```bash
# From the sdk directory
cd gateway/tests/sdk
./run_tests.sh --provider together --mode ci
```

### Full Integration Tests

Requires `TOGETHER_API_KEY` environment variable:

```bash
# Set your API key
export TOGETHER_API_KEY="your-together-api-key"

# Run full tests
./run_tests.sh --provider together --mode full
```

### Using the Together-specific runner

```bash
# Run all Together AI tests
./run_tests_together.sh
```

## Test Configuration

Tests use two configuration files:

1. `test_config_together.toml` - Real Together AI provider configuration
2. `test_config_together_ci.toml` - Dummy provider for CI testing

## Supported Models

The tests cover popular Together AI models including:

- **Llama Models**: Llama 3.3, 3.2, 3.1 (various sizes)
- **Qwen Models**: Qwen 2.5 72B
- **Mistral Models**: Mixtral 8x7B
- **DeepSeek Models**: DeepSeek v2.5

## Key Features Tested

1. **Basic Chat Completions** - Standard message generation
2. **Streaming Responses** - Real-time token streaming
3. **System Prompts** - Model instruction capabilities
4. **Temperature Control** - Output randomness control
5. **Multi-turn Conversations** - Context preservation
6. **Cross-provider Compatibility** - Uniform behavior across providers

## Architecture Benefits

These tests demonstrate that:

- ✅ Together AI models work perfectly with OpenAI SDK
- ✅ No Together AI SDK required - just OpenAI SDK
- ✅ Same code works across OpenAI, Anthropic, and Together AI
- ✅ Full feature parity with native implementations

## Example Usage

```python
from openai import OpenAI

# One client for all providers
client = OpenAI(
    base_url="http://localhost:3001/v1",
    api_key="your-tensorzero-key"
)

# Works with Together AI models
response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Adding New Together AI Models

To test a new Together AI model:

1. Add it to `test_config_together.toml` and `test_config_together_ci.toml`
2. Include it in the model lists in test files
3. Ensure it's configured with `endpoints = ["chat"]`

## Troubleshooting

- **Gateway not running**: Start with `cargo run --bin gateway -- --config-file gateway/tests/sdk/test_config_together.toml`
- **API key errors**: Ensure `TOGETHER_API_KEY` is set in your environment
- **Model not found**: Check the model name matches Together AI's exact naming