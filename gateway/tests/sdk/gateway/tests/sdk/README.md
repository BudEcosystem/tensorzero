# TensorZero SDK Testing Framework

This directory contains comprehensive tests for TensorZero's compatibility with various AI provider SDKs. The framework supports testing with both real API providers and dummy providers for CI environments.

## Quick Start

```bash
# Run all CI tests (uses dummy providers)
./run_tests.sh

# Run Anthropic tests specifically
./run_tests.sh --provider anthropic --mode ci

# Run with real API keys (requires .env file)
./run_tests.sh --provider anthropic --mode full
```

## Supported Providers

### ✅ Anthropic SDK (Native Support)

**Status**: Fully implemented with native `/v1/messages` endpoint support

The Anthropic SDK now works natively with TensorZero through the `/v1/messages` endpoint:

- **Native SDK Support**: Direct compatibility with the official Anthropic Python SDK
- **Full Feature Support**: Streaming, tool use, system prompts, multi-turn conversations  
- **Request/Response Conversion**: Automatic translation between Anthropic Messages API and TensorZero formats
- **Testing**: Comprehensive test suite with 10+ tests covering all major features

**Test Files**:
- `anthropic_tests/test_ci_messages.py` - CI tests (dummy provider)
- `anthropic_tests/test_native_messages.py` - Native SDK tests
- `anthropic_tests/test_messages.py` - Full integration tests
- `anthropic_tests/test_streaming.py` - Streaming functionality
- `anthropic_tests/test_openai_compat.py` - OpenAI compatibility tests (legacy)

### ✅ OpenAI SDK

**Status**: Fully supported via `/v1/chat/completions` endpoint

- Complete OpenAI SDK compatibility
- All endpoint types supported (chat, embeddings, moderation, audio, images)
- Comprehensive test coverage

**Test Files**:
- `openai_tests/test_ci_*.py` - CI tests
- `openai_tests/test_*.py` - Full integration tests

## Architecture

### Directory Structure

```
gateway/tests/sdk/
├── anthropic_tests/           # Anthropic SDK tests  
│   ├── test_ci_messages.py    # CI tests using /v1/messages
│   ├── test_native_messages.py # Native SDK test suite
│   ├── test_messages.py       # Full integration tests
│   ├── test_streaming.py      # Streaming tests
│   └── test_openai_compat.py  # Legacy OpenAI compatibility
├── openai_tests/              # OpenAI SDK tests
│   ├── test_ci_*.py           # CI tests
│   └── test_*.py              # Full integration tests
├── common/                    # Shared utilities
│   ├── base_test.py           # Abstract base classes
│   └── utils.py               # Test utilities
├── fixtures/                  # Test assets (images, audio)
├── test_config_*.toml        # Provider configurations
├── requirements.txt          # Python dependencies
├── run_tests.sh             # Universal test runner
└── README.md                # This file
```

### Endpoint Implementation

#### `/v1/messages` (Anthropic Messages API)

**Implementation**: `tensorzero-internal/src/endpoints/openai_compatible.rs::anthropic_messages_handler`

Features:
- **Request Format**: Native Anthropic Messages API format
- **Response Format**: Native Anthropic response format  
- **Bidirectional Conversion**: Anthropic ↔ TensorZero internal format
- **Streaming Support**: Server-Sent Events compatible with Anthropic SDK
- **Error Handling**: Anthropic-formatted error responses

**Key Components**:
- `AnthropicMessagesParams` - Request structure
- `anthropic_to_openai_request()` - Format conversion
- `openai_to_anthropic_response()` - Response conversion
- Stream handling for real-time responses

#### `/v1/chat/completions` (OpenAI Chat Completions API)

**Implementation**: `tensorzero-internal/src/endpoints/openai_compatible.rs::inference_handler`

Features:
- Full OpenAI Chat Completions API compatibility
- Works with both OpenAI and Anthropic models
- Cross-provider model support

## Configuration

### CI Tests (Dummy Provider)

```toml
# test_config_anthropic_ci.toml
[models."claude-3-haiku-20240307"]
routing = ["dummy"]
endpoints = ["chat"]

[models."claude-3-haiku-20240307".providers.dummy]
type = "dummy"
model_name = "test"
```

### Full Integration Tests

```toml
# test_config_anthropic.toml  
[models."claude-3-haiku-20240307"]
routing = ["anthropic"]
endpoints = ["chat"]

[models."claude-3-haiku-20240307".providers.anthropic]
type = "anthropic"
model_name = "claude-3-haiku-20240307"
api_key_location = "env::ANTHROPIC_API_KEY"
```

## Environment Setup

### For CI (Dummy Provider)

No API keys required. The test runner automatically sets dummy values.

### For Full Integration Tests

Create a `.env` file:

```bash
TENSORZERO_BASE_URL=http://localhost:3001
TENSORZERO_API_KEY=your-api-key
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

## Running Tests

### Using the Test Runner (Recommended)

```bash
# All providers, CI mode
./run_tests.sh

# Specific provider
./run_tests.sh --provider anthropic --mode ci

# Full integration with real API keys
./run_tests.sh --provider anthropic --mode full

# Custom port
./run_tests.sh --provider anthropic --port 3000
```

### Direct pytest

```bash
# Set environment
export TENSORZERO_BASE_URL=http://localhost:3001
source venv/bin/activate

# Run specific tests
pytest anthropic_tests/test_ci_messages.py -v
pytest anthropic_tests/test_native_messages.py -v

# Run all Anthropic tests
pytest anthropic_tests/ -v
```

## Gateway Setup

### For CI Tests

```bash
cargo run --bin gateway -- --config-file gateway/tests/sdk/test_config_anthropic_ci.toml
```

### For Full Tests

```bash
cargo run --bin gateway -- --config-file gateway/tests/sdk/test_config_anthropic.toml
```

## Test Types

### CI Tests (`test_ci_*.py`)

- Use dummy providers for predictable responses
- No real API keys required
- Fast execution for continuous integration
- Test SDK compatibility and basic functionality

### Integration Tests (`test_*.py`)

- Use real API providers
- Require valid API keys
- Test full end-to-end functionality
- Validate actual model responses

### Native Tests (`test_native_*.py`)

- Test native SDK functionality with `/v1/messages`
- Verify endpoint-specific features
- Confirm proper request/response handling

## Key Features Tested

### Anthropic SDK Tests

- ✅ Basic message creation via `/v1/messages`
- ✅ System prompts
- ✅ Multi-turn conversations  
- ✅ Streaming responses
- ✅ Tool/function calling
- ✅ Async client support
- ✅ Temperature and parameter control
- ✅ Multiple model support
- ✅ Error handling
- ✅ Stop sequences

### Cross-Provider Features

- ✅ Request/response format conversion
- ✅ Model routing and fallback
- ✅ Authentication and API key handling
- ✅ Rate limiting and error responses
- ✅ Usage tracking and metrics

## Troubleshooting

### Gateway Not Running

```bash
# Check if gateway is running
curl http://localhost:3001/health

# Start gateway with correct config
cargo run --bin gateway -- --config-file gateway/tests/sdk/test_config_anthropic_ci.toml
```

### Import Errors

```bash
# Ensure virtual environment is active
source venv/bin/activate  # or .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

```bash
# Check current environment
echo $TENSORZERO_BASE_URL
echo $TENSORZERO_API_KEY

# Set manually if needed
export TENSORZERO_BASE_URL=http://localhost:3001
export TENSORZERO_API_KEY=test-api-key
```

## Development

### Adding New Provider Tests

1. Create provider directory: `provider_tests/`
2. Add SDK to `requirements.txt`
3. Create configuration files
4. Implement test classes inheriting from base classes
5. Add provider to test runner script

### Adding New Test Cases

1. Choose appropriate base class (`BaseChatTest`, `BaseStreamingTest`, etc.)
2. Implement provider-specific methods
3. Add both CI and integration test variants
4. Update documentation

## Contributing

When adding tests:

1. Follow existing patterns and structure
2. Test both CI (dummy) and full (real API) modes
3. Include comprehensive error handling tests
4. Update documentation for new features
5. Ensure tests pass in both environments

## Reference

- [GitHub Issue #39](https://github.com/tensorzero/tensorzero/issues/39) - SDK Testing Framework Implementation
- [TensorZero Architecture](../../../CLAUDE.md) - Main codebase documentation
- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) - Official Anthropic Python SDK
- [OpenAI SDK](https://github.com/openai/openai-python) - Official OpenAI Python SDK