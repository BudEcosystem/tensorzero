# TensorZero SDK Integration Tests

This directory contains integration tests that validate TensorZero's compatibility with various AI provider SDKs including OpenAI, Anthropic, and more.

## Quick Start

```bash
# Run all CI tests (no API keys required)
./run_tests.sh

# Run OpenAI tests only
./run_tests.sh --provider openai

# Run Anthropic tests with real API
./run_tests.sh --provider anthropic --mode full

# Run Universal SDK Compatibility tests
./run_tests.sh --provider universal

# Run interactive architecture demonstration
./run_tests.sh --demo

# See all options
./run_tests.sh --help
```

## Setup

1. **Install Dependencies**
   ```bash
   cd gateway/tests/sdk
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env and set your API keys for full tests
   ```

3. **Start TensorZero Gateway**
   ```bash
   # For CI tests (dummy provider)
   cargo run --bin gateway -- --config-file gateway/tests/sdk/test_config_[provider]_ci.toml
   
   # For full tests (real provider)
   cargo run --bin gateway -- --config-file gateway/tests/sdk/test_config_[provider].toml
   ```

## Running Tests

### Universal Test Runner

The new `run_tests.sh` script provides a unified interface for all SDK tests:

```bash
# Run all providers in CI mode (default)
./run_tests.sh

# Run specific provider
./run_tests.sh --provider openai
./run_tests.sh --provider anthropic

# Run Universal SDK Compatibility tests (NEW!)
./run_tests.sh --provider universal

# Run interactive architecture demonstration (NEW!)
./run_tests.sh --demo

# Run full integration tests (requires API keys)
./run_tests.sh --provider openai --mode full
./run_tests.sh --provider anthropic --mode full

# Run with comparison tests
./run_tests.sh --provider openai --mode full --compare

# Custom port
./run_tests.sh --port 3002
```

**Note**: If the gateway is running with `test_config_unified_ci.toml`, individual provider tests will still work since the unified config includes all models from both providers. The script will show the expected config file but won't fail if a compatible config is already running.

### Universal SDK Compatibility Tests

The universal tests demonstrate a key architectural principle: **OpenAI SDK works with ALL providers**:

```bash
# Test OpenAI SDK universal compatibility
./run_tests.sh --provider universal

# This runs:
# - openai_tests/test_all_providers.py (OpenAI SDK with all providers)
# - anthropic_tests/test_native_messages.py (Native SDK comparison)
```

### Architecture Demonstration

See the architecture in action:

```bash
# Run interactive demonstration
./run_tests.sh --demo

# Or run directly
python demonstrate_universal_sdk.py
```

### Legacy Scripts (Deprecated)

The following scripts still work but redirect to the new runner:
- `./run_tests_full.sh` → `./run_tests.sh --provider openai --mode full`
- `./run_tests_ci.sh` → `./run_tests.sh --provider openai --mode ci`

### Provider-Specific Shortcuts

```bash
# Anthropic tests
./run_tests_anthropic.sh           # CI mode
./run_tests_anthropic.sh --mode full  # Full mode
```

### Run Specific Test Module

```bash
# CI tests (for dummy provider)
pytest openai_tests/test_ci_basic.py -v
pytest openai_tests/test_ci_chat.py -v
pytest anthropic_tests/test_ci_messages.py -v

# Full tests (requires real API keys)
pytest openai_tests/test_chat.py -v
pytest openai_tests/test_embeddings.py -v
pytest anthropic_tests/test_messages.py -v
pytest anthropic_tests/test_streaming.py -v
```

**Important**: When running with the dummy provider (CI mode), use `test_ci_*.py` files. The regular test files expect real API responses and will fail with dummy provider.

## Supported Providers

### OpenAI
- Chat Completions
- Embeddings
- Moderation
- Audio (Transcription, Translation, TTS)
- Images (Generation, Editing, Variations)
- Realtime API

### Anthropic
- **Native Messages API** (`/v1/messages`) - Full native Anthropic SDK support
- **Universal OpenAI SDK** - Anthropic models also work via OpenAI SDK through `/v1/chat/completions`
- Streaming - Both native and OpenAI SDK
- Tool Use / Function Calling - Both native and OpenAI SDK
- Content Types - Both native and OpenAI SDK
- Error Handling - Both native and OpenAI SDK

### Universal SDK Architecture

TensorZero implements a unique architecture where:
- **OpenAI SDK is universal**: Works with ALL providers through `/v1/chat/completions`
- **Native SDKs are specialized**: Work with their provider-specific endpoints

This means you can use the OpenAI SDK with Anthropic models, Google models, or any other provider!

### Coming Soon
- Google (Gemini)
- Cohere
- AWS Bedrock
- Azure OpenAI

## Test Coverage

### Chat Completions (`test_chat.py`)
- Basic completions
- Streaming responses
- Function calling
- Multiple models
- Various parameters (temperature, max_tokens, etc.)
- Error handling
- Async operations

### Embeddings (`test_embeddings.py`)
- Single and batch embeddings
- Different models (ada-002, text-embedding-3-small)
- Custom dimensions
- Large batches
- Special characters
- Similarity testing
- Async operations

### Moderation (`test_moderation.py`)
- Content moderation
- Batch moderation
- Category scores
- Unicode and special characters
- Large batches
- Async operations

### Audio (`test_audio.py`)
- **Transcription**: Convert audio to text
- **Translation**: Convert non-English audio to English text
- **Text-to-Speech**: Generate audio from text
- Various voices and formats
- Response format options
- Async operations

### Images (`test_images.py`)
- **Generation**: Create images from text prompts (DALL-E 2, DALL-E 3, GPT-Image-1)
- **Editing**: Modify existing images with prompts and masks
- **Variations**: Generate variations of existing images
- Multiple image sizes and quality settings
- URL and base64 response formats
- Model-specific parameters (style, quality, background)
- Async operations
- Performance and stress testing

## Configuration

### Configuration Files

- `test_config_openai.toml` - OpenAI models with real API
- `test_config_openai_ci.toml` - OpenAI models with dummy provider
- `test_config_anthropic.toml` - Anthropic models with real API
- `test_config_anthropic_ci.toml` - Anthropic models with dummy provider
- `test_config_unified_ci.toml` - **NEW!** Unified config for universal SDK tests

All configurations use:
- Port 3001 (configurable)
- Authentication disabled for testing
- Debug mode enabled

For universal SDK tests, use the unified configuration:
```bash
cargo run --bin gateway -- --config-file gateway/tests/sdk/test_config_unified_ci.toml
```

## Troubleshooting

### Gateway Not Running
```
Error: TensorZero gateway is not running!
```
Solution: Start the gateway with the test configuration file.

### Missing API Key
```
Error: OPENAI_API_KEY is not set!
```
Solution: Set your OpenAI API key in the `.env` file.

### Audio Tests Failing
The audio tests require a sample MP3 file. The test runner automatically downloads a sample, but if this fails:
```bash
cd fixtures/audio_samples
# Download any small MP3 file and name it sample.mp3
```

### Image Tests Failing
The image tests require test images which are automatically generated by `create_test_images.py`. If this fails:
```bash
cd fixtures/images
# The script creates various test images including:
# - test_image.png (512x512 test image)
# - simple_shape.png (simple shape for variations)
# - mask.png (mask for editing tests)
```

## Adding New Provider Tests

See [SDK_TESTING.md](SDK_TESTING.md) for a detailed guide on adding tests for new providers.

Quick overview:
1. Create provider directory: `mkdir provider_name_tests` (note the `_tests` suffix)
2. Add SDK to `requirements.txt`
3. Create test configuration files
4. Implement test classes inheriting from base classes
5. Add CI tests for dummy provider
6. Update documentation

## CI Integration

### For GitHub Actions (No API Keys Required)

```yaml
# Test all providers
- name: Start TensorZero (OpenAI CI)
  run: cargo run --bin gateway -- --config-file gateway/tests/sdk/test_config_openai_ci.toml &
  
- name: Run OpenAI CI Tests
  run: cd gateway/tests/sdk && ./run_tests.sh --provider openai --mode ci

- name: Restart TensorZero (Anthropic CI)
  run: |
    pkill -f "cargo run --bin gateway" || true
    cargo run --bin gateway -- --config-file gateway/tests/sdk/test_config_anthropic_ci.toml &
    
- name: Run Anthropic CI Tests  
  run: cd gateway/tests/sdk && ./run_tests.sh --provider anthropic --mode ci
```

### For Full Testing (Requires API Keys)

```yaml
- name: Start TensorZero
  run: cargo run --bin gateway -- --config-file gateway/tests/sdk/test_config_${{ matrix.provider }}.toml &
  
- name: Run Full Integration Tests
  run: cd gateway/tests/sdk && ./run_tests.sh --provider ${{ matrix.provider }} --mode full
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Directory Structure

```
/gateway/tests/sdk/
├── common/                    # Shared utilities and base classes
│   ├── base_test.py          # Abstract base test classes
│   └── utils.py              # Common test utilities
├── openai_tests/             # OpenAI SDK tests
│   ├── test_*.py            # Full integration tests
│   ├── test_ci_*.py         # CI tests with dummy provider
│   ├── test_all_providers.py # Universal SDK compatibility tests
│   └── test_universal_sdk_demo.py # Focused demo tests
├── anthropic_tests/          # Anthropic SDK tests
│   ├── test_*.py            # Full integration tests
│   ├── test_ci_*.py         # CI tests with dummy provider
│   └── test_native_messages.py # Native Anthropic SDK tests
├── fixtures/                 # Test data
├── test_config_*.toml       # Provider configurations
├── test_config_unified_ci.toml # Unified config for universal tests
├── demonstrate_universal_sdk.py # Architecture demonstration script
├── run_tests.sh             # Universal test runner
├── requirements.txt         # Python dependencies
├── SDK_TESTING.md          # Guide for adding new providers
└── SDK_ARCHITECTURE.md     # Universal SDK architecture guide
```