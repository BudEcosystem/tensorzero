# TensorZero OpenAI Integration Tests

This directory contains integration tests that validate TensorZero's OpenAI-compatible endpoints using the official OpenAI Python SDK.

## Setup

1. **Install Dependencies**
   ```bash
   cd integration_tests
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env and set your OpenAI API key
   ```

3. **Start TensorZero Gateway**
   ```bash
   # From the repository root
   cargo run --bin gateway -- --config-file integration_tests/test_config.toml
   ```

## Running Tests

### Run All Tests
```bash
./run_tests.sh
```

### Run Specific Test Module
```bash
pytest test_chat.py -v
pytest test_embeddings.py -v
pytest test_moderation.py -v
pytest test_audio.py -v
```

### Run with Direct OpenAI Comparison
```bash
./run_tests.sh --compare
```

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

## Configuration

The `test_config.toml` file configures TensorZero with:
- All OpenAI model variants
- Authentication enabled (static API key: `test-api-key`)
- Debug mode for troubleshooting

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

## Adding New Tests

1. Create a new test file following the pattern `test_<endpoint>.py`
2. Import the configured clients:
   ```python
   from openai import OpenAI
   tensorzero_client = OpenAI(base_url=f"{TENSORZERO_BASE_URL}/v1", api_key=TENSORZERO_API_KEY)
   ```
3. Write test classes and methods
4. Update `run_tests.sh` if needed

## CI Integration

These tests can be integrated into CI/CD pipelines:
```yaml
- name: Start TensorZero
  run: cargo run --bin gateway -- --config-file integration_tests/test_config.toml &
  
- name: Run Integration Tests
  run: cd integration_tests && ./run_tests.sh
```