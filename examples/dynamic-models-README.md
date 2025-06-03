# Dynamic Models Integration

This example demonstrates how to integrate TensorZero with an external model management service to dynamically load and use models that aren't statically defined in the configuration file.

## Overview

The dynamic models feature allows TensorZero to:
- Query an external API for available models at different endpoints
- Cache model configurations per endpoint with automatic refresh
- Use dynamically loaded models seamlessly alongside static ones
- Support model providers like OpenAI, Anthropic, Azure, etc.

## Configuration

Enable dynamic models by adding a `[dynamic_model_service]` section to your `tensorzero.toml`:

```toml
[dynamic_model_service]
service_url = "https://your-model-service.com/api/models"
api_key = "your-api-key"
# Optional default endpoint name
# endpoint_name = "production"
refresh_interval_secs = 60
request_timeout_secs = 10
```

## Model Naming Convention

When using dynamic models, the model name is treated as an endpoint name to query your service:

1. **Simple endpoint name**: The model name is used as the endpoint parameter
   ```toml
   model = "test"  # Queries ?endpoint_name=test and uses the returned model
   ```

2. **Endpoint with specific model** (if multiple models per endpoint):
   ```toml
   model = "production::gpt-4-specific"  # Uses endpoint "production" and model "gpt-4-specific"
   ```

In most cases, you'll just use the endpoint name, and the system will automatically use the model configuration returned by your service.

## API Response Format

Your model service should return a JSON response in this format when queried with `?endpoint_name=<endpoint>`:

```json
{
  "success": true,
  "result": {
    "project_id": "...",
    "project_name": "...",
    "endpoint_name": "production",
    "model_configuration": [
      {
        "model_name": "custom-gpt-4",
        "litellm_params": {
          "model": "openai/gpt-4",
          "api_base": "https://api.openai.com/v1",
          "api_key": "sk-..."
        },
        "model_info": {
          "metadata": {
            "name": "GPT-4",
            "provider": "OpenAI"
          }
        }
      }
    ]
  }
}
```

## Usage Examples

### Basic usage - endpoint as model name:

```toml
[functions.chat.variants.test]
type = "chat_completion"
model = "test"  # Queries endpoint "test" and uses the returned model

[functions.chat.variants.prod]
type = "chat_completion"
model = "production"  # Queries endpoint "production" and uses the returned model
```

### Advanced usage - specific model selection:

```toml
# If your endpoint returns multiple models, you can specify which one to use
[functions.chat.variants.specific]
type = "chat_completion"
model = "production::gpt-4-turbo"  # Uses "production" endpoint but specifically selects "gpt-4-turbo"
```

## Caching Behavior

- Each endpoint maintains its own cache of models
- Caches refresh independently based on `refresh_interval_secs`
- First request to an endpoint triggers a fetch from the service
- Subsequent requests use cached data until refresh interval expires

This allows you to efficiently work with multiple environments (production, staging, development) while minimizing API calls to your model service.