# Model Router

A smart LLM routing system that automatically selects the most cost-effective model based on task complexity and requirements.

## Features

- **Intelligent Model Selection**: Routes tasks to the most appropriate model based on complexity
- **Cost Optimization**: Tracks and optimizes costs across different model providers
- **Performance Monitoring**: Monitors latency and request counts
- **Multi-Provider Support**: Supports OpenAI, Anthropic, and Mistral models

## Task Types

The router supports four main task types, each optimized for different use cases:

1. **Classification** (Mistral Small) - Lightweight tasks like spam detection
2. **Summarization** (Claude 3 Haiku) - Text summarization and extraction
3. **Content Generation** (Claude 3.5 Sonnet) - Creative writing and content creation
4. **Complex Reasoning** (GPT-4o) - Advanced reasoning and analysis

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your API keys as environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export MISTRAL_API_KEY="your-mistral-api-key"
```

## Usage

### Basic Example

```python
from model_router import ModelManager, TaskType
import os

# Initialize with API keys
api_keys = {
    "openai": os.environ.get("OPENAI_API_KEY"),
    "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
    "mistral": os.environ.get("MISTRAL_API_KEY")
}

model_manager = ModelManager(api_keys)

# Classification task
result = model_manager.generate(
    TaskType.CLASSIFICATION,
    "Classify this email as spam or not spam: 'Congratulations! You've won $1,000,000!'"
)

print(f"Result: {result['result']}")
print(f"Model: {result['model']}")
print(f"Latency: {result['latency_ms']}ms")
print(f"Cost: ${result['estimated_cost']:.6f}")
```

### Cost Tracking

```python
# Get cost summary
summary = model_manager.get_cost_summary()
print(f"Total cost: ${summary['total_cost']:.6f}")
print(f"Total requests: {summary['total_requests']}")

# Cost by task type
for task_type, cost in summary['cost_by_task'].items():
    requests = summary['requests_by_task'].get(task_type, 0)
    print(f"{task_type}: ${cost:.6f} ({requests} requests)")
```

## Model Configuration

Each task type is configured with optimal parameters:

| Task Type | Model | Max Tokens | Temperature | Input Cost/1M | Output Cost/1M |
|-----------|-------|------------|-------------|---------------|----------------|
| Classification | Mistral Small | 100 | 0.1 | $2.00 | $6.00 |
| Summarization | Claude 3 Haiku | 500 | 0.3 | $0.25 | $1.25 |
| Content Generation | Claude 3.5 Sonnet | 2048 | 0.7 | $3.00 | $15.00 |
| Complex Reasoning | GPT-4o | 4096 | 0.7 | $5.00 | $15.00 |

## Error Handling

The router includes comprehensive error handling:

- Missing API keys are detected during initialization
- Invalid task types raise descriptive errors
- API failures are logged and re-raised
- Graceful degradation when specific models are unavailable

## Production Considerations

- Use a secure secret manager for API keys in production
- Implement proper token counting for accurate cost tracking
- Add retry logic for transient API failures
- Consider implementing circuit breakers for model availability
- Monitor and log all requests for debugging and optimization

## Running the Example

```bash
python model_router.py
```

This will run the example tasks and display cost summaries, assuming you have the required API keys set up. 