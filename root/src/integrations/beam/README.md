# Beam Cloud Deployment

This directory contains the integration code for running models on Beam Cloud infrastructure.

## Deployment Instructions

### Prerequisites

1. Install the Beam SDK:
```bash
pip install beam-client
```

2. Configure your Beam API token:
```bash
beam configure default --token YOUR_BEAM_API_TOKEN
```

### Deploying Endpoints

To deploy the chat endpoint:

```bash
beam deploy app.py:chat_endpoint
```

This will create a REST API endpoint that you can call with:

```bash
curl -X POST 'https://app.beam.cloud/endpoint/chat' \
  -H 'Authorization: Bearer YOUR_BEAM_API_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello, world!"}]}'
```

### Deploying Task Queues

To deploy the inference task queue:

```bash
beam deploy app.py:inference_task
```

You can then submit tasks to this queue:

```bash
beam invoke inference_task '{"prompt": "Explain quantum computing"}'
```

## File Structure

- `__init__.py` - Package exports
- `runtime.py` - Beam runtime wrappers and utilities
- `ai_model.py` - Model loading and inference
- `app.py` - Deployment entry points

## Environment Variables

The deployment uses the following environment variables:

- `BEAM_API_TOKEN` - Your Beam API token
- `BEAM_VOLUME` - Name of the Beam volume containing model weights
- `BEAM_MODEL_ID` - Default model ID to use
- `BEAM_GPU` - GPU type (e.g., "A10G")
- `BEAM_MEMORY` - Memory allocation (e.g., "32Gi")
- `BEAM_CPU` - CPU cores (e.g., 4)

## Customizing Deployments

You can customize the deployment by modifying `app.py`. For example, to change the dependencies:

```python
def get_image():
    return Image(
        python_version="python3.10",
        python_packages=[
            "torch==2.1.0",
            # Add your dependencies here
        ],
    )
```

Or to add more endpoints/tasks:

```python
@endpoint(
    name="my_custom_endpoint",
    # ...configuration...
)
async def my_custom_endpoint(**inputs):
    # Your code here
    return {"result": "success"}
``` 