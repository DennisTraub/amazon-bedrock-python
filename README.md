# Amazon Bedrock for Python

A Python library for easy interaction with Amazon Bedrock, providing streamlined access to foundation models and AI capabilities.

## Features

- Simple interface for the Amazon Bedrock API
- Support for multiple foundation models
- Easy text generation and completion

## Installation

Install the package using pip:

```shell
pip install amazon-bedrock
```

## Quick Start

```python
from amazon_bedrock import Bedrock

client = Bedrock(region="us-west-2")

models = client.models.list()
model_ids = [model["modelId"] for model in models]

print(model_ids)

model_id = "claude-3-5-haiku"
model = client.models.retrieve(model_id)

print(model)
```

## Authentication

The client uses standard AWS credentials. You can configure these in several ways:

1. Environment variables:

```shell
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION="us-west-2"
```

2. AWS credentials file (`~/.aws/credentials`)

3. IAM role when running on AWS services

## Requirements

- Python 3.8+
- boto3
- An AWS account with Bedrock access

## License

This project is licensed under the MIT License - see the LICENSE file for details.