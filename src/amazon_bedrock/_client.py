import boto3

from .resources import Messages, Models

__all__ = ["Bedrock"]


class Bedrock:
    models: Models
    messages: Messages

    def __init__(self, region: str = None) -> None:
        kwargs = {"region_name": region} if region else {}
        self.bedrock_client = boto3.client("bedrock", **kwargs)
        self.bedrock_runtime_client = boto3.client("bedrock-runtime", **kwargs)
        self.region = self.bedrock_client.meta.region_name
        self.models = Models(self)
        self.messages = Messages(self)
