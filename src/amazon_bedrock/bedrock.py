import boto3

from .resources import Models


class Bedrock:
    models: Models

    def __init__(self, region: str = None) -> None:
        kwargs = {"region_name": region} if region else {}
        self.boto3_client = boto3.client("bedrock", **kwargs)
        self.region = self.boto3_client.meta.region_name
        self.models = Models(self)
