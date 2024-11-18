from typing import Iterable

from . import APIResource
from .._utils import transform
from ..types import Message, MessageParam


class Messages(APIResource):
    def create(self,
        messages: Iterable[MessageParam],
        model: str
    )-> Message:
        transformed_messages = transform(messages)

        api_response = self._client.bedrock_runtime_client.converse(
            modelId=model,
            messages=transformed_messages
        )
        text = api_response["output"]["message"]["content"][0]["text"]

        return Message(
            role="assistant",
            content=[{"type": "text", "text": text}]
        )
