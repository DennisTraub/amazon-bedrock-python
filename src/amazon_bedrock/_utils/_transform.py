from __future__ import annotations

from typing import List, Iterable


class PropertyInfo:
    discriminator: str | None

    def __init__(self, *, discriminator: str | None) -> None:
        self.discriminator = discriminator


def transform(messages: Iterable) -> List:
    transformed_messages = []
    for message in messages:
        transformed_message = {
            "role": message["role"],
            "content": [{"text": _maybe_transform(message["content"])}]
        }
        transformed_messages.append(transformed_message)

    return transformed_messages


def _maybe_transform(message):
    if isinstance(message, str):
        return message
    elif _is_text_block(message):
        return message.text


def _is_text_block(message):
    return (
            hasattr(message, "text")
            and hasattr(message, "type")
            and isinstance(message.text, str)
            and message.type == "text"
    )
