from typing import Union, Iterable
from typing_extensions import TypedDict, Required, Literal

__all__ = ["MessageParam"]

from .text_block_param import TextBlockParam


class MessageParam(TypedDict, total=False):
    role: Required[Literal["user", "assistant"]]
    content: Required[
        Union[
            str, Iterable[Union[TextBlockParam]]
        ]
    ]
