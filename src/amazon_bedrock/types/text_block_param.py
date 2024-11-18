from typing import Required

from typing_extensions import TypedDict, Literal

__all__ = ["TextBlockParam"]


class TextBlockParam(TypedDict, total=False):
    text: Required[str]
    type: Required[Literal["text"]]
