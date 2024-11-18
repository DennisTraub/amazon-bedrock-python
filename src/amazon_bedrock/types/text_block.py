from typing import Literal

from pydantic import BaseModel


class TextBlock(BaseModel):
    text: str
    type: Literal["text"]
