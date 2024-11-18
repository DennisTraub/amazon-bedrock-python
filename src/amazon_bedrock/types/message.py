import pydantic
from typing import List, Literal

from .content_block import ContentBlock


class Message(pydantic.BaseModel):
    content: List[ContentBlock]
    role: Literal["assistant"]
