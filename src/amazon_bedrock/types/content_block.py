from typing_extensions import Annotated, TypeAlias

from .text_block import TextBlock
from .._utils import PropertyInfo

ContentBlock: TypeAlias = Annotated[TextBlock, PropertyInfo(discriminator="type")]
