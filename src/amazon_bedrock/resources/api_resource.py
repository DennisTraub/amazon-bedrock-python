from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..bedrock import Bedrock


class APIResource:
    _client: Bedrock

    def __init__(self, client: "Bedrock") -> None:
        self._client = client
