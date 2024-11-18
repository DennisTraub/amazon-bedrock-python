import pytest
from unittest.mock import Mock

from src.amazon_bedrock import Bedrock
from src.amazon_bedrock.resources import APIResource


class TestAPIResource:
    @pytest.fixture
    def mock_client(self) -> Mock:
        return Mock(spec=Bedrock)

    def test_init_sets_client_reference(self, mock_client) -> None:
        resource = APIResource(mock_client)
        assert resource._client == mock_client
