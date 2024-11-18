import pytest
from unittest.mock import Mock

from amazon_bedrock import Bedrock
from amazon_bedrock.resources.api_resource import APIResource


class TestAPIResource:
    @pytest.fixture
    def mock_bedrock(self):
        return Mock(spec=Bedrock)

    def test_init_stores_client_reference(self, mock_bedrock):
        resource = APIResource(mock_bedrock)
        assert resource._client == mock_bedrock
