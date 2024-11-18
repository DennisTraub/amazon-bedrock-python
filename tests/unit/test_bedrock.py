import pytest
from unittest.mock import Mock, patch

from amazon_bedrock import Bedrock


class TestClient:

    @pytest.fixture
    def mock(self):
        with patch("boto3.client") as mock_client:
            mock_boto3_client = Mock()
            mock_client.return_value = mock_boto3_client
            yield mock_client, mock_boto3_client

    def test_init_initializes_resources(self):
        client = Bedrock()
        assert client.boto3_client is not None
        assert client.models is not None

    def test_init_without_region_uses_default(self, mock):
        default_region = "default-region"
        mock_client, mock_boto3_client = mock
        mock_boto3_client.meta.region_name = default_region

        client = Bedrock()

        mock_client.assert_called_once_with("bedrock")
        assert client.region == default_region

    def test_init_with_region_uses_specified_region(self, mock):
        test_region = "test-region"
        mock_client, mock_boto3_client = mock
        mock_boto3_client.meta.region_name = test_region

        client = Bedrock(region=test_region)

        mock_client.assert_called_once_with("bedrock", region_name=test_region)
        assert client.region == test_region
