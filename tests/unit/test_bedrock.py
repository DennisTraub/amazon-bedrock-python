import pytest
from typing import Generator, Tuple
from unittest.mock import Mock, patch, call

from src.amazon_bedrock import Bedrock


class TestBedrock:
    REGION = "test-region"

    @pytest.fixture
    def boto_client_fixtures(self) -> Generator[Tuple[Mock, Mock], None, None]:
        with patch("boto3.client") as mock_client:
            mock_boto3_client = Mock()
            mock_client.return_value = mock_boto3_client
            yield mock_client, mock_boto3_client

    def test_init_initializes_resources(self) -> None:
        client = Bedrock()
        assert client.bedrock_client is not None
        assert client.models is not None
        assert client.messages is not None

    def test_init_sets_region(self, boto_client_fixtures) -> None:
        _, boto_client_mock = boto_client_fixtures
        boto_client_mock.meta.region_name = self.REGION

        client = Bedrock()

        assert client.region == self.REGION

    def test_no_region_creates_default_clients(self, boto_client_fixtures) -> None:
        boto_client_spy, _ = boto_client_fixtures

        Bedrock()

        expected_config = [
            call("bedrock"),
            call("bedrock-runtime")
        ]
        boto_client_spy.assert_has_calls(expected_config, any_order=True)
        assert boto_client_spy.call_count == len(expected_config)

    def test_region_parameter_configures_clients(self, boto_client_fixtures) -> None:
        boto_client_spy, boto_client_mock = boto_client_fixtures
        boto_client_mock.meta.region_name = self.REGION

        Bedrock(region=self.REGION)

        expected_config = [
            call("bedrock", region_name=self.REGION),
            call("bedrock-runtime", region_name=self.REGION)
        ]
        boto_client_spy.assert_has_calls(expected_config, any_order=True)
        assert boto_client_spy.call_count == len(expected_config)
