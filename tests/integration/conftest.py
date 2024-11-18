import pytest

from src.amazon_bedrock import Bedrock

TEST_REGION = "us-west-2"

@pytest.fixture
def bedrock_client() -> Bedrock:
    return Bedrock(region=TEST_REGION)
