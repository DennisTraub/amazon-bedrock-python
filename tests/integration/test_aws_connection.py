import pytest
from typing import Dict


@pytest.mark.integration
class TestListModels:
    def test_aws_connection(self, bedrock_client) -> None:
        try:
            response = bedrock_client.bedrock_client.list_foundation_models()
            assert isinstance(response, Dict)
        except Exception as e:
            pytest.fail(f"Failed to connect to AWS Bedrock: {str(e)}")
