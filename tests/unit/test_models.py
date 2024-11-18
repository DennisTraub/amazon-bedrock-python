import pytest
from unittest.mock import Mock

from src.amazon_bedrock.resources import Models


class TestModels:
    MOCK_MODELS = [
        {"modelId": "anthropic.claude-v1"},
        {"modelId": "anthropic.claude-v2"}
    ]

    @pytest.fixture
    def mock_boto3_client(self) -> Mock:
        client = Mock()
        client.list_foundation_models.return_value = {"modelSummaries": self.MOCK_MODELS}
        return client

    @pytest.fixture
    def models_resource(self, mock_boto3_client) -> Models:
        models = Models(Mock())
        models._client.bedrock_client = mock_boto3_client
        return models

    def test_list_models(self, models_resource, mock_boto3_client) -> None:
        result = models_resource.list()

        mock_boto3_client.list_foundation_models.assert_called_once_with(
            byInferenceType="ON_DEMAND"
        )

        assert result == self.MOCK_MODELS

    def test_retrieve_unique_model(self, models_resource) -> None:
        result = models_resource.retrieve("claude-v1")
        assert result == {"modelId": "anthropic.claude-v1"}

    def test_retrieve_latest_model(self, models_resource) -> None:
        result = models_resource.retrieve("claude")
        assert result == {"modelId": "anthropic.claude-v2"}

    def test_retrieve_model_not_found(self, models_resource) -> None:
        with pytest.raises(ValueError, match="Model 'nonexistent-model' cannot be found."):
            models_resource.retrieve("nonexistent-model")

    def test_retrieve_model_empty_input(self, models_resource) -> None:
        with pytest.raises(ValueError, match="Expected a non-empty value for `model`"):
            models_resource.retrieve("")
