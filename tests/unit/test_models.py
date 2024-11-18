import pytest
from unittest.mock import Mock

from amazon_bedrock.resources import Models

MODELS = [
    {"modelId": "anthropic.claude-v1"},
    {"modelId": "anthropic.claude-v2"}
]


class TestModels:
    @pytest.fixture
    def mock_boto3_client(self):
        client = Mock()
        client.list_foundation_models.return_value = {"modelSummaries": MODELS}
        return client

    @pytest.fixture
    def models_resource(self, mock_boto3_client):
        models = Models(Mock())
        models._client.boto3_client = mock_boto3_client
        return models

    def test_list_models(self, models_resource, mock_boto3_client):
        result = models_resource.list()
        assert result == MODELS
        mock_boto3_client.list_foundation_models.assert_called_once_with(byInferenceType="ON_DEMAND")

    def test_retrieve_unique_model(self, models_resource):
        result = models_resource.retrieve("claude-v1")
        assert result == {"modelId": "anthropic.claude-v1"}

    def test_retrieve_latest_model(self, models_resource):
        result = models_resource.retrieve("claude")
        assert result == {"modelId": "anthropic.claude-v2"}

    def test_retrieve_model_not_found(self, models_resource):
        with pytest.raises(ValueError, match="Model 'nonexistent-model' cannot be found."):
            models_resource.retrieve("nonexistent-model")

    def test_retrieve_model_empty_input(self, models_resource):
        with pytest.raises(ValueError, match="Expected a non-empty value for `model`"):
            models_resource.retrieve("")
