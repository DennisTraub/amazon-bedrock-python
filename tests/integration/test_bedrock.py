import pytest

from amazon_bedrock import Bedrock


@pytest.mark.integration
class TestClient:

    @pytest.fixture
    def bedrock_client(self):
        return Bedrock(region="us-west-2")

    def test_aws_connection(self):
        try:
            client = Bedrock(region="us-west-2")
            response = client.boto3_client.list_foundation_models()
            assert response is not None
            assert 'modelSummaries' in response
        except Exception as e:
            pytest.fail(f"Failed to connect to AWS Bedrock: {str(e)}")

    def test_list_models(self, bedrock_client):
        models = bedrock_client.models.list()

        assert isinstance(models, list)
        assert len(models) > 0
        for model in models:
            assert isinstance(model, dict)
            assert ("modelId" in model)

    def test_retrieve_existing_model_id(self, bedrock_client):
        models = bedrock_client.models.list()
        existing_model_id = models[0]["modelId"]

        retrieved_model = bedrock_client.models.retrieve(existing_model_id)

        assert isinstance(retrieved_model, dict)
        assert "modelId" in retrieved_model
        assert existing_model_id == retrieved_model["modelId"]

    def test_retrieve_nonexistent_model(self, bedrock_client):
        nonexistent_model_id = "nonexistent-model"
        with pytest.raises(ValueError) as exc_info:
            bedrock_client.models.retrieve(nonexistent_model_id)
        assert f"Model {nonexistent_model_id!r} cannot be found." in str(exc_info.value)

    def test_retrieve_empty_model_id(self, bedrock_client):
        empty_model_id = ""
        with pytest.raises(ValueError) as exc_info:
            bedrock_client.models.retrieve(empty_model_id)
        assert "Expected a non-empty value for `model`" in str(exc_info.value)
