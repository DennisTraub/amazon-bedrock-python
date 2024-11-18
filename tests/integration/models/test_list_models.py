import pytest
from typing import Dict, List


@pytest.mark.integration
class TestListModels:
    def test_list_models(self, bedrock_client) -> None:
        models = bedrock_client.models.list()

        assert isinstance(models, List)
        assert len(models) > 0
        for model in models:
            assert isinstance(model, Dict)
            assert ("modelId" in model)

    def test_retrieve_existing_model_id(self, bedrock_client) -> None:
        models = bedrock_client.models.list()
        existing_model_id = models[0]["modelId"]

        retrieved_model = bedrock_client.models.retrieve(existing_model_id)

        assert isinstance(retrieved_model, Dict)
        assert "modelId" in retrieved_model
        assert existing_model_id == retrieved_model["modelId"]

    def test_retrieve_nonexistent_model(self, bedrock_client) -> None:
        nonexistent_model_id = "nonexistent-model"

        with pytest.raises(ValueError) as exc_info:
            bedrock_client.models.retrieve(nonexistent_model_id)
        assert f"Model {nonexistent_model_id!r} cannot be found." in str(exc_info.value)

    def test_retrieve_empty_model_id(self, bedrock_client) -> None:
        empty_model_id = ""

        with pytest.raises(ValueError) as exc_info:
            bedrock_client.models.retrieve(empty_model_id)
        assert "Expected a non-empty value for `model`" in str(exc_info.value)
