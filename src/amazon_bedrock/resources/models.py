from typing import Dict, List
from .api_resource import APIResource


class Models(APIResource):

    def retrieve(self, model: str) -> Dict:
        if not model:
            raise ValueError(f"Expected a non-empty value for `model` but received {model!r}")

        models = self.list()
        model_ids = [model_details["modelId"] for model_details in models if model in model_details["modelId"]]

        if len(model_ids) == 0:
            raise ValueError(f"Model {model!r} cannot be found.")
        else:
            return next(model for model in models if model["modelId"] == max(model_ids))

    def list(self) -> List[Dict]:
        kwargs = {"byInferenceType": "ON_DEMAND"}
        models = self._client.boto3_client.list_foundation_models(**kwargs)["modelSummaries"]
        return models
