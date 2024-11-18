from src.amazon_bedrock import Bedrock
from pprint import pprint

client = Bedrock(region="us-west-2")

print("\n----- List models -----")
models = client.models.list()
model_ids = [model["modelId"] for model in models]
pprint(model_ids)

print("\n----- Retrieve model -----")
model_id = "mistral-large"
model = client.models.retrieve(model_id)
pprint(model)
