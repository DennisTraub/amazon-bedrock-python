from src.amazon_bedrock import Bedrock

client = Bedrock(region="us-west-2")

response = client.messages.create(
    messages=[
        {
            "role": "user",
            "content": "Hello"
        }
    ],
    model="anthropic.claude-3-5-haiku-20241022-v1:0"
)

print(response)
