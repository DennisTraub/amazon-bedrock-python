import pytest
from typing import Generator, Tuple
from unittest.mock import Mock

from src.amazon_bedrock.resources import Messages
from src.amazon_bedrock.types import TextBlock


class TestMessages:
    MODEL_ID = "test.model_id"

    FIRST_USER_PROMPT = "Hello"
    SECOND_USER_PROMPT = "How are you?"

    FIRST_MODEL_RESPONSE = "Hi there!"

    MOCK_RESPONSE = {
        # "ResponseMetadata": {
        #     'HTTPHeaders': {
        #         'connection': 'keep-alive',
        #         'content-length': '253',
        #         'content-type': 'application/json',
        #         'date': 'Mon, 18 Nov 2024 16:59:43 GMT',
        #         'x-amzn-requestid': 'b3631d56-e5c0-4e1f-9b5f-556f3418471e'
        #     },
        #     'HTTPStatusCode': 200,
        #     'RequestId': 'b3631d56-e5c0-4e1f-9b5f-556f3418471e',
        #     'RetryAttempts': 0
        # },
        # "metrics": {
        #     'latencyMs': 828
        # },
        "output": {
            "message": {
                "content": [{
                    "text": FIRST_MODEL_RESPONSE
                }],
                "role": "assistant"
            }
        },
        # "stopReason": "end_turn",
        # "usage": {
        #     'inputTokens': 8,
        #     'outputTokens': 23,
        #     'totalTokens': 31
        # }
    }

    @pytest.fixture
    def get_mocks(self) -> Generator[Tuple[Mock, Mock], None, None]:
        mock_boto3_client = Mock()
        mock_boto3_client.converse.return_value = self.MOCK_RESPONSE
        mock_messages_resource = Messages(Mock())
        mock_messages_resource._client.bedrock_runtime_client = mock_boto3_client
        yield mock_messages_resource, mock_boto3_client

    def test_create_returns_expected_response(self, get_mocks) -> None:
        messages_resource, boto3_client = get_mocks

        response = messages_resource.create(
            model=self.MODEL_ID,
            messages=[{"role": "user", "content": self.FIRST_USER_PROMPT}]
        )

        assert response.role == "assistant"
        assert len(response.content) == 1
        assert response.content[0].type == "text"
        assert response.content[0].text == self.FIRST_MODEL_RESPONSE

    def test_create_accepts_simple_user_message(self, get_mocks) -> None:
        messages_resource, boto3_client = get_mocks

        messages_resource.create(
            model=self.MODEL_ID,
            messages=[{"role": "user", "content": self.FIRST_USER_PROMPT}]
        )

        boto3_client.converse.assert_called_once_with(
            modelId=self.MODEL_ID,
            messages=[{"role": "user", "content": [{"text": self.FIRST_USER_PROMPT}]}]
        )

    def test_create_accepts_text_block_user_message(self, get_mocks) -> None:
        messages_resource, boto3_client = get_mocks

        messages_resource.create(
            model=self.MODEL_ID,
            messages=[{
                "role": "user",
                "content": TextBlock(type="text", text=self.FIRST_USER_PROMPT)
            }]
        )

        boto3_client.converse.assert_called_once_with(
            modelId=self.MODEL_ID,
            messages=[{"role": "user", "content": [{"text": self.FIRST_USER_PROMPT}]}]
        )

    def test_create_accepts_multi_turn_simple_messages(self, get_mocks) -> None:
        messages_resource, boto3_client = get_mocks

        messages_resource.create(
            model=self.MODEL_ID,
            messages=[
                {"role": "user", "content":self.FIRST_USER_PROMPT},
                {"role": "assistant", "content": self.FIRST_MODEL_RESPONSE},
                {"role": "user", "content": "How are you?"}
            ]
        )

        boto3_client.converse.assert_called_once_with(
            modelId=self.MODEL_ID,
            messages=[
                {"role": "user", "content": [{"text": self.FIRST_USER_PROMPT}]},
                {"role": "assistant", "content": [{"text": self.FIRST_MODEL_RESPONSE}]},
                {"role": "user", "content": [{"text": "How are you?"}]}
            ]
        )

    def test_create_accepts_multi_turn_mixed_messages(self, get_mocks) -> None:
        messages_resource, boto3_client = get_mocks

        messages_resource.create(
            model=self.MODEL_ID,
            messages=[
                {"role": "user", "content": TextBlock(type="text", text=self.FIRST_USER_PROMPT)},
                {"role": "assistant", "content": TextBlock(type="text", text=self.FIRST_MODEL_RESPONSE)},
                {"role": "user", "content": self.SECOND_USER_PROMPT}
            ]
        )

        boto3_client.converse.assert_called_once_with(
            modelId=self.MODEL_ID,
            messages=[
                {"role": "user", "content": [{"text": self.FIRST_USER_PROMPT}]},
                {"role": "assistant", "content": [{"text": self.FIRST_MODEL_RESPONSE}]},
                {"role": "user", "content": [{"text": self.SECOND_USER_PROMPT}]}
            ]
        )
