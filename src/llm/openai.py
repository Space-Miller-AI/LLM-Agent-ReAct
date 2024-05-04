from typing import Any, Dict, List, Optional, Tuple

from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, SystemMessage


class ChatOpenAIWithMessagePrefix(ChatOpenAI):
    prefix_messages: List[BaseMessage] = []

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        messages = self.prefix_messages + messages
        return super()._create_message_dicts(messages, stop)


class ChatOpenAIForInference(ChatOpenAIWithMessagePrefix):
    prefix_messages: List[BaseMessage] = [
        SystemMessage(
            content=(
                "You precisely follow instructions. "
                "Do not generate more output than necessary. "
                "At all times you strictly stick to the output format the user gives you."
            )
        )
    ]
