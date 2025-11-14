"""Chat domain interfaces following DDD principles - dependency inversion."""

from abc import ABC, abstractmethod
from typing import Iterator, List

from domain.chat.entities import ChatMessage


class IModelProvider(ABC):
    """Interface for model providers (Domain Interface - Dependency Inversion Principle)."""

    @abstractmethod
    def generate_response(
        self,
        user_input: str,
        conversation_history: List[ChatMessage],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """Generate a response based on user input and conversation history.

        Args:
            user_input: The user's message
            conversation_history: Previous messages in the conversation
            temperature: Optional temperature override for this generation
            max_tokens: Optional max tokens override for this generation

        Returns:
            Generated response text
        """
        pass

    @abstractmethod
    def generate_response_stream(
        self,
        user_input: str,
        conversation_history: List[ChatMessage],
        temperature: float = None,
        max_tokens: int = None,
    ) -> Iterator[str]:
        """Generate a streaming response based on user input and conversation history.

        Args:
            user_input: The user's message
            conversation_history: Previous messages in the conversation
            temperature: Optional temperature override for this generation
            max_tokens: Optional max tokens override for this generation

        Yields:
            Generated response tokens/text as they're produced
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the model (load, prepare, etc.)."""
        pass
