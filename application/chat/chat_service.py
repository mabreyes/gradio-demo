"""Chat service (Application Layer - orchestrates domain and infrastructure)."""

from typing import Iterator, List, Tuple

from domain.chat.entities import ChatHistory, ChatMessage
from domain.chat.interfaces import IModelProvider


class ChatService:
    """Chat service following SRP - single responsibility for chat operations."""

    def __init__(self, model_provider: IModelProvider):
        """Initialize chat service with a model provider.

        Args:
            model_provider: Model provider implementing IModelProvider interface
        """
        self.model_provider = model_provider
        self.chat_history = ChatHistory(messages=[])

    def send_message(
        self, user_input: str, temperature: float = None, max_tokens: int = None
    ) -> str:
        """Process user message and generate response.

        Args:
            user_input: The user's message
            temperature: Optional temperature override for this generation
            max_tokens: Optional max tokens override for this generation

        Returns:
            Generated response from the model
        """
        # Create user message
        user_message = ChatMessage(content=user_input, role="user")
        self.chat_history.add_message(user_message)

        # Generate response using model provider
        response = self.model_provider.generate_response(
            user_input=user_input,
            conversation_history=self.chat_history.get_conversation_context(),
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Create assistant message
        assistant_message = ChatMessage(content=response, role="assistant")
        self.chat_history.add_message(assistant_message)

        return response

    def get_conversation_history(self) -> List[Tuple[str, str]]:
        """Get conversation history in format suitable for Gradio.

        Returns:
            List of tuples (user_message, assistant_message)
        """
        history = []
        current_user = None

        for msg in self.chat_history.messages:
            if msg.role == "user":
                current_user = msg.content
            elif msg.role == "assistant" and current_user is not None:
                history.append((current_user, msg.content))
                current_user = None

        return history

    def send_message_stream(
        self, user_input: str, temperature: float = None, max_tokens: int = None
    ) -> Iterator[str]:
        """Process user message and generate streaming response.

        Args:
            user_input: The user's message
            temperature: Optional temperature override for this generation
            max_tokens: Optional max tokens override for this generation

        Yields:
            Generated response tokens/text as they're produced
        """
        # Create user message
        user_message = ChatMessage(content=user_input, role="user")
        self.chat_history.add_message(user_message)

        # Generate streaming response using model provider
        accumulated_response = ""
        for token in self.model_provider.generate_response_stream(
            user_input=user_input,
            conversation_history=self.chat_history.get_conversation_context(),
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            accumulated_response += token
            yield token

        # Create assistant message with complete response
        if accumulated_response.strip():
            assistant_message = ChatMessage(
                content=accumulated_response.strip(), role="assistant"
            )
            self.chat_history.add_message(assistant_message)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.chat_history.clear()
