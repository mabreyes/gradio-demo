"""Chat domain entities following DDD principles."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class ChatMessage:
    """Represents a single chat message (Domain Entity)."""

    content: str
    role: str  # 'user' or 'assistant'
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ChatHistory:
    """Represents a chat conversation history (Domain Entity)."""

    messages: List[ChatMessage]

    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the history."""
        self.messages.append(message)

    def get_conversation_context(self) -> List[ChatMessage]:
        """Get all messages as conversation context."""
        return self.messages.copy()

    def clear(self) -> None:
        """Clear all messages from history."""
        self.messages.clear()
