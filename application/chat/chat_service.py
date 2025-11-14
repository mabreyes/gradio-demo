"""Chat service (Application Layer - orchestrates domain and infrastructure)."""
from typing import List, Tuple, Iterator
from domain.chat.entities import ChatMessage, ChatHistory
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
    
    def send_message(self, user_input: str) -> str:
        """Process user message and generate response.
        
        Args:
            user_input: The user's message
            
        Returns:
            Generated response from the model
        """
        # Create user message
        user_message = ChatMessage(content=user_input, role="user")
        self.chat_history.add_message(user_message)
        
        # Generate response using model provider
        response = self.model_provider.generate_response(
            user_input=user_input,
            conversation_history=self.chat_history.get_conversation_context()
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
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.chat_history.clear()
    
    def send_message_stream(self, user_input: str) -> Iterator[str]:
        """Process user message and generate streaming response.
        
        Args:
            user_input: The user's message
            
        Yields:
            Generated response tokens/text as they're produced
        """
        # Create user message and add to history
        user_message = ChatMessage(content=user_input, role="user")
        self.chat_history.add_message(user_message)
        
        # Generate streaming response using model provider
        full_response = ""
        for chunk in self.model_provider.generate_response_stream(
            user_input=user_input,
            conversation_history=self.chat_history.get_conversation_context()
        ):
            full_response += chunk
            yield chunk
        
        # Create assistant message with full response
        if full_response.strip():
            assistant_message = ChatMessage(content=full_response.strip(), role="assistant")
            self.chat_history.add_message(assistant_message)

