"""Gradio interface (Presentation Layer - user interface)."""
import gradio as gr
from typing import List, Dict, Tuple, Iterator
from application.chat.chat_service import ChatService
from domain.chat.interfaces import IModelProvider
from config.settings import settings


class GradioChatInterface:
    """Gradio chat interface following SRP - single responsibility for UI."""
    
    def __init__(self, chat_service: ChatService):
        """Initialize Gradio interface with chat service.
        
        Args:
            chat_service: Chat service instance
        """
        self.chat_service = chat_service
        self.interface = None
    
    def _chat_function(self, message: str, history: List[Dict[str, str]]) -> Iterator[Tuple[str, List[Dict[str, str]]]]:
        """Handle chat message and update history with streaming.
        
        Args:
            message: User's message
            history: Current conversation history (list of dicts with 'role' and 'content')
            
        Yields:
            Tuple of (empty string, updated history) as response streams
        """
        if not message.strip():
            yield "", history
            return
        
        # Add user message to history immediately (so it shows right away)
        updated_history = history + [{"role": "user", "content": message}]
        yield "", updated_history
        
        # Stream response and update history incrementally
        assistant_response = ""
        for chunk in self.chat_service.send_message_stream(message):
            assistant_response += chunk
            # Update history with partial response
            stream_history = updated_history + [{"role": "assistant", "content": assistant_response}]
            yield "", stream_history
        
        # Final yield with complete response
        final_history = updated_history + [{"role": "assistant", "content": assistant_response.strip()}]
        yield "", final_history
    
    def _clear_history(self) -> Tuple[str, List]:
        """Clear conversation history.
        
        Returns:
            Tuple of (empty string, empty history)
        """
        self.chat_service.clear_history()
        return "", []
    
    def create_interface(self) -> gr.Blocks:
        """Create and configure Gradio interface.
        
        Returns:
            Configured Gradio Blocks interface
        """
        with gr.Blocks(title="Gradio Chat Interface", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🤖 Gradio Chat Interface with Hugging Face")
            gr.Markdown("Chat with an AI model powered by Hugging Face Transformers")
            
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                show_label=True,
                type="messages"
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    scale=4,
                    show_label=False
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear", variant="secondary", scale=1)
            
            # Event handlers with streaming support
            # Use .then() for streaming - shows user message immediately, streams response
            def submit_message(message, history):
                """Submit message and stream response."""
                if not message.strip():
                    return "", history
                
                # Add user message immediately
                new_history = history + [{"role": "user", "content": message}]
                yield "", new_history
                
                # Stream assistant response
                assistant_response = ""
                for chunk in self.chat_service.send_message_stream(message):
                    assistant_response += chunk
                    stream_history = new_history + [{"role": "assistant", "content": assistant_response}]
                    yield "", stream_history
                
                # Final update
                final_history = new_history + [{"role": "assistant", "content": assistant_response.strip()}]
                yield "", final_history
            
            msg.submit(
                submit_message,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            submit_btn.click(
                submit_message,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            clear_btn.click(
                self._clear_history,
                inputs=[],
                outputs=[msg, chatbot]
            )
        
        return interface
    
    def launch(self, share: bool = None, server_name: str = None, server_port: int = None) -> None:
        """Launch the Gradio interface.
        
        Args:
            share: Whether to create a public link
            server_name: Server hostname
            server_port: Server port
        """
        if self.interface is None:
            self.interface = self.create_interface()
        
        self.interface.launch(
            share=share if share is not None else settings.GRADIO_SHARE,
            server_name=server_name or settings.GRADIO_SERVER_NAME,
            server_port=server_port or settings.GRADIO_SERVER_PORT
        )


def create_gradio_interface(model_provider: IModelProvider) -> GradioChatInterface:
    """Factory function to create Gradio interface with dependencies.
    
    Args:
        model_provider: Model provider instance
        
    Returns:
        Configured Gradio interface
    """
    chat_service = ChatService(model_provider=model_provider)
    return GradioChatInterface(chat_service=chat_service)

