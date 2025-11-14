"""Gradio interface (Presentation Layer - user interface)."""
import gradio as gr
from typing import List, Dict, Tuple, Optional
from application.chat.chat_service import ChatService
from domain.chat.interfaces import IModelProvider
from config.settings import settings


class GradioChatInterface:
    """Gradio chat interface following SRP - single responsibility for UI."""
    
    def __init__(self, chat_service: ChatService, model_provider: Optional[IModelProvider] = None):
        """Initialize Gradio interface with chat service.
        
        Args:
            chat_service: Chat service instance
            model_provider: Optional model provider for advanced features
        """
        self.chat_service = chat_service
        self.model_provider = model_provider
        self.interface = None
    
    def _chat_function(
        self, 
        message: str, 
        history: List[Dict[str, str]], 
        temperature: float = None,
        max_tokens: int = None
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Handle chat message and update history.
        
        Args:
            message: User's message
            history: Current conversation history (list of dicts with 'role' and 'content')
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Returns:
            Tuple of (empty string, updated history)
        """
        if not message.strip():
            return "", history
        
        # Generate response with optional parameter overrides
        response = self.chat_service.send_message(
            message,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Update history with messages format (role and content)
        updated_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]
        
        return "", updated_history
    
    def _clear_history(self) -> Tuple[str, List]:
        """Clear conversation history.
        
        Returns:
            Tuple of (empty string, empty history)
        """
        self.chat_service.clear_history()
        return "", []
    
    def create_interface(self) -> gr.Blocks:
        """Create and configure Gradio interface with enhanced features.
        
        Returns:
            Configured Gradio Blocks interface
        """
        with gr.Blocks(
            title="Gradio Chat Interface", 
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .main-header {
                text-align: center;
                padding: 20px;
            }
            """
        ) as interface:
            # Header
            with gr.Column():
                gr.Markdown(
                    "# 🤖 Gradio Chat Interface with Hugging Face\n"
                    "Chat with AI models powered by Hugging Face Transformers",
                    elem_classes=["main-header"]
                )
            
            # Main content area
            with gr.Row():
                # Left column - Chat interface
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=600,
                        show_label=True,
                        type="messages",
                        avatar_images=(None, None),  # User, Bot avatars
                        show_copy_button=True
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Your Message",
                            placeholder="Type your message here and press Enter...",
                            scale=4,
                            show_label=False,
                            lines=2,
                            max_lines=5
                        )
                        submit_btn = gr.Button("Send", variant="primary", scale=1, size="lg")
                    
                    with gr.Row():
                        clear_btn = gr.Button("Clear Conversation", variant="secondary", scale=1)
                
                # Right column - Settings and Info
                with gr.Column(scale=1):
                    with gr.Accordion("⚙️ Generation Parameters", open=False):
                        temperature_slider = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=settings.TEMPERATURE,
                            step=0.1,
                            label="Temperature",
                            info="Higher = more creative, Lower = more focused"
                        )
                        
                        max_tokens_slider = gr.Slider(
                            minimum=50,
                            maximum=1024,
                            value=min(settings.MAX_LENGTH, 256),
                            step=50,
                            label="Max Tokens",
                            info="Maximum length of generated response"
                        )
                    
                    with gr.Accordion("ℹ️ Model Information", open=False):
                        model_info = gr.Markdown(
                            f"""
                            **Current Model:** `{settings.MODEL_NAME}`\n
                            **Device:** CPU\n
                            **Chat Template:** {'Enabled' if settings.USE_CHAT_TEMPLATE else 'Disabled'}
                            """
                        )
                    
                    with gr.Accordion("📊 Statistics", open=False):
                        stats_display = gr.Markdown("**Conversation Stats:**\n- Messages: 0\n- Tokens: 0")
                    
                    # Example prompts
                    with gr.Accordion("💡 Example Prompts", open=False):
                        example_btn1 = gr.Button("Example 1: General Question", size="sm", variant="secondary")
                        example_btn2 = gr.Button("Example 2: Creative Task", size="sm", variant="secondary")
                        example_btn3 = gr.Button("Example 3: Code Help", size="sm", variant="secondary")
            
            # Footer
            gr.Markdown(
                "---\n"
                "<div style='text-align: center; color: #666; font-size: 0.9em;'>"
                "Powered by Hugging Face Transformers | Built with Gradio"
                "</div>"
            )
            
            # Event handlers
            def chat_with_params(message, history, temp, max_tok):
                return self._chat_function(message, history, temp, max_tok)
            
            msg.submit(
                chat_with_params,
                inputs=[msg, chatbot, temperature_slider, max_tokens_slider],
                outputs=[msg, chatbot]
            )
            
            submit_btn.click(
                chat_with_params,
                inputs=[msg, chatbot, temperature_slider, max_tokens_slider],
                outputs=[msg, chatbot]
            )
            
            clear_btn.click(
                self._clear_history,
                inputs=[],
                outputs=[msg, chatbot]
            )
            
            # Example prompts
            example_btn1.click(
                lambda: "What is artificial intelligence?",
                outputs=msg
            )
            
            example_btn2.click(
                lambda: "Write a short poem about technology.",
                outputs=msg
            )
            
            example_btn3.click(
                lambda: "Explain how to use Python decorators with an example.",
                outputs=msg
            )
            
            # Update stats when chat updates
            def update_stats(history):
                message_count = len(history) if history else 0
                # Rough token estimate (4 chars per token)
                total_chars = sum(len(msg.get("content", "")) for msg in history) if history else 0
                estimated_tokens = total_chars // 4
                return f"**Conversation Stats:**\n- Messages: {message_count}\n- Estimated Tokens: {estimated_tokens}"
            
            chatbot.change(
                update_stats,
                inputs=[chatbot],
                outputs=[stats_display]
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
    return GradioChatInterface(chat_service=chat_service, model_provider=model_provider)

