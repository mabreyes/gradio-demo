"""Gradio interface (Presentation Layer - user interface)."""

import re
import time
from typing import Dict, Iterator, List, Optional, Tuple

import gradio as gr

from application.chat.chat_service import ChatService
from config.settings import Settings
from config.settings import settings as app_settings
from domain.chat.interfaces import IModelProvider


class GradioChatInterface:
    """Gradio chat interface following SRP - single responsibility for UI."""

    _CODE_BLOCK_PATTERN = re.compile(r"```([^\n]*)\n(.*?)(```)", re.DOTALL)

    def __init__(
        self,
        chat_service: ChatService,
        model_provider: Optional[IModelProvider] = None,
        settings: Settings = app_settings,
    ):
        """Initialize Gradio interface with chat service.

        Args:
            chat_service: Chat service instance
            model_provider: Optional model provider for advanced features
            settings: Application settings instance
        """
        self.chat_service = chat_service
        self.model_provider = model_provider
        self.settings = settings
        self.interface = None

    @staticmethod
    def _guess_code_language(code: str) -> str:
        """Best-effort guess of code language for syntax highlighting."""
        snippet = code.lower()
        if "public class" in code or "system.out" in snippet or "static void main" in snippet:
            return "java"
        if "def " in snippet or "import " in snippet or "print(" in snippet:
            return "python"
        if "#include" in snippet or "int main(" in snippet:
            return "cpp"
        if "<?php" in snippet:
            return "php"
        if "console.log" in snippet or "document." in snippet or "function " in snippet:
            return "javascript"
        return ""

    @classmethod
    def _normalize_code_blocks(cls, text: str) -> str:
        """Ensure fenced code blocks have a language for syntax highlighting."""

        def _replace(match: re.Match) -> str:
            lang = match.group(1).strip()
            code = match.group(2)

            # If language is already specified, keep as-is
            if lang:
                return match.group(0)

            guessed = cls._guess_code_language(code)
            if not guessed:
                guessed = "text"

            return f"```{guessed}\n{code}```"

        return cls._CODE_BLOCK_PATTERN.sub(_replace, text)

    def _chat_function(
        self,
        message: str,
        history: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[Tuple[str, List[Dict[str, str]]]]:
        """Handle chat message and update history with streaming.

        Args:
            message: User's message
            history: Current conversation history (list of dicts with 'role' and 'content')
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Yields:
            Tuple of (empty string, updated history) as response streams
        """
        if not message.strip():
            yield "", history
            return

        # Add user message to history immediately
        updated_history = history + [{"role": "user", "content": message}]

        # Initialize assistant message
        assistant_message = {"role": "assistant", "content": ""}
        updated_history = updated_history + [assistant_message]

        # Stream response tokens with a simple typewriter-style animation
        accumulated_response = ""
        for token in self.chat_service.send_message_stream(
            message, temperature=temperature, max_tokens=max_tokens
        ):
            for char in token:
                accumulated_response += char
                # Update the assistant message in history with a cursor-like indicator
                updated_history[-1] = {
                    "role": "assistant",
                    "content": f"{accumulated_response}▌",
                }
                yield "", updated_history
                # Small sleep to make the animation visible without slowing too much
                time.sleep(0.01)

        # Final update without the cursor once generation is complete
        if updated_history and updated_history[-1]["role"] == "assistant":
            normalized = self._normalize_code_blocks(accumulated_response)
            updated_history[-1] = {
                "role": "assistant",
                "content": normalized,
            }
            yield "", updated_history

    def _clear_history(self) -> Tuple[str, List[Dict[str, str]]]:
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
        with gr.Blocks(title="Gradio Chat Interface") as interface:
            # Header
            with gr.Column():
                gr.Markdown(
                    "# Gradio Chat Interface with Hugging Face\n"
                    "Chat with AI models powered by Hugging Face Transformers"
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
                        show_copy_button=True,
                    )

                    with gr.Row():
                        msg = gr.Textbox(
                            label="Your Message",
                            placeholder="Type your message here and press Enter...",
                            scale=4,
                            show_label=False,
                            lines=2,
                            max_lines=5,
                        )
                        submit_btn = gr.Button(
                            "Send", variant="primary", scale=1, size="lg"
                        )

                    with gr.Row():
                        clear_btn = gr.Button(
                            "Clear Conversation", variant="secondary", scale=1
                        )

                # Right column - Settings and Info
                with gr.Column(scale=1):
                    with gr.Accordion("Generation Parameters", open=False):
                        temperature_slider = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=self.settings.TEMPERATURE,
                            step=0.1,
                            label="Temperature",
                            info="Higher = more creative, Lower = more focused",
                        )

                        max_tokens_slider = gr.Slider(
                            minimum=50,
                            maximum=1024,
                            value=min(self.settings.MAX_LENGTH, 256),
                            step=50,
                            label="Max Tokens",
                            info="Maximum length of generated response",
                        )

                    with gr.Accordion("Model Information", open=False):
                        gr.Markdown(
                            f"""
                            **Current Model:** `{self.settings.MODEL_NAME}`\n
                            **Device:** CPU\n
                            **Chat Template:** {'Enabled' if self.settings.USE_CHAT_TEMPLATE else 'Disabled'}
                            """
                        )

                    with gr.Accordion("Statistics", open=False):
                        stats_display = gr.Markdown(
                            "**Conversation Stats:**\n- Messages: 0\n- Tokens: 0"
                        )

                    # Example prompts
                    with gr.Accordion("Example Prompts", open=False):
                        example_btn1 = gr.Button(
                            "Example 1: General Question",
                            size="sm",
                            variant="secondary",
                        )
                        example_btn2 = gr.Button(
                            "Example 2: Creative Task", size="sm", variant="secondary"
                        )
                        example_btn3 = gr.Button(
                            "Example 3: Code Help", size="sm", variant="secondary"
                        )

            # Footer
            gr.Markdown(
                "---\n"
                "<div style='text-align: center; color: #666; font-size: 0.9em;'>"
                "Powered by Hugging Face Transformers | Built with Gradio"
                "</div>"
            )

            # Event handlers with streaming support
            msg.submit(
                self._chat_function,
                inputs=[msg, chatbot, temperature_slider, max_tokens_slider],
                outputs=[msg, chatbot],
                show_progress="full",
            )

            submit_btn.click(
                self._chat_function,
                inputs=[msg, chatbot, temperature_slider, max_tokens_slider],
                outputs=[msg, chatbot],
                show_progress="full",
            )

            clear_btn.click(self._clear_history, inputs=[], outputs=[msg, chatbot])

            # Example prompts
            example_btn1.click(lambda: "What is artificial intelligence?", outputs=msg)

            example_btn2.click(
                lambda: "Write a short poem about technology.", outputs=msg
            )

            example_btn3.click(
                lambda: "Explain how to use Python decorators with an example.",
                outputs=msg,
            )

            # Update stats when chat updates
            def update_stats(history):
                """Compute a markdown summary of conversation statistics.

                Args:
                    history: Current chatbot history as a list of message
                        dictionaries.

                Returns:
                    str: Markdown text with message count and estimated token
                    usage.
                """
                message_count = len(history) if history else 0
                # Rough token estimate (4 chars per token)
                total_chars = (
                    sum(len(msg.get("content", "")) for msg in history)
                    if history
                    else 0
                )
                estimated_tokens = total_chars // 4
                return (
                    "**Conversation Stats:**\n"
                    f"- Messages: {message_count}\n"
                    f"- Estimated Tokens: {estimated_tokens}"
                )

            chatbot.change(update_stats, inputs=[chatbot], outputs=[stats_display])

        return interface

    def launch(
        self,
        share: Optional[bool] = None,
        server_name: Optional[str] = None,
        server_port: Optional[int] = None,
    ) -> None:
        """Launch the Gradio interface.

        Args:
            share: Whether to create a public link
            server_name: Server hostname
            server_port: Server port
        """
        if self.interface is None:
            self.interface = self.create_interface()

        self.interface.launch(
            share=share if share is not None else self.settings.GRADIO_SHARE,
            server_name=server_name or self.settings.GRADIO_SERVER_NAME,
            server_port=server_port or self.settings.GRADIO_SERVER_PORT,
        )


def create_gradio_interface(
    model_provider: IModelProvider,
    settings: Optional[Settings] = None,
) -> GradioChatInterface:
    """Factory function to create Gradio interface with dependencies.

    Args:
        model_provider: Model provider instance
        settings: Application settings instance

    Returns:
        Configured Gradio interface
    """
    configured_settings = settings or app_settings
    chat_service = ChatService(model_provider=model_provider)
    return GradioChatInterface(
        chat_service=chat_service,
        model_provider=model_provider,
        settings=configured_settings,
    )
