"""Main entry point for the Gradio chat application."""

from config.settings import settings
from infrastructure.models.huggingface_adapter import HuggingFaceModelAdapter
from presentation.gradio_interface import create_gradio_interface


def main():
    """Main function to initialize and launch the application."""
    print("Initializing model...")

    # Initialize model provider (Infrastructure layer)
    model_provider = HuggingFaceModelAdapter(settings=settings)
    model_provider.initialize()

    print("Creating Gradio interface...")

    # Create Gradio interface (Presentation layer)
    gradio_interface = create_gradio_interface(
        model_provider=model_provider,
        settings=settings,
    )

    print("Launching Gradio interface...")
    print(f"Server: {settings.GRADIO_SERVER_NAME}:{settings.GRADIO_SERVER_PORT}")

    # Launch interface
    gradio_interface.launch()


if __name__ == "__main__":
    main()
