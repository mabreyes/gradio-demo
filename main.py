"""Main entry point for the Gradio chat application."""

import logging

from config.settings import settings
from infrastructure.models.huggingface_adapter import HuggingFaceModelAdapter
from presentation.gradio_interface import create_gradio_interface

logger = logging.getLogger(__name__)


def main() -> None:
    """Initialize dependencies and launch the Gradio application."""
    logger.info("Initializing model...")

    # Initialize model provider (Infrastructure layer)
    model_provider = HuggingFaceModelAdapter(settings=settings)
    model_provider.initialize()

    logger.info("Creating Gradio interface...")

    # Create Gradio interface (Presentation layer)
    gradio_interface = create_gradio_interface(
        model_provider=model_provider,
        settings=settings,
    )

    logger.info(
        "Launching Gradio interface on %s:%s",
        settings.GRADIO_SERVER_NAME,
        settings.GRADIO_SERVER_PORT,
    )

    # Launch interface
    gradio_interface.launch()


if __name__ == "__main__":
    main()
