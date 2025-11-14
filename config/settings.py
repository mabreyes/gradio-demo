"""Application configuration settings."""

import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings following SRP - single responsibility for configuration."""

    # Model configuration
    # Recommended edge-optimized models:
    # - Qwen/Qwen2.5-0.5B-Instruct (smallest, fastest, ~1GB)
    # - Qwen/Qwen2.5-1.5B-Instruct (better quality, ~3GB)
    # - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (very fast, ~2.3GB)
    # - microsoft/Phi-2 (excellent quality, ~5GB)
    # - google/gemma-2b-it (good quality, ~5GB)
    MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    MAX_LENGTH: int = int(os.getenv("MAX_LENGTH", "512"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    USE_CHAT_TEMPLATE: bool = os.getenv("USE_CHAT_TEMPLATE", "True").lower() == "true"

    # Gradio configuration
    GRADIO_SERVER_NAME: str = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    GRADIO_SERVER_PORT: int = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    GRADIO_SHARE: bool = os.getenv("GRADIO_SHARE", "False").lower() == "true"

    # Hugging Face configuration
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
    USE_GPU: bool = os.getenv("USE_GPU", "False").lower() == "true"


settings = Settings()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
