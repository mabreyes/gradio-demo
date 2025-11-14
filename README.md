# Gradio Chat Interface with Hugging Face

A production-ready Gradio chat interface using Hugging Face models, following Pythonic/idiomatic SRP (Single Responsibility Principle), DDD (Domain-Driven Design), and DRY (Don't Repeat Yourself) principles.

## Architecture

The project follows Domain-Driven Design (DDD) principles with clear separation of concerns:

- **Domain Layer** (`domain/`): Core business entities and interfaces
  - `domain/chat/entities.py`: Chat entities (ChatMessage, ChatHistory)
  - `domain/chat/interfaces.py`: Domain interfaces (IModelProvider)

- **Application Layer** (`application/`): Business logic and use cases
  - `application/chat/chat_service.py`: Chat service orchestrating domain and infrastructure

- **Infrastructure Layer** (`infrastructure/`): External adapters and implementations
  - `infrastructure/models/huggingface_adapter.py`: Hugging Face model adapter

- **Presentation Layer** (`presentation/`): User interface
  - `presentation/gradio_interface.py`: Gradio interface

- **Configuration** (`config/`): Application settings
  - `config/settings.py`: Configuration management

## Features

- ✅ Clean architecture with DDD principles
- ✅ Single Responsibility Principle (SRP) for each component
- ✅ Dependency Inversion Principle (interfaces and abstractions)
- ✅ DRY (Don't Repeat Yourself) - reusable components
- ✅ Docker and Docker Compose support
- ✅ Environment-based configuration
- ✅ Type hints and proper error handling

## Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose (for containerized deployment)
- Hugging Face account (optional, for private models)

## Setup

### Option 1: Using Virtual Environment (Local Development)

1. **Create and activate virtual environment:**
   ```bash
   chmod +x setup_venv.sh
   ./setup_venv.sh
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables (optional):**
   Create a `.env` file with the following variables:
   ```
   MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
   MAX_LENGTH=512
   TEMPERATURE=0.7
   USE_CHAT_TEMPLATE=True
   GRADIO_SERVER_NAME=0.0.0.0
   GRADIO_SERVER_PORT=7860
   GRADIO_SHARE=False
   HF_TOKEN=your_huggingface_token_here
   USE_GPU=False
   ```

4. **Run the application:**
   ```bash
   python main.py
   ```

5. **Access the interface:**
   Open your browser and navigate to `http://localhost:7860`

### Option 2: Using Docker Compose

1. **Create environment file (optional):**
   Create a `.env` file with the following variables:
   ```
   MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
   MAX_LENGTH=512
   TEMPERATURE=0.7
   USE_CHAT_TEMPLATE=True
   GRADIO_SHARE=False
   HF_TOKEN=your_huggingface_token_here
   USE_GPU=False
   ```

2. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

3. **Access the interface:**
   Open your browser and navigate to `http://localhost:7860`

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | Hugging Face model name/path | `Qwen/Qwen2.5-0.5B-Instruct` |
| `MAX_LENGTH` | Maximum response length | `512` |
| `TEMPERATURE` | Generation temperature | `0.7` |
| `USE_CHAT_TEMPLATE` | Use chat template for modern models | `True` |
| `GRADIO_SERVER_NAME` | Server hostname | `0.0.0.0` |
| `GRADIO_SERVER_PORT` | Server port | `7860` |
| `GRADIO_SHARE` | Create public Gradio link | `False` |
| `HF_TOKEN` | Hugging Face API token | `None` |
| `USE_GPU` | Use GPU for inference | `False` |

## Available Models

### Edge-Optimized Models (Recommended for Local/CPU)

The default model is optimized for edge devices and local machines:

**Recommended Models (sorted by size/performance):**

1. **Qwen/Qwen2.5-0.5B-Instruct** (Default) - ~1GB
   - Smallest and fastest
   - Good for edge devices
   - Excellent instruction following
   - Best for: Quick responses, low memory

2. **Qwen/Qwen2.5-1.5B-Instruct** - ~3GB
   - Better quality than 0.5B
   - Still very fast on CPU
   - Best for: Better quality responses

3. **TinyLlama/TinyLlama-1.1B-Chat-v1.0** - ~2.3GB
   - Very fast chat model
   - Optimized for conversations
   - Best for: Conversational chat

4. **microsoft/Phi-2** - ~5GB
   - Excellent quality for size
   - Very efficient
   - Best for: High-quality responses

5. **google/gemma-2b-it** - ~5GB
   - Google's instruction-tuned model
   - Good quality
   - Best for: General purpose

### Legacy Models

- `microsoft/DialoGPT-medium` - Older conversational model
- `microsoft/DialoGPT-large` - Larger conversational model
- `gpt2` - GPT-2 model

### Changing Models

To use a different model, set the `MODEL_NAME` environment variable:

```bash
# In .env file or docker-compose.yml
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
```

Or export it before running:
```bash
export MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
python main.py
```

### Model Features

- **Chat Template Support**: Modern instruction-tuned models (Qwen, Phi-2, Gemma) use chat templates for better formatting
- **CPU Optimization**: Models are optimized for CPU inference with low memory usage
- **Edge Device Support**: Small models work well on edge devices and local machines

## Project Structure

```
gradio-demo/
├── domain/                   # Domain layer
│   └── chat/
│       ├── entities.py       # Domain entities
│       └── interfaces.py     # Domain interfaces
├── application/              # Application layer
│   └── chat/
│       └── chat_service.py   # Chat service
├── infrastructure/           # Infrastructure layer
│   └── models/
│       └── huggingface_adapter.py  # Hugging Face adapter
├── presentation/             # Presentation layer
│   └── gradio_interface.py   # Gradio interface
├── config/                   # Configuration
│   └── settings.py           # Settings
├── main.py                   # Entry point
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker image definition
├── docker-compose.yml        # Docker Compose configuration
├── setup_venv.sh             # Virtual environment setup script
└── README.md                 # This file
```

## Development

### Running Tests

```bash
# Add your test files and run them
pytest
```

### Code Quality

The project follows Python best practices:
- Type hints for all functions
- Docstrings for all classes and methods
- Clear separation of concerns
- Dependency injection via interfaces

## Troubleshooting

### Model Loading Issues

If you encounter issues loading models:
1. Check your internet connection (models are downloaded from Hugging Face)
2. Verify the model name is correct
3. If using a private model, ensure `HF_TOKEN` is set correctly

### Memory Issues

If you run out of memory:
1. Use a smaller model (e.g., `microsoft/DialoGPT-small`)
2. Reduce `MAX_LENGTH` in settings
3. Use CPU instead of GPU (set `USE_GPU=False`)

### Docker Issues

If Docker build fails:
1. Ensure Docker is running
2. Check disk space
3. Verify Docker Compose version (3.8+)

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please follow the existing code structure and principles:
- Follow SRP, DDD, and DRY principles
- Add type hints and docstrings
- Keep components focused and single-purpose
- Maintain clear separation between layers

